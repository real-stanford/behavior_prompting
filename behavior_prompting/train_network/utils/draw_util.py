import os
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import imageio

from behavior_prompting.common.replay_buffer import ReplayBuffer

def vis_prompt(prompt, output_path):
    """Visualizes the RGB and trajectory from a prompt batch sampled from the DrawImageDataset with prompting enabled."""

    """Save RGB video stream with trajectory plot"""
    image = prompt['obs']['image'] # 0 to 1
    # Convert the numpy videos to uint8 format (0-255)
    image = (image * 255).astype('uint8')
    image = np.transpose(image, (0, 2, 3, 1)) # (T, H, W, C)

    # Define the output video file path and parameters
    height, width, _ = image[0].shape

    if len(prompt['action'].shape) == 3:
        actions = prompt['action'] # (T, chunk_size, action_dim)
    else:
        # TODO
        raise NotImplementedError

    T, chunk_size, action_dim = actions.shape
    total_frames = T * chunk_size
    # Flatten actions for plotting
    actions_flat = actions.reshape(-1, action_dim) # (T*chunk_size, action_dim)
    action_pos = actions_flat[:, :2]
    action_pos[:, 1] = 512 - action_pos[:, 1] # flip y-axis
    fps = 10

    # Create trajectory plot dimensions (make square)
    plot_size = height  # Make the plot square with the same height as the video
    plot_width = plot_size
    plot_height = plot_size

    # Combined video dimensions
    combined_width = width + plot_width
    combined_height = max(height, plot_height)

    video_writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

    # Write each frame to the video file
    for t in range(total_frames):
        # Select the image for this chunk
        image_idx = t // chunk_size
        frame = image[image_idx]

        # Create trajectory plot for current timestep (square)
        fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100), dpi=100)

        # Plot full trajectory up to current timestep
        if t > 0:
            ax.plot(action_pos[:t+1, 0], action_pos[:t+1, 1], 'b-', alpha=0.5, linewidth=2)

        # Plot current position
        ax.scatter(action_pos[t, 0], action_pos[t, 1], c='red', s=100, zorder=5)

        # Set fixed axis limits for consistent scale
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)

        # Ensure equal aspect ratio for x and y axes
        ax.set_aspect('equal', adjustable='box')

        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Actions (t={t})')
        ax.grid(True, alpha=0.3)

        # Convert plot to image
        fig.tight_layout()
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = plot_img[..., :3]  # Drop alpha channel to get RGB
        plt.close(fig)

        # Resize plot image to match square size
        plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))

        # Combine RGB frame with trajectory plot
        # If frame height != plot_height, pad frame to match
        if frame.shape[0] != plot_height:
            pad_vert = plot_height - frame.shape[0]
            if pad_vert > 0:
                pad_top = pad_vert // 2
                pad_bottom = pad_vert - pad_top
                frame_padded = np.pad(frame, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
            else:
                frame_padded = frame[:plot_height, :, :]
        else:
            frame_padded = frame
        combined_frame = np.hstack([frame_padded, plot_img_resized])

        # Write combined frame
        video_writer.append_data(combined_frame)

    # Release the video writer
    video_writer.close()

    print(f"Saved prompt video to {output_path}")

def get_target_drawing_image_for_task_idx(replay_buffer: ReplayBuffer, task_idx: int) -> Tuple[np.ndarray, float]:
    """The target drawing image is the last image of the task in the replay buffer"""
    task_data_end = replay_buffer.task_data_ends[task_idx]
    target_drawing = replay_buffer.labels['drawing_image'][task_data_end-1]
    boundary_angle = replay_buffer.labels['boundary_angle'][task_data_end-1].item()

    return target_drawing, boundary_angle

def get_target_drawing_image_first_demonstration(replay_buffer: ReplayBuffer, task_name: str) -> np.ndarray:
    # Find first episode containing this task
    demonstration_index = np.where(replay_buffer.task_names[:] == task_name)[0][0]
    
    return get_target_drawing_image_for_task_idx(replay_buffer, demonstration_index)

def compute_draw_distance_points(A, B, max_x, max_y, center=False, debug_dir=None, image_name=None):
    if center:
        # === Center on centroid === #
        # Find centroid of A and B
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)

        # Center A and B on their respective centroids
        image_center = np.array([max_y // 2, max_x // 2]) #(y, x)
        A = (A - centroid_A) + image_center
        B = (B - centroid_B) + image_center
        raise NotImplementedError('is converting to int reasonable here? Might mess up the chamfer distance since it should probably stay as float')
        A = A.astype(int)
        B = B.astype(int)

    # Chamfer distance (symmetric)
    if len(A) > 0 and len(B) > 0:
        tree = KDTree(B)
        dist_A = tree.query(A)[0]
        tree = KDTree(A)
        dist_B = tree.query(B)[0]
        chamfer_distance = np.mean(dist_A) + np.mean(dist_B)
    else:
        chamfer_distance = float('inf')  # Return infinity if no white pixels found

    return chamfer_distance

def compute_draw_distance(mask_target, mask_rollout, center=False, debug_dir=None, image_name=None):
    """Given two binary masks (bool arrays) where false indicates no drawing and true indicates drawing, compute the chamfer distance between the two masks by first optionally centering the masks on their centroids. This is the faster, but less accurate, version of the function. Centering does have some issues: for example drawing letter T and only drawing top bar will center the top bar instead of aligning it with the top bar of the letter T. Thus it's disabled by default."""
    # Get coordinates of white pixels
    A = np.argwhere(mask_target) # returns list of coordiantes (y, x) corresponding to white pixel
    B = np.argwhere(mask_rollout)

    max_y, max_x = mask_target.shape

    chamfer_distance = compute_draw_distance_points(A, B, max_x, max_y, center, debug_dir, image_name)

     # Save centered images if debug directory is provided
    if debug_dir and image_name:
        uncentered_ims = np.hstack([mask_target, mask_rollout])

        if center:
            # Create centered images by placing points at their centered positions
            centered_img1 = np.zeros_like(mask_target)
            centered_img2 = np.zeros_like(mask_rollout)

            # Set black pixels (0 = black)
            for y, x in A:
                if 0 <= y < centered_img1.shape[0] and 0 <= x < centered_img1.shape[1]:
                    centered_img1[y, x] = True

            for y, x in B:
                if 0 <= y < centered_img2.shape[0] and 0 <= x < centered_img2.shape[1]:
                    centered_img2[y, x] = True

            # Stack side by side
            centered_ims = np.hstack([centered_img1, centered_img2])
            concat_img = np.vstack([uncentered_ims, centered_ims])
        else:
            concat_img = uncentered_ims
        concat_img = concat_img.astype(np.uint8) * 255

        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}.png"), concat_img)
    
    return chamfer_distance

def compute_draw_distance_translation(mask_target, mask_rollout, debug_dir=None, image_name=None, step_size=None):
    """Given two binary masks (bool arrays) where false indicates no drawing and true indicates drawing, compute the minimum chamfer distance between the two masks by trying all possible xy translations of the rollout image. This is the slower, but more accurate, version of the function."""
    
    if step_size is None:
        step_size = mask_target.shape[0] // 50

    # Get coordinates of white pixels
    A = np.argwhere(mask_target) # returns list of coordinates (y, x) corresponding to white pixel
    B = np.argwhere(mask_rollout)

    if len(A) == 0 or len(B) == 0:
        return float('inf')  # Return infinity if no white pixels found

    # Get image dimensions
    height, width = mask_target.shape
    
    # Compute bounding boxes for A and B
    if len(A) > 0:
        A_min_y, A_min_x = A.min(axis=0)
        A_max_y, A_max_x = A.max(axis=0)
        A_bbox = (A_min_y, A_min_x, A_max_y, A_max_x)
    else:
        A_bbox = (0, 0, 0, 0)
    
    if len(B) > 0:
        B_min_y, B_min_x = B.min(axis=0)
        B_max_y, B_max_x = B.max(axis=0)
        B_bbox = (B_min_y, B_min_x, B_max_y, B_max_x)
    else:
        B_bbox = (0, 0, 0, 0)
    
    # Compute the intersection of bounding boxes to determine translation range
    # We want to translate B so that its bounding box overlaps with A's bounding box
    A_width = A_max_x - A_min_x
    A_height = A_max_y - A_min_y
    B_width = B_max_x - B_min_x
    B_height = B_max_y - B_min_y
    
    # Translation range: B should be translated so that its bounding box can overlap with A's
    # This means B's translated bounding box should intersect with A's bounding box
    min_dy = A_min_y - B_max_y  # B's top edge should be at least at A's top edge
    max_dy = A_max_y - B_min_y  # B's bottom edge should be at most at A's bottom edge
    min_dx = A_min_x - B_max_x  # B's left edge should be at least at A's left edge
    max_dx = A_max_x - B_min_x  # B's right edge should be at most at A's right edge
    
    # Ensure we stay within image bounds
    min_dy = max(min_dy, -height)
    max_dy = min(max_dy, height)
    min_dx = max(min_dx, -width)
    max_dx = min(max_dx, width)
    
    # Try translations within the computed range
    min_chamfer_distance = float('inf')
    best_translation = None
    best_A = None
    best_B = None
    
    for dy in range(min_dy, max_dy + 1, step_size):
        for dx in range(min_dx, max_dx + 1, step_size):
            # Apply translation to B
            B_translated = B + np.array([dy, dx])
            
            # Filter points that are within image bounds
            valid_mask = (B_translated[:, 0] >= 0) & (B_translated[:, 0] < height) & \
                        (B_translated[:, 1] >= 0) & (B_translated[:, 1] < width)
            B_valid = B_translated[valid_mask]
            
            if len(B_valid) == 0:
                continue
                
            # Compute chamfer distance for this translation
            tree = KDTree(B_valid)
            dist_A = tree.query(A)[0]
            tree = KDTree(A)
            dist_B = tree.query(B_valid)[0]
            chamfer_distance = np.mean(dist_A) + np.mean(dist_B)
            
            # Update minimum if this translation is better
            if chamfer_distance < min_chamfer_distance:
                min_chamfer_distance = chamfer_distance
                best_translation = (dy, dx)
                best_A = A.copy()
                best_B = B_valid.copy()
    
    # Save debug images if debug directory is provided and we found a valid translation
    if debug_dir and image_name and best_translation is not None:
        # Create images showing the optimal translation
        optimal_img1 = np.zeros_like(mask_target)
        optimal_img2 = np.zeros_like(mask_rollout)

        # Set white pixels for the optimally translated positions
        for y, x in best_A:
            if 0 <= y < optimal_img1.shape[0] and 0 <= x < optimal_img1.shape[1]:
                optimal_img1[y, x] = True

        for y, x in best_B:
            if 0 <= y < optimal_img2.shape[0] and 0 <= x < optimal_img2.shape[1]:
                optimal_img2[y, x] = True

        # Stack side by side: original images and optimally translated images
        original_ims = np.hstack([mask_target, mask_rollout])
        optimal_ims = np.hstack([optimal_img1, optimal_img2])
        concat_img = np.vstack([original_ims, optimal_ims])
        concat_img = concat_img.astype(np.uint8) * 255

        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{image_name}.png"), concat_img)

    return min_chamfer_distance
