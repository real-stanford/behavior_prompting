from typing import Optional
import matplotlib
import torch

from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import PairPromptObsEncoder
from behavior_prompting.train_network.utils.attention_vis_v1_util import plot_prompt_obs, get_prompt_frame_images

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import cv2
import io
import os
import imageio
import logging

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)

def vis_attention_map_receding_obs_with_prompt(output_path, attention_map, receding_obs_names, prompt_obs_names, fps, exec_action_horizon, prompt, prompt_img_keys, max_frames=None, num_cols=6, stack_prompt_imgs: str="off", cmap: str = "GnBu", flip_cmap: bool = False, scale_attention_weights: bool = False, use_proportion_of_colormap: float = 1.0):
    """
    Visualize the attention map as a video with colored borders on prompt frames. For use in policies where prompt is cross attended to the receding obs.

    Args:
        output_path (str): The path to save the attention map video. Should end with .mp4 or .gif.
        attention_map (numpy.ndarray): The attention map to visualize. (num_inference_steps, num_layers, num_receding_obs, num_prompt_obs)
        receding_obs_names (list): The names of the receding obs.
        prompt_obs_names (list): The names of the prompt obs.
        fps (int): Frames per second for the output video.
        exec_action_horizon (int): Number of action steps per inference step.
        prompt (dict): Dictionary containing image tensors for each key in prompt_img_keys. Each tensor is (T, 3, H, W) and dtype float32.
        prompt_img_keys (list): List of keys to extract images from the prompt.
        max_frames (int, optional): Maximum number of frames to render. If None, use all frames.
        num_cols (int, optional): Number of columns in the grid layout. Default is 6.
        stack_prompt_imgs (str, optional): How to combine multiple prompt image keys per timestep.
            "horizontal": stack side by side. "vertical": stack top to bottom.
            "off" (default): only the first prompt image key is visualized.
        cmap (str, optional): Colormap to use for attention score borders. Default is "GnBu".
        flip_cmap (bool, optional): If True, flip the colormap (use 1 - attention_score). Default is False.
        scale_attention_weights (bool, optional): If True, scale attention weights to [0, 1] range after processing. Default is False.
        use_proportion_of_colormap (float, optional): Use only this proportion of the colormap range (0, 1]. Default is 1.0.
    """
    prompt_img_keys = [x for x in prompt_img_keys if x in prompt['obs'].keys()]
    attention_map = attention_map[..., :len(prompt_obs_names)]

    num_inference_steps, num_layers, num_receding_obs, num_prompt_obs = attention_map.shape

    attention_map = attention_map[:, num_layers//2] # (num_inference_steps, num_receding_obs, num_prompt_obs)
    
    # Collapse num_receding_obs dimension by taking the mean
    attention_map = attention_map.mean(axis=1) # (num_inference_steps, num_prompt_obs)
    
    # Compute vmin/vmax for colormap to use full range (but keep actual values)
    if scale_attention_weights:
        vmin = 0.0  # Always set minimum to 0
        vmax = attention_map.max()
        if vmax <= vmin:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Use only a proportion of the colormap so that max value maps to that proportion
    vmax_display = vmax / use_proportion_of_colormap

    num_inference_steps = attention_map.shape[0]
    if max_frames is not None:
        num_frames = min(num_inference_steps, max_frames)
    else:
        num_frames = num_inference_steps

    # Get the number of timesteps (T) from the first key
    T = prompt['obs'][prompt_img_keys[0]].shape[0]
    
    # Prepare prompt frames
    stacked_imgs = []
    widths = []
    for t in range(T):
        if stack_prompt_imgs in ("horizontal", "vertical"):
            imgs_at_t = [np.transpose(prompt['obs'][key][t], (1, 2, 0)) for key in prompt_img_keys]  # (3, H, W) -> (H, W, 3)
            stack_fn = np.hstack if stack_prompt_imgs == "horizontal" else np.vstack
            stacked_img = stack_fn(imgs_at_t)
        else:
            # "off": use only the first prompt image key
            key = prompt_img_keys[0]
            stacked_img = np.transpose(prompt['obs'][key][t], (1, 2, 0))  # (3, H, W) -> (H, W, 3)
        # Handle float32 images: scale to [0, 255] and convert to uint8
        if stacked_img.dtype == np.float32 or stacked_img.dtype == np.float64:
            if stacked_img.max() <= 1.0:
                stacked_img = np.clip(stacked_img, 0, 1)
                stacked_img = (stacked_img * 255).round().astype(np.uint8)
            else:
                stacked_img = np.clip(stacked_img, 0, 255).astype(np.uint8)
        stacked_imgs.append(stacked_img)
        widths.append(stacked_img.shape[1])
    
    # Get colormap for attention scores
    colormap = plt.colormaps[cmap]
    # vmax_display: max data value maps to use_proportion_of_colormap of the colormap

    def add_colored_border(img, attention_score):
        """Add a colored border around an image based on attention score."""
        h, w = img.shape[:2]
        # Calculate border width as 10% of the smaller dimension
        border_width = int(min(h, w) * 0.10)
        
        # Get color from colormap (normalize so max value uses use_proportion_of_colormap of the colormap)
        normalized_score = (attention_score - vmin) / (vmax_display - vmin) if vmax_display > vmin else 0.5
        if flip_cmap:
            normalized_score = 1.0 - normalized_score
        rgba = np.array(colormap(np.clip(normalized_score, 0, 1)))
        rgb = (rgba[:3] * 255).astype(np.uint8)
        
        # Create a larger image with border
        bordered_img = np.full((h + 2 * border_width, w + 2 * border_width, 3), rgb, dtype=np.uint8)
        
        # Place original image in the center
        bordered_img[border_width:border_width + h, border_width:border_width + w] = img
        
        return bordered_img
    
    # Create video writer
    if output_path.endswith('.mp4'):
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    elif output_path.endswith('.gif'):
        writer = imageio.get_writer(output_path, fps=fps)
    else:
        raise ValueError('Output path must end with .mp4 or .gif')
    
    # Generate frames
    for step in tqdm(range(num_frames), desc='Rendering frames', leave=False, total=num_frames):
        # Get attention scores for this step (num_prompt_obs,)
        step_attention = attention_map[step]
        
        # Create bordered prompt frames
        bordered_frames = []
        for t in range(T):
            # Get attention score for this prompt timestep
            attn_score = step_attention[t]
            
            # Add colored border to the frame
            bordered_frame = add_colored_border(stacked_imgs[t].copy(), attn_score)
            bordered_frames.append(bordered_frame)
        
        # Arrange frames in a grid with specified number of columns
        if bordered_frames:
            # Get the width of a single frame to calculate target row width
            single_frame_width = bordered_frames[0].shape[1]
            target_row_width = single_frame_width * num_cols
        
        rows = []
        for i in range(0, len(bordered_frames), num_cols):
            row_frames = bordered_frames[i:i + num_cols]
            row = np.hstack(row_frames)
            
            # Pad row to target width if it's shorter (last row might have fewer frames)
            if row.shape[1] < target_row_width:
                pad_width = target_row_width - row.shape[1]
                pad = np.ones((row.shape[0], pad_width, 3), dtype=row.dtype) * 255  # White padding
                row = np.hstack([row, pad])
            
            rows.append(row)
        
        # Stack all rows vertically
        frame = np.vstack(rows)
        
        # Write frame
        writer.append_data(frame)
    
    writer.close()


def combine_attention_and_rollout_video(rollout_video_path, output_path, prompt, prompt_img_keys, exec_action_horizon, steps_per_render, prompt_attention_map_path: Optional[str] = None, rollout_width_fraction: float = 1/3):
    """
    Combine the attention map video and the rollout video into a single video, with the prompt observation image below each frame.
    """
    assert prompt_attention_map_path is not None, 'prompt_attention_map_path must be provided'
    
    rollout_cap = cv2.VideoCapture(rollout_video_path)
    roll_width = int(rollout_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roll_height = int(rollout_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roll_frames = int(rollout_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roll_fps = rollout_cap.get(cv2.CAP_PROP_FPS)
    
    # Attention map video (assumed to always be present)
    prompt_attention_map_cap = cv2.VideoCapture(prompt_attention_map_path)
    prompt_attn_width = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prompt_attn_height = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prompt_attn_fps = prompt_attention_map_cap.get(cv2.CAP_PROP_FPS)
    prompt_attn_frames = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine output width such that:
    # - Rollout is not downscaled (target_rollout_width >= roll_width)
    # - Attention visualization is not downscaled (out_width >= prompt_attn_width)
    # - Rollout occupies rollout_width_fraction of the total width
    out_width = max(prompt_attn_width, int(roll_width / rollout_width_fraction))

    # Compute rollout size (rollout_width_fraction of total width, but never smaller than original)
    target_rollout_width = int(out_width * rollout_width_fraction)
    if target_rollout_width < roll_width:
        # This should not happen given out_width definition, but guard just in case
        target_rollout_width = roll_width
        out_width = max(out_width, int(roll_width / rollout_width_fraction))
    roll_scale = target_rollout_width / roll_width
    resized_roll_width = target_rollout_width
    resized_roll_height = int(roll_height * roll_scale)

    # Compute attention size (never smaller than original)
    attn_scale = out_width / prompt_attn_width
    resized_attn_width = out_width
    resized_attn_height = int(prompt_attn_height * attn_scale)

    # Final output height: rollout on top, attention below
    out_height = resized_roll_height + resized_attn_height
    out_fps = roll_fps * steps_per_render # the video is written out at the actions frequency. If steps_per_render > 1, then we will just repeat rollout frames as needed
    out = imageio.get_writer(output_path, fps=out_fps, codec='libx264')

    num_frames = roll_frames * steps_per_render

    # Read and combine frames
    for action_idx in tqdm(range(num_frames), desc='Combining attention, rollout, and prompt obs', leave=False):
        # Only read new rollout frame every steps_per_render frames
        if action_idx % steps_per_render == 0:
            ret_roll, roll_frame = rollout_cap.read()
            if not ret_roll:
                break
            current_roll_frame = roll_frame

        # Read prompt attention map at diffusion frequency
        if action_idx % exec_action_horizon == 0:
            ret_prompt_attn, prompt_attn_frame = prompt_attention_map_cap.read()
            if not ret_prompt_attn:
                break
        
        # Resize rollout frame to the target rollout size
        resized_roll = cv2.resize(current_roll_frame, (resized_roll_width, resized_roll_height))
        
        # Convert BGR to RGB for rollout frame
        resized_roll_rgb = resized_roll[...,::-1]

        # Center the rollout horizontally with white padding on sides
        padding_width = max(0, (out_width - resized_roll_width) // 2)
        white_pad = np.ones((resized_roll_height, padding_width, 3), dtype=np.uint8) * 255
        centered_rollout = np.hstack([white_pad, resized_roll_rgb, white_pad])
        # Handle case where padding doesn't divide evenly
        if centered_rollout.shape[1] < out_width:
            extra_pad = np.ones((resized_roll_height, out_width - centered_rollout.shape[1], 3), dtype=np.uint8) * 255
            centered_rollout = np.hstack([centered_rollout, extra_pad])

        # Stack rollout on top, attention on bottom
        # Resize attention frame (upscale only) and convert from BGR to RGB
        resized_attn_frame = cv2.resize(prompt_attn_frame, (resized_attn_width, resized_attn_height))
        prompt_attn_frame_rgb = resized_attn_frame[...,::-1]
        cur_frame = np.vstack([centered_rollout, prompt_attn_frame_rgb])
        
        out.append_data(cur_frame)

    # Release everything
    prompt_attention_map_cap.release()
    rollout_cap.release()
    out.close()


def vis_attention_and_rollout_v2(out_path: str, rollout_video_path: str, exec_action_horizon: int, policy: BasePolicy, prompt_for_vis, fps: int, rgb_keys: list[str], steps_per_render: int, prompt_cross_attn_weights: Optional[torch.Tensor]=None, diffusion_cross_attn_weights: Optional[torch.Tensor]=None, max_frames=None, num_cols=6, cmap="GnBu", scale_attention_weights: bool = False, use_proportion_of_colormap: float = 1.0, stack_prompt_imgs: str = "horizontal", rollout_width_fraction: float = 1/3):
    """Visualizes attention maps overlayed on top of the prompt frames."""
    prompt_len = prompt_for_vis['action'].shape[0]

    if prompt_cross_attn_weights is not None:
        # prompt_cross_attn_weights: (num_inference_steps, num_layers, receding_obs_len, prompt_obs_len)
        prompt_attention_map_path = os.path.join(os.path.dirname(out_path), f"tmp_{os.path.splitext(os.path.basename(out_path))[0]}_attention_map_prompt.mp4")
        obs_encoder = policy.obs_encoder
        assert isinstance(obs_encoder, PairPromptObsEncoder)
        prompt_current_obs_names, prompt_obs_names = obs_encoder.get_prompt_cross_attn_dim_names(prompt_len)
        vis_attention_map_receding_obs_with_prompt(prompt_attention_map_path, prompt_cross_attn_weights, prompt_current_obs_names, prompt_obs_names, fps, exec_action_horizon, prompt_for_vis, rgb_keys, max_frames, num_cols, stack_prompt_imgs=stack_prompt_imgs, cmap=cmap, scale_attention_weights=scale_attention_weights, use_proportion_of_colormap=use_proportion_of_colormap)
    else:
        prompt_attention_map_path = None

    combine_attention_and_rollout_video(rollout_video_path, out_path, prompt_for_vis, rgb_keys, exec_action_horizon, steps_per_render, prompt_attention_map_path, rollout_width_fraction=rollout_width_fraction)

    if prompt_attention_map_path is not None:
        os.remove(prompt_attention_map_path)


def save_prompt_attention_artifacts(
    save_weights_base_path: str,
    prompt_cross_attn_weights: np.ndarray,
    rollout_video_path: str,
    prompt_for_vis,
    rgb_keys: list[str],
    exec_action_horizon: int,
    steps_per_render: int,
    policy: BasePolicy,
    cmap: str = "GnBu",
    flip_cmap: bool = False,
    scale_attention_weights: bool = False,
    include_colorbar_in_plots: bool = True,
    use_proportion_of_colormap: float = 1.0,
    fps: Optional[int] = None,
) -> None:
    """
    Save prompt-cross attention weights as per-timestep heatmap PNGs, the corresponding
    rollout frames, and a prompt visualization.

    This uses only prompt_cross_attn_weights and mirrors the receding-obs-with-prompt
    heatmap style from attention_vis_v1_util.

    Args:
        cmap: Colormap to use for heatmaps. Default is "GnBu".
        flip_cmap: If True, flip the colormap. Default is False.
        scale_attention_weights: If True, scale attention weights to [0, 1] range after processing. Default is False.
    """
    # Directories
    weights_dir = os.path.join(save_weights_base_path, "weights")
    weights_flattened_dir = os.path.join(save_weights_base_path, "weights_flattened")
    rollout_dir = os.path.join(save_weights_base_path, "rollout")
    prompt_dir = os.path.join(save_weights_base_path, "prompt")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(weights_flattened_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    # Save prompt visualization (single image)
    prompt_img_path = os.path.join(prompt_dir, "prompt_obs.png")
    plot_prompt_obs(prompt_for_vis, prompt_img_path, rgb_keys, add_titles=False)

    prompt_frames = get_prompt_frame_images(prompt_for_vis, rgb_keys)
    for idx, frame in enumerate(prompt_frames):
        frame_path = os.path.join(prompt_dir, f"prompt_frame_{idx:04d}.png")
        imageio.imwrite(frame_path, frame)

    # Derive attention labels
    prompt_len = prompt_for_vis["action"].shape[0]
    obs_encoder = policy.obs_encoder
    assert isinstance(obs_encoder, PairPromptObsEncoder)
    prompt_current_obs_names, prompt_obs_names = obs_encoder.get_prompt_cross_attn_dim_names(prompt_len)

    # Shape: (T, num_layers, num_prompt_current_obs, num_prompt_obs)
    attention_map = prompt_cross_attn_weights[..., : len(prompt_obs_names)]
    num_inference_steps, num_layers, num_prompt_current_obs, num_prompt_obs = attention_map.shape

    # Use middle layer for visualization, same as v1 util - select it once at the start
    mid_layer = num_layers // 2
    attention_map_mid_layer = attention_map[:, mid_layer]  # (num_inference_steps, num_receding_obs, num_prompt_obs)

    # Compute vmin/vmax for per-timestep heatmaps (from full 2D attention map)
    if scale_attention_weights:
        vmin_per_timestep = 0.0  # Always set minimum to 0
        vmax_per_timestep = attention_map_mid_layer.max()
        if vmax_per_timestep <= vmin_per_timestep:
            vmax_per_timestep = 1.0
    else:
        vmin_per_timestep, vmax_per_timestep = 0.0, 1.0

    # Save per-timestep heatmaps
    for step in range(num_inference_steps):
        attn = attention_map_mid_layer[step]  # (receding_obs_len, prompt_obs_len)

        # Set figure size: 10px per square entry
        n_rows, n_cols = attn.shape
        fig, ax = plt.subplots(figsize=(n_cols / 5, n_rows / 5))
        
        hm = sns.heatmap(
            attn,
            ax=ax,
            cmap=cmap,
            cbar=include_colorbar_in_plots,
            vmin=vmin_per_timestep,
            vmax=vmax_per_timestep / use_proportion_of_colormap,
            xticklabels=False,
            yticklabels=False,
            square=True,  # enforce square cells
        )
        
        # Customize colorbar if present
        if include_colorbar_in_plots:
            cbar = hm.collections[0].colorbar
            # Set only 3 ticks: bottom, middle, top
            tick_values = [vmin_per_timestep, (vmin_per_timestep + vmax_per_timestep) / 2, vmax_per_timestep]
            cbar.set_ticks(tick_values)
            # Format tick labels to 2 decimal places
            cbar.ax.set_yticklabels([f'{val:.2f}' for val in tick_values])
            # Remove tick marks, keep labels only
            cbar.ax.tick_params(length=0)
            cbar.ax.set_ylabel("x", rotation=0, ha='left', va='center')
            cbar.ax.set_xlabel("y")
        
        ax.set_aspect("equal")
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)

        # Save both SVG and PNG versions
        svg_path = os.path.join(weights_dir, f"weights_{step:04d}.svg")
        png_path = os.path.join(weights_dir, f"weights_{step:04d}.png")
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # Save flattened weights (mean across receding_obs dimension)
    # Take mean first, then compute vmin/vmax separately to match video output (which computes after mean)
    attention_map_flattened = attention_map_mid_layer.mean(axis=1)  # (num_inference_steps, num_prompt_obs)
    
    # Compute vmin/vmax for flattened weights (AFTER mean, to match video output)
    if scale_attention_weights:
        vmin_flattened = 0.0  # Always set minimum to 0
        vmax_flattened = attention_map_flattened.max()
        if vmax_flattened <= vmin_flattened:
            vmax_flattened = 1.0
    else:
        vmin_flattened, vmax_flattened = 0.0, 1.0
    
    # Build flattened heatmaps list (without saving individual files)
    flattened_heatmaps = []
    for step in range(num_inference_steps):
        attn_flat = attention_map_flattened[step]  # (num_prompt_obs,)
        # Reshape to 2D for heatmap: (1, num_prompt_obs) to create a horizontal bar
        attn_2d = attn_flat.reshape(1, -1)
        flattened_heatmaps.append(attn_2d)
    
    # Save cumulative combined flattened heatmaps (one file per timestep, rotated 90 degrees)
    for end_step in range(num_inference_steps):
        # Stack up to current timestep (cumulative)
        cumulative_flat = np.vstack(flattened_heatmaps[:end_step + 1])  # (end_step+1, num_prompt_obs)
        # Rotate 90 degrees counterclockwise
        cumulative_flat = np.rot90(cumulative_flat, 1)  # (num_prompt_obs, end_step+1)
        
        n_rows, n_cols = cumulative_flat.shape
        fig, ax = plt.subplots(figsize=(n_cols / 5, n_rows / 5))
        
        hm = sns.heatmap(
            cumulative_flat,
            ax=ax,
            cmap=cmap,
            cbar=include_colorbar_in_plots,
            vmin=vmin_flattened,
            vmax=vmax_flattened / use_proportion_of_colormap,
            xticklabels=False,
            yticklabels=False,
            square=True,
            rasterized=True
        )

        for _, spine in hm.spines.items():
            spine.set_visible(False)
        
        # Customize colorbar if present
        if include_colorbar_in_plots:
            cbar = hm.collections[0].colorbar
            # Set only 3 ticks: bottom, middle, top
            tick_values = [vmin_flattened, (vmin_flattened + vmax_flattened) / 2, vmax_flattened]
            cbar.set_ticks(tick_values)
            # Format tick labels to 2 decimal places
            cbar.ax.set_yticklabels([f'{val:.2f}' for val in tick_values])
            # Remove tick marks, keep labels only
            cbar.ax.tick_params(length=0)
        
        ax.set_aspect("equal")
        # ax.axis("off")
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # plt.margins(0, 0)
        
        # Save both SVG and PNG versions
        combined_svg_path = os.path.join(weights_flattened_dir, f"weights_flattened_combined_{end_step:04d}.svg")
        combined_png_path = os.path.join(weights_flattened_dir, f"weights_flattened_combined_{end_step:04d}.png")
        fig.savefig(combined_svg_path, bbox_inches="tight", pad_inches=0, dpi=1000) # rasterized=True + high DPI avoids weird white line artifacts showing up when rendering heatmap grid squares
        fig.savefig(combined_png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # Build weights_flattened.mp4: one frame per timestep (heatmap without colorbar), tight output then pad right/bottom with white for fixed frame size
    video_dpi = 25
    max_n_cols = num_inference_steps
    max_n_rows = num_prompt_obs
    max_width_px = int(max_n_cols * video_dpi)
    max_height_px = int(max_n_rows * video_dpi)
    # Same plot size for all steps: fixed figsize and layout (no colorbar space)
    video_figsize = (max_n_cols, max_n_rows)
    video_path = os.path.join(weights_flattened_dir, "weights_flattened.mp4")
    fps_val = (fps / exec_action_horizon) * steps_per_render
    with imageio.get_writer(video_path, fps=fps_val, codec="libx264") as writer:
        for end_step in tqdm(range(num_inference_steps), desc='Rendering flattened weights video', leave=False):
            cumulative_flat = np.vstack(flattened_heatmaps[:end_step + 1])
            cumulative_flat = np.rot90(cumulative_flat, 1)
            fig, ax = plt.subplots(figsize=video_figsize)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.set_position([0, 0, 1, 1])
            sns.heatmap(
                cumulative_flat,
                ax=ax,
                cmap=cmap,
                cbar=False,
                vmin=vmin_flattened,
                vmax=vmax_flattened / use_proportion_of_colormap,
                xticklabels=False,
                yticklabels=False,
                square=True,
                rasterized=True,
            )
            for _, spine in ax.spines.items():
                spine.set_visible(False)
            ax.set_aspect("equal")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=video_dpi)
            plt.close(fig)
            buf.seek(0)
            img = imageio.v3.imread(buf)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            # Pad with white to fixed size (no stretching); heatmap stays top-left, empty space to the right for earlier steps
            h, w = img.shape[0], img.shape[1]
            padded = np.full((max_height_px, max_width_px, 3), 255, dtype=np.uint8)
            padded[: min(h, max_height_px), : min(w, max_width_px)] = img[: max_height_px, : max_width_px]
            writer.append_data(padded)

    # Save corresponding rollout frames
    cap = cv2.VideoCapture(rollout_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    for step in range(num_inference_steps):
        # Mirror the correspondence in attention_vis_v1_util:
        # each attention step corresponds to exec_action_horizon action steps.
        frame_idx = step * exec_action_horizon / steps_per_render
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(rollout_dir, f"frame_{step:04d}.png")
        cv2.imwrite(frame_path, frame)

    cap.release()
    
    # Always save colorbars in both SVG and PNG format
    # Create separate colorbars for per-timestep (non-flattened) and flattened data
    
    # Colorbar for per-timestep heatmaps (non-flattened)
    fig, ax = plt.subplots(figsize=(1.5, 5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_per_timestep, vmax=vmax_per_timestep))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    tick_values = [vmin_per_timestep, (vmin_per_timestep + vmax_per_timestep) / 2, vmax_per_timestep]
    cbar.set_ticks(tick_values)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # Set font to Inconsolata and increase font size
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('monospace')
        label.set_fontname('Inconsolata')
        label.set_fontsize(14)
    # Set border and tick line width to 1
    cbar.outline.set_linewidth(1)
    cbar.ax.tick_params(width=1, length=6)
    plt.tight_layout()
    
    colorbar_per_timestep_svg_path = os.path.join(save_weights_base_path, "colorbar_per_timestep.svg")
    colorbar_per_timestep_png_path = os.path.join(save_weights_base_path, "colorbar_per_timestep.png")
    fig.savefig(colorbar_per_timestep_svg_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(colorbar_per_timestep_png_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
    plt.close(fig)
    
    # Colorbar for flattened weights
    fig, ax = plt.subplots(figsize=(1.5, 2.5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_flattened, vmax=vmax_flattened))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    tick_values = [vmin_flattened, (vmin_flattened + vmax_flattened) / 2, vmax_flattened]
    cbar.set_ticks(tick_values)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # Set font to Inconsolata and increase font size
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('monospace')
        label.set_fontname('Inconsolata')
        label.set_fontsize(14)
    # Set border and tick line width to 1
    cbar.outline.set_linewidth(1)
    cbar.ax.tick_params(width=1, length=6)
    plt.tight_layout()
    
    colorbar_flattened_svg_path = os.path.join(save_weights_base_path, "colorbar_flattened.svg")
    colorbar_flattened_png_path = os.path.join(save_weights_base_path, "colorbar_flattened.png")
    fig.savefig(colorbar_flattened_svg_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(colorbar_flattened_png_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
    plt.close(fig)
