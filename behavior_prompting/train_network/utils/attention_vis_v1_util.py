from typing import Optional
import matplotlib
import torch

from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import PairPromptObsEncoder
from behavior_prompting.train_network.policy.diffusion_transformer_policy import DiffusionTransformerPolicy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.animation as animation
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import os
import matplotlib.animation as animation
import imageio
import logging
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)


def get_prompt_frame_images(prompt, prompt_img_keys):
    """
    Convert prompt observations into a list of stacked RGB frames (H, W, 3) as uint8.
    """
    # Filter out keys that don't exist in the prompt
    prompt_img_keys = [x for x in prompt_img_keys if x in prompt['obs'].keys()]
    stacked_imgs = []
    for t in range(prompt['obs'][prompt_img_keys[0]].shape[0]):
        imgs_at_t = [np.transpose(prompt['obs'][key][t], (1, 2, 0)) for key in prompt_img_keys]
        stacked_img = np.vstack(imgs_at_t)
        if stacked_img.dtype == np.float32 or stacked_img.dtype == np.float64:
            if stacked_img.max() <= 1.0:
                stacked_img = np.clip(stacked_img, 0, 1)
                stacked_img = (stacked_img * 255).round().astype(np.uint8)
            else:
                stacked_img = np.clip(stacked_img, 0, 255).astype(np.uint8)
        stacked_imgs.append(stacked_img)
    return stacked_imgs


def plot_prompt_obs(prompt, output_path, prompt_img_keys, add_titles: bool = True):
    """
    Plot the prompt observations by vertically stacking images from different keys for each timestep, then concatenate all timesteps horizontally with no whitespace. Adds a title above each image.
    Args:
        prompt (dict): Dictionary containing image tensors for each key in prompt_img_keys. Each tensor is (T, 3, H, W) and dtype float32.
        output_path (str): Path to save the resulting plot.
        prompt_img_keys (list): List of keys to extract images from the prompt.
        add_titles (bool): If True, adds titles above each prompt frame.
    """
    # For each timestep, vertically stack the images from all keys
    stacked_imgs = get_prompt_frame_images(prompt, prompt_img_keys)
    widths = [img.shape[1] for img in stacked_imgs]

    # Concatenate all stacked images horizontally (side by side)
    concat_img = np.hstack(stacked_imgs)  # (sum_H, T*W, 3)
    total_width = concat_img.shape[1]
    height = concat_img.shape[0]

    # Plot the single concatenated image
    fig, ax = plt.subplots(figsize=(total_width / 100, height / 100 + 0.5), dpi=100)
    ax.imshow(concat_img)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)

    # Optionally add a title above each image
    if add_titles:
        x = 0
        for t, w in enumerate(widths):
            ax.text(x + w / 2, -10, f'Prompt Img {t}', va='bottom', ha='center', fontsize=12, color='black', fontweight='bold', transform=ax.transData, clip_on=False)
            x += w

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def vis_attention_map_receding_obs_with_prompt(output_path, attention_map, prompt_current_obs_names, prompt_obs_names, fps, exec_action_horizon, max_frames=None, cmap='viridis', scale_attention_weights: bool = False, use_proportion_of_colormap: float = 1.0):
    """
    Visualize the attention map as a video. For use in policies where prompt is cross attended to the receding obs.

    Args:
        output_path (str): The path to save the attention map video. Should end with .mp4 or .gif.
        attention_map (numpy.ndarray): The attention map to visualize. (num_inference_steps, num_layers, num_receding_obs, num_prompt_obs)
        prompt_current_obs_names (list): The names of the receding obs.
        prompt_obs_names (list): The names of the prompt obs.
        fps (int): Frames per second for the output video.
        exec_action_horizon (int): Number of action steps per inference step.
        max_frames (int, optional): Maximum number of frames to render. If None, use all frames.
        cmap (str, optional): Colormap to use. Default is 'viridis'.
        scale_attention_weights (bool, optional): If True, scale attention weights to [0, 1] range after processing. Default is False.
        use_proportion_of_colormap (float, optional): Use only this proportion of the colormap range (0, 1]. Default is 1.0.
    """
    attention_map = attention_map[..., :len(prompt_obs_names)]

    num_inference_steps, num_layers, num_receding_obs, num_prompt_obs = attention_map.shape

    attention_map = attention_map[:, num_layers//2] # (num_inference_steps, num_receding_obs, num_prompt_obs)
    
    # Compute vmin/vmax for colormap to use full range (but keep actual values)
    if scale_attention_weights:
        vmin = 0.0  # Always set minimum to 0
        vmax = attention_map.max()
        if vmax <= vmin:
            vmax = 1.0
    else:
        vmin, vmax = 0, 1  # we are plotting attention values after softmax so they are all between 0 and 1

    # Use only a proportion of the colormap so that max value maps to that proportion
    vmax_display = vmax / use_proportion_of_colormap

    num_inference_steps = attention_map.shape[0]
    if max_frames is not None:
        num_frames = min(num_inference_steps, max_frames)
    else:
        num_frames = num_inference_steps

    # receding obs on y-axis, prompt obs on x-axis
    fig, ax = plt.subplots(figsize=(8+len(prompt_obs_names)//3, 10))
    
    # One-time setup: create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    def update(step):
        # Clear previous heatmap
        for artist in ax.get_children():
            if isinstance(artist, (plt.matplotlib.collections.QuadMesh,)):
                artist.remove()
        
        # Draw new heatmap
        heatmap = sns.heatmap(attention_map[step], ax=ax, cmap=cmap, cbar=False, vmin=vmin, vmax=vmax_display, 
                             xticklabels=prompt_obs_names, yticklabels=prompt_current_obs_names)
        
        # Update colorbar
        im = heatmap.get_children()[0]
        plt.colorbar(im, cax=cax)
        
        # Set plot properties
        ax.set_aspect(1.0)
        ax.set_xlabel('Prompt', fontsize=22)
        ax.set_ylabel('Receding Obs', fontsize=22)
        ax.set_title(f'Prompt Attention Map - Step {step*exec_action_horizon}/{num_frames*exec_action_horizon}', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()

    # Use tqdm for progress bar during animation saving
    frame_generator = tqdm(range(num_frames), total=num_frames, desc='Rendering frames', leave=False)

    ani = animation.FuncAnimation(fig, update, frames=frame_generator, repeat=False)

    # Save as video (mp4)
    if output_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_path, writer=writer)
    else:
        raise ValueError('Output path must end with .mp4 or .gif')
    plt.close()


def vis_attention_map_action_with_obs(output_path, attention_map, exec_action_horizon, cross_attn_dim_names, fps, max_frames=None, cmap='viridis'):
    """
    Visualize the attention map as a video.

    Args:
        output_path (str): The path to save the attention map video. Should end with .mp4 or .gif.
        attention_map (numpy.ndarray): The attention map to visualize. (num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        exec_action_horizon (int): Number of action steps per inference step.
        cross_attn_dim_names (list): The names of the cross-attention dimensions.
        fps (int): Frames per second for the output video.
        max_frames (int, optional): Maximum number of frames to render. If None, use all frames.
    """
    attention_map = attention_map[..., :len(cross_attn_dim_names)]

    num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim = attention_map.shape

    # Select the attention map at the middle of the diffusion steps
    mid_diff_step = num_diffusion_steps // 2
    attention_map = attention_map[:, mid_diff_step]  # (num_inference_steps, num_layers, action_pred_steps, cross_attn_dim)
    attention_map = attention_map[:, :, :exec_action_horizon] # (num_inference_steps, num_layers, action_pred_steps, cross_attn_dim)
    attention_map = np.transpose(attention_map, (1, 0, 2, 3)) # (num_layers, num_inference_steps, exec_action_horizon, cross_attn_dim)
    attention_map = attention_map.reshape(num_layers, num_inference_steps*exec_action_horizon, cross_attn_dim) # (num_layers, num_inference_steps*exec_action_horizon, cross_attn_dim)
    attention_map = np.transpose(attention_map, (1, 0, 2)) # (num_inference_steps*exec_action_horizon, num_layers, cross_attn_dim)

    num_exec_steps = num_inference_steps * exec_action_horizon
    if max_frames is not None:
        num_frames = min(num_exec_steps, max_frames)
    else:
        num_frames = num_exec_steps
    
    # Reshape to have layers on y-axis and cross_attn_dim on x-axis
    attention_map = attention_map.reshape(num_exec_steps, num_layers, cross_attn_dim)

    fig, ax = plt.subplots(figsize=(7+len(cross_attn_dim_names)//3, 7))
    vmin, vmax = 0, 1 # we are plotting attention values after softmax so they are all between 0 and 1
    # Draw the first frame and colorbar
    heatmap = sns.heatmap(attention_map[0], ax=ax, cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, xticklabels=cross_attn_dim_names)
    ax.set_aspect(1.0)
    # Use make_axes_locatable to match colorbar height to axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = heatmap.get_children()[0]
    plt.colorbar(im, cax=cax)
    ax.set_xlabel('Encoder Tokens', fontsize=22)
    ax.set_ylabel('Decoder Layers', fontsize=22)
    ax.set_title(f'Attention Map - Step 1/{num_frames}', pad=10, fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    def update(step):
        ax.set_title(f'Action Attention Map - Step {step+1}/{num_frames}', fontsize=28)
        # Remove the old heatmap but keep the colorbar
        for artist in ax.get_children():
            if isinstance(artist, (plt.matplotlib.collections.QuadMesh,)):
                artist.remove()
        sns.heatmap(attention_map[step], ax=ax, cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, xticklabels=cross_attn_dim_names)
        ax.set_aspect(1.0)
        ax.set_xlabel('Encoder Tokens', fontsize=22)
        ax.set_ylabel('Decoder Layers', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()

    # Use tqdm for progress bar during animation saving
    frame_generator = tqdm(range(num_frames), total=num_frames, desc='Rendering frames', leave=False)

    ani = animation.FuncAnimation(fig, update, frames=frame_generator, repeat=False)

    # Save as video (mp4)
    if output_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_path, writer=writer)
    else:
        raise ValueError('Output path must end with .mp4 or .gif')
    plt.close()


def combine_attention_and_rollout_video(rollout_video_path, output_path, prompt, prompt_img_keys, exec_action_horizon, steps_per_render, action_attention_map_path: Optional[str] = None, prompt_attention_map_path: Optional[str] = None):
    """
    Combine the attention map video and the rollout video into a single video, with the prompt observation image below each frame.
    """
    assert action_attention_map_path is not None or prompt_attention_map_path is not None, 'at least one of action_attention_map_path or prompt_attention_map_path must be provided'

    # Read videos
    if action_attention_map_path is not None:
        diffusion_attention_cap = cv2.VideoCapture(action_attention_map_path)
        diffusion_attn_width = int(diffusion_attention_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        diffusion_attn_height = int(diffusion_attention_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diffusion_attn_fps = diffusion_attention_cap.get(cv2.CAP_PROP_FPS)
        diffusion_attn_frames = int(diffusion_attention_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    rollout_cap = cv2.VideoCapture(rollout_video_path)
    roll_width = int(rollout_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roll_height = int(rollout_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roll_frames = int(rollout_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roll_fps = rollout_cap.get(cv2.CAP_PROP_FPS)
    
    if prompt_attention_map_path is not None:
        prompt_attention_map_cap = cv2.VideoCapture(prompt_attention_map_path)
        prompt_attn_width = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prompt_attn_height = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prompt_attn_fps = prompt_attention_map_cap.get(cv2.CAP_PROP_FPS)
        prompt_attn_frames = int(prompt_attention_map_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate resized rollout width to match attention frame height
    if action_attention_map_path is not None:
        scale = diffusion_attn_height / roll_height
        resized_roll_width = int(roll_width * scale)
        resized_roll_height = int(roll_height * scale)

        diffusion_with_rollout_width = diffusion_attn_width + resized_roll_width
        diffusion_with_rollout_height = diffusion_attn_height
    else:
        scale = prompt_attn_height / roll_height
        resized_roll_width = int(roll_width * scale)
        resized_roll_height = int(roll_height * scale)
        diffusion_with_rollout_width = resized_roll_width
        diffusion_with_rollout_height = resized_roll_height

    # if prompt attention map exists, compute scale to match the width of the diffusion attention map
    if prompt_attention_map_path is not None:
        if action_attention_map_path is not None:
            scale = diffusion_with_rollout_width / prompt_attn_width
            resized_prompt_attn_width = int(prompt_attn_width * scale)
            resized_prompt_attn_height = int(prompt_attn_height * scale)
            final_attention_height = diffusion_with_rollout_height + resized_prompt_attn_height
        else:
            final_attention_height = prompt_attn_height
            diffusion_with_rollout_width = prompt_attn_width + resized_roll_width

    # --- Generate prompt observation image ---
    vis_dir = os.path.dirname(output_path)
    prompt_obs_path = os.path.join(vis_dir, f'tmp_{os.path.splitext(os.path.basename(output_path))[0]}_prompt_obs.png')
    plot_prompt_obs(prompt, prompt_obs_path, prompt_img_keys)
    prompt_obs_img = cv2.imread(prompt_obs_path)
    if prompt_obs_img is None:
        raise RuntimeError(f"Failed to load prompt observation image from {prompt_obs_path}")
    # Resize prompt_obs_img to match the combined width of the video frames, scaling height proportionally
    combined_width = diffusion_with_rollout_width
    orig_height, orig_width = prompt_obs_img.shape[:2]
    scale_factor = combined_width / orig_width
    new_height = int(orig_height * scale_factor)
    prompt_obs_img = cv2.resize(prompt_obs_img, (combined_width, new_height))
    prompt_obs_height = prompt_obs_img.shape[0]

    # Create video writer with correct dimensions (width, height)
    out_height = final_attention_height + prompt_obs_height
    out_fps = diffusion_attn_fps if action_attention_map_path is not None else (roll_fps * steps_per_render) # the video is written out at the actions frequency. If steps_per_render > 1, then we will just repeat rollout frames as needed
    out = imageio.get_writer(output_path, fps=out_fps, codec='libx264')

    num_frames = diffusion_attn_frames if action_attention_map_path is not None else (roll_frames * steps_per_render)

    # Read and combine frames
    for action_idx in tqdm(range(num_frames), desc='Combining attention, rollout, and prompt obs', leave=False):
        if action_attention_map_path is not None:
            ret_attn, diffusion_attn_frame = diffusion_attention_cap.read()
            if not ret_attn:
                break

        # Only read new rollout frame every steps_per_render frames
        if action_idx % steps_per_render == 0:
            ret_roll, roll_frame = rollout_cap.read()
            if not ret_roll:
                break
            current_roll_frame = roll_frame

        # read prompt attention map if it exists
        if prompt_attention_map_path is not None:
            if action_idx % exec_action_horizon == 0:
                ret_prompt_attn, prompt_attn_frame = prompt_attention_map_cap.read()
                if not ret_prompt_attn:
                    break
                if action_attention_map_path is not None:
                    prompt_attn_frame = cv2.resize(prompt_attn_frame, (resized_prompt_attn_width, resized_prompt_attn_height))
        
        # Resize rollout frame to match attention frame height and precomputed width
        resized_roll = cv2.resize(current_roll_frame, (resized_roll_width, resized_roll_height))

        # stack attention and rollout horizontally
        if action_attention_map_path is not None:
            attn_rollout_frame = np.hstack((diffusion_attn_frame, resized_roll))
        else:
            attn_rollout_frame = resized_roll

        # compute attention frame (might be combination of diffusion and prompt attention maps)
        if prompt_attention_map_path is not None:
            if action_attention_map_path is not None:
                cur_frame = np.vstack((attn_rollout_frame, prompt_attn_frame))
            else:
                cur_frame = np.hstack((prompt_attn_frame, attn_rollout_frame))
            
        # Stack prompt observation image below
        combined_full = np.vstack((cur_frame, prompt_obs_img))
        out.append_data(combined_full[...,::-1]) # convert to RGB

    # Release everything
    if action_attention_map_path is not None:
        diffusion_attention_cap.release()
    if prompt_attention_map_path is not None:
        prompt_attention_map_cap.release()
    rollout_cap.release()
    out.close()
    # Remove the temporary prompt observation image
    if os.path.exists(prompt_obs_path):
        os.remove(prompt_obs_path)


def vis_attention_and_rollout_v1(out_path: str, rollout_video_path: str, exec_action_horizon: int, policy: BasePolicy, prompt_for_vis, fps: int, rgb_keys: list[str], steps_per_render: int, prompt_cross_attn_weights: Optional[torch.Tensor]=None, diffusion_cross_attn_weights: Optional[torch.Tensor]=None, max_frames=None, cmap='viridis', scale_attention_weights: bool = False, use_proportion_of_colormap: float = 1.0):
    """Visualizes the attention map and rollout video, with the prompt observation image below each frame. Supports both action-with-obs and receding-obs-with-prompt policies."""
    prompt_len = prompt_for_vis['action'].shape[0]

    if prompt_cross_attn_weights is not None:
        # prompt_cross_attn_weights: (num_inference_steps, num_layers, num_prompt_current_obs, prompt_obs_len)
        prompt_attention_map_path = os.path.join(os.path.dirname(out_path), f"tmp_{os.path.splitext(os.path.basename(out_path))[0]}_attention_map_prompt.mp4")
        obs_encoder = policy.obs_encoder
        assert isinstance(obs_encoder, PairPromptObsEncoder)
        prompt_current_obs_names, prompt_obs_names = obs_encoder.get_prompt_cross_attn_dim_names(prompt_len)
        vis_attention_map_receding_obs_with_prompt(prompt_attention_map_path, prompt_cross_attn_weights, prompt_current_obs_names, prompt_obs_names, fps, exec_action_horizon, max_frames, cmap=cmap, scale_attention_weights=scale_attention_weights, use_proportion_of_colormap=use_proportion_of_colormap)
    else:
        prompt_attention_map_path = None
    
    if diffusion_cross_attn_weights is not None:
    # diffusion_cross_attn_weights: (num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        action_attention_map_path = os.path.join(os.path.dirname(out_path), f"tmp_{os.path.splitext(os.path.basename(out_path))[0]}_attention_map_diffusion.mp4")
        cross_attn_dim_names = policy.get_diffusion_cross_attn_dim_names(prompt_len)
        vis_attention_map_action_with_obs(action_attention_map_path, diffusion_cross_attn_weights, exec_action_horizon, cross_attn_dim_names, fps, max_frames, cmap=cmap)
    else:
        action_attention_map_path = None

    combine_attention_and_rollout_video(rollout_video_path, out_path, prompt_for_vis, rgb_keys, exec_action_horizon, steps_per_render, action_attention_map_path, prompt_attention_map_path)

    if action_attention_map_path is not None:
        os.remove(action_attention_map_path)

    if prompt_attention_map_path is not None:
        os.remove(prompt_attention_map_path)
