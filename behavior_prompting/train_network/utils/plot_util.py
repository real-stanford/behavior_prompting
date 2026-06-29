import os
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import PairPromptObsEncoder
from behavior_prompting.train_network.policy.diffusion_transformer_policy import DiffusionTransformerPolicy
import numpy as np

from behavior_prompting.train_network.utils.attention_vis_v1_util import vis_attention_and_rollout_v1
from behavior_prompting.train_network.utils.attention_vis_v2_util import (
    vis_attention_and_rollout_v2,
    save_prompt_attention_artifacts,
)

default_action_dim_names_rot6d = ['rel pos x', 'rel pos y', 'rel pos z', 'rel rot6d 1', 'rel rot6d 2', 'rel rot6d 3', 'rel rot6d 4', 'rel rot6d 5', 'rel rot6d 6', 'gripper action']

def vis_training_batch_action_preds(prompt, action_preds, output_path, action_dim_names=default_action_dim_names_rot6d):
    """
    Visualize the action predictions for a given prompt. Assumes that the action preds are for the entire sequence.

    Args:
        prompt (dict): The prompt to visualize. (tensor dict)
        action_preds (numpy.ndarray): The action predictions to visualize. (T, num_pred_steps, action_dim) (numpy dict)
    """

    prompt_actions = prompt['action'].numpy() # (T, chunk_dim, num_pred_steps, action_dim)
    T, chunk_dim, num_pred_steps, action_dim = prompt_actions.shape
    action_preds = action_preds.numpy() # (T, num_pred_steps, action_dim)

    prompt_actions = prompt_actions[:, :, 0, :] # (T, chunk_dim, action_dim)
    prompt_actions = prompt_actions.reshape(T*chunk_dim, action_dim)
    action_preds = action_preds[:, :chunk_dim, :] # (T, num_pred_steps, action_dim)-> (T, chunk_dim, action_dim)
    action_preds = action_preds.reshape(T*chunk_dim, action_dim)

    mask_location = np.where(prompt['metadata']['prompt_mask'].numpy())[0][0] * chunk_dim # this is a time index where the prompt ends

    eos = prompt['metadata']['eos'].numpy()
    end_segment_locations = list((np.where(eos)[0] + 1) * chunk_dim)
        
    # Create figure with subplots for each action dimension
    action_dim = prompt_actions.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 3*action_dim))
    time_steps = np.arange(len(prompt_actions))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    # Plot each action dimension
    for i in range(action_dim):
        if action_dim == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Plot lines and store handles for legend
        gt_line = ax.plot(time_steps, prompt_actions[:, i], label='Ground Truth', alpha=0.7)[0]
        pred_line = ax.plot(time_steps, action_preds[:, i], label='Predicted', alpha=0.7)[0]
        
        # Only add handles and labels once
        if i == 0:
            legend_handles.extend([gt_line, pred_line])
            legend_labels.extend(['Ground Truth', 'Predicted'])
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{action_dim_names[i]}')
        ax.grid(True)

        # Add background shading to indicate prompt region
        # Shade prompt region (0 to mask_location) in light blue
        prompt_patch = ax.axvspan(0, mask_location-1, color='lightblue', alpha=0.3, label='Prompt Region')
        # Shade prediction region (after mask_location) in light gray 
        pred_patch = ax.axvspan(mask_location, len(prompt_actions), color='lightgray', alpha=0.3, label='Rollout Region')

        # Add vertical lines to indicate end of segments
        for end_loc in end_segment_locations:
            ax.axvline(x=end_loc, color='black', linestyle='--', alpha=0.5)
        
        # Only add patch handles once
        if i == 0:
            legend_handles.extend([prompt_patch, pred_patch])
            legend_labels.extend(['Prompt Region', 'Prediction Region'])

    # Add a single legend at the bottom of the figure
    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()

    print(f'Saved action predictions figure to {output_path}')

def vis_actions_rollout(output_path, actions, prompt=None, action_dim_names=default_action_dim_names_rot6d):
    """
    Visualize the actions for a given rollout (potentially with a prompt before the rollout).

    Args:
        actions (numpy.ndarray): The actions to visualize. (T, action_dim) (numpy dict)
        prompt (dict) [optional]: The prompt to visualize. (tensor dict). Prompt should not have a batch dimension.
    """

    if prompt is not None:
        prompt_actions = prompt['action'].cpu().numpy() # (T_prompt, chunk dim, num_pred_steps, action_dim)

        if len(prompt_actions.shape) == 4: # remove the num_pred_steps dimension if present
            prompt_actions = prompt_actions[:, :, 0] # (T_prompt, chunk dim, num_pred_steps, action_dim) -> (T_prompt, chunk_dim, action_dim)

        T_prompt, chunk_dim, action_dim = prompt_actions.shape
        prompt_actions = prompt_actions.reshape(T_prompt*chunk_dim, action_dim)
    else:
        action_dim = actions.shape[1]

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 3*action_dim))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    # Plot each action dimension
    for i in range(action_dim):
        ax = axes[i]
        
        if prompt is not None:
            # Plot prompt actions starting at 0
            prompt_line = ax.plot(np.arange(len(prompt_actions)), prompt_actions[:, i], label='Prompt Actions')[0]
            
            # Plot rollout actions continuing after prompt
            rollout_line = ax.plot(np.arange(len(prompt_actions), len(prompt_actions) + len(actions)), 
                                 actions[:, i], label='Rollout Actions')[0]
            
            # Add background shading to indicate prompt region
            prompt_patch = ax.axvspan(0, len(prompt_actions), color='lightblue', alpha=0.3, label='Prompt Region')
            rollout_patch = ax.axvspan(len(prompt_actions), len(prompt_actions) + len(actions), 
                                     color='lightgray', alpha=0.3, label='Rollout Region')
            
            ax.axvline(x=len(prompt_actions), color='black', linestyle='--', alpha=0.5)
        else:
            # If no prompt, just plot actions starting at 0
            rollout_line = ax.plot(np.arange(len(actions)), actions[:, i], label='Actions')[0]
        
        ax.set_ylabel(f'{action_dim_names[i]}')
        ax.set_xlabel('Time Step')
        ax.grid(True)

        # Only add handles once
        if i == 0:
            if prompt is not None:
                legend_handles.extend([prompt_line, rollout_line, prompt_patch, rollout_patch])
                legend_labels.extend(['Prompt Actions', 'Rollout Actions', 'Prompt Region', 'Rollout Region'])
            else:
                legend_handles.append(rollout_line)
                legend_labels.append('Actions')

    # Add a single legend at the bottom of the figure
    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()
    
    print(f'Saved action rollout figure to {output_path}')

def vis_proprio_rollout(output_path, proprio_list, proprio_key_names, proprio_key_sizes, prompt=None, proprio_legend_names=None):
    """
    Visualize the proprio for a given rollout (potentially with a prompt before the rollout).

    Args:
        proprio_list (list of numpy.ndarray): List of proprio arrays to visualize. Each array should be (T, proprio_dim)
        proprio_key_names (list): The names of the proprio keys.
        proprio_key_sizes (list): The sizes of the proprio keys.
        prompt (dict) [optional]: The prompt to visualize. (tensor dict). Prompt should not have a batch dimension.
        proprio_legend_names (list) [optional]: List of names to use in the legend for each proprio array. If None, will use default names "Proprio 1", "Proprio 2", etc.
    """

    # Process prompt data if available
    prompt_proprio = None
    if prompt is not None:
        remove_pred_steps = lambda x: x[:, 0] if len(x.shape) == 3 else x
        prompt_proprio = np.concatenate([remove_pred_steps(prompt['obs'][key].cpu().numpy()) for key in proprio_key_names], axis=-1)
        downsampled_prompt_len = prompt_proprio.shape[0]
        chunk_n_actions = prompt['action'].shape[1]
        prompt_proprio = prompt_proprio.repeat(chunk_n_actions, axis=0)  # repeat the prompt actions because they are downsampled

        if 'eos' in prompt['metadata']:
            eos = prompt['metadata']['eos'].cpu().numpy()
            end_segment_locations = list(np.where(eos)[0])
        else:
            end_segment_locations = [downsampled_prompt_len - 1]

    # Create figure with subplots for each proprio dimension
    proprio_dim = proprio_list[0].shape[1]
    fig, axes = plt.subplots(proprio_dim, 1, figsize=(10, 3*proprio_dim))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    proprio_dim_names = []
    for key, size in zip(proprio_key_names, proprio_key_sizes):
        for i in range(size):
            proprio_dim_names.append(f'{key} {i}')

    # Plot each proprio dimension
    for i in range(proprio_dim):
        ax = axes[i]
        
        if prompt_proprio is not None:
            # Plot prompt data
            prompt_line = ax.plot(np.arange(len(prompt_proprio)), 
                                prompt_proprio[:, i], 
                                label='Prompt', alpha=0.7)[0]
            
            # Plot rollout data for each trajectory
            for j, proprio in enumerate(proprio_list):
                rollout_line = ax.plot(np.arange(len(prompt_proprio), len(prompt_proprio) + len(proprio)), 
                                     proprio[:, i],
                                     label=proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}',
                                     alpha=0.7)[0]
                
                if i == 0:  # Only add to legend once
                    legend_handles.append(rollout_line)
                    legend_labels.append(proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')
            
            # Add background shading
            prompt_patch = ax.axvspan(0, len(prompt_proprio), color='lightblue', alpha=0.3, label='Prompt Region')
            rollout_patch = ax.axvspan(len(prompt_proprio), len(prompt_proprio) + len(proprio_list[0]), 
                                     color='lightgray', alpha=0.3, label='Rollout Region')
            
            # Add vertical lines for segment ends
            for end_loc in end_segment_locations:
                ax.axvline(x=(end_loc+1)*chunk_n_actions, color='black', linestyle='--', alpha=0.5)
                
            if i == 0:  # Only add to legend once
                legend_handles.extend([prompt_line, prompt_patch, rollout_patch])
                legend_labels.extend(['Prompt', 'Prompt Region', 'Rollout Region'])
        else:
            # If no prompt, just plot rollout data
            for j, proprio in enumerate(proprio_list):
                rollout_line = ax.plot(np.arange(len(proprio)), proprio[:, i],
                                     label=proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')[0]
                if i == 0:  # Only add to legend once
                    legend_handles.append(rollout_line)
                    legend_labels.append(proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')

        ax.set_ylabel(f'{proprio_dim_names[i]}')
        ax.set_xlabel('Time Step')
        ax.grid(True)

    # Add a single legend at the bottom of the figure
    if len(legend_handles) > 0:
        fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()
    
    print(f'Saved proprio rollout figure to {output_path}')


class PromptAttentionLogger:
    def __init__(self, policy: BasePolicy, env_index: int):
        self.policy = policy
        self.has_prompt_with_obs_decoder = isinstance(policy.obs_encoder, PairPromptObsEncoder) and policy.obs_encoder.prompt_with_obs_decoder_enabled
        self.has_diffusion_attention = isinstance(policy, DiffusionTransformerPolicy)
        self.env_index = env_index
        
        self.diffusion_cross_attn_weights = [] # (T, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        if self.has_prompt_with_obs_decoder:
            self.prompt_cross_attn_weights = [] # (T, num_layers, receding_obs_len, prompt_obs_len)

    def log(self, action_dict: dict):
        if self.has_prompt_with_obs_decoder:
            cross_attn_weights = action_dict['obs_encoder_metadata']['prompt_with_obs_transfomer_attention_weights'].to("cpu").numpy() # (num_layers, batch, receding_obs_len, prompt_obs_len)
            self.prompt_cross_attn_weights.append(cross_attn_weights[:,self.env_index])
        
        if self.has_diffusion_attention:
            cross_attn_weights = action_dict['diffusion_attention_weights'].to("cpu").numpy() # (num_diffusion_steps, num_layers, batch, action_pred_steps, cross_attn_dim)
            self.diffusion_cross_attn_weights.append(cross_attn_weights[:,:,self.env_index])

    def vis(self, out_path, prompt_for_env, rollout_video_path: str, rgb_keys: list, steps_per_render: int, exec_action_horizon: int, fps: int, version: str = 'v2', save_weights: bool = False, cmap: str = "GnBu", scale_attention_weights: bool = False, use_proportion_of_colormap: float = 1.0, stack_prompt_imgs: str = "horizontal", rollout_width_fraction: float = 1/3):
        """
        There are two versions of the visualization:
        - v1: plot attention in separate grid (more information)
        - v2: plot attention overlayed on top of the prompt frames (more interpretable)
        
        Args:
            cmap: Colormap to use for attention visualization. Default is "GnBu".
            scale_attention_weights: If True, scale attention weights to [0, 1] range to use full colormap. Default is False.
            use_proportion_of_colormap: Use only this proportion of the colormap range (0, 1]. Default is 1.0.
        """

        save_weights_base_path = os.path.join(os.path.dirname(out_path), os.path.basename(out_path).replace('.mp4', ''))

        if self.has_diffusion_attention:
            diffusion_cross_attn_weights = np.stack(self.diffusion_cross_attn_weights, axis=0)  # (T, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        else:
            diffusion_cross_attn_weights = None

        if self.has_prompt_with_obs_decoder:
            prompt_cross_attn_weights = np.stack(self.prompt_cross_attn_weights, axis=0)  # (T, num_layers, receding_obs_len, prompt_obs_len)

            if save_weights:
                save_prompt_attention_artifacts(
                    save_weights_base_path=save_weights_base_path,
                    prompt_cross_attn_weights=prompt_cross_attn_weights,
                    rollout_video_path=rollout_video_path,
                    prompt_for_vis=prompt_for_env,
                    rgb_keys=rgb_keys,
                    exec_action_horizon=exec_action_horizon,
                    steps_per_render=steps_per_render,
                    policy=self.policy,
                    cmap=cmap,
                    scale_attention_weights=scale_attention_weights,
                    use_proportion_of_colormap=use_proportion_of_colormap,
                    fps=fps,
                )
        else:
            prompt_cross_attn_weights = None

        # visualization for diffusion
        assert version in ['v1', 'v2']
        vis_function = vis_attention_and_rollout_v1 if version == 'v1' else vis_attention_and_rollout_v2
        vis_function(out_path, rollout_video_path, exec_action_horizon, self.policy, prompt_for_env, fps, rgb_keys, steps_per_render=steps_per_render, prompt_cross_attn_weights=prompt_cross_attn_weights, diffusion_cross_attn_weights=diffusion_cross_attn_weights, cmap=cmap, scale_attention_weights=scale_attention_weights, use_proportion_of_colormap=use_proportion_of_colormap, stack_prompt_imgs=stack_prompt_imgs, rollout_width_fraction=rollout_width_fraction)
