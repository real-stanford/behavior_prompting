import os
import time
import wandb
import numpy as np
import torch
import tqdm
import imageio
from typing import Optional
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import get_training_task_names_from_training_split_info
from behavior_prompting.train_network.dataset.umi_task_dataset import UmiTaskDataset
from behavior_prompting.train_network.utils.plot_util import PromptAttentionLogger
from behavior_prompting.train_network.utils.umi_util import vis_prompt
from behavior_prompting.common.pytorch_util import move_batch_to_device, move_batch_to_numpy, add_batch_dim, remove_batch_dim
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.env_runner.base_runner import BaseRunner

class UmiPromptingRunner(BaseRunner):
    def __init__(self,
            output_dir,
            replay_buffer: ReplayBuffer,
            task_name: str,
            shape_meta: dict,
            max_steps: int = 800,
            tqdm_interval_sec=1.0,
            exec_action_horizon=12,
            vis_attention_map: bool=False,
            attention_map_version: str='v1',
            save_attention_weights: bool=False,
            vis_prompt: bool=False,
            cache_dir: Optional[str]=None,
            pose_repr: dict={},
            obs_down_sample_steps: int=3,
            is_eval_dataset: bool=False,
        ):
        super().__init__(output_dir)

        self.use_prompting = shape_meta['use_prompting']
        self.prompt_sample_mode = shape_meta['prompt_sample_mode'] if self.use_prompting else None
        assert exec_action_horizon <= shape_meta['action']['horizon'], 'number of steps to execute must not be greater than the action horizon predicted by the policy'
        
        self.replay_buffer = replay_buffer
        self.task_name = task_name
        self.is_eval_dataset = is_eval_dataset
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.exec_action_horizon = exec_action_horizon
        self.prompt_sequence_length = shape_meta.get('prompt_sequence_length', None)
        self.vis_attention_map = vis_attention_map
        self.attention_map_version = attention_map_version
        self.save_attention_weights = save_attention_weights
        self.vis_prompt = vis_prompt
        self.cache_dir = cache_dir
        self.pose_repr = pose_repr
        self.shape_meta = shape_meta
        self.obs_down_sample_steps = obs_down_sample_steps
        self.rgb_keys = []
        self.lowdim_keys = []
        for key in shape_meta['obs']:
            if shape_meta['obs'][key]['type'] == 'rgb':
                self.rgb_keys.append(key)
            elif shape_meta['obs'][key]['type'] == 'low_dim':
                self.lowdim_keys.append(key)

    def run(self, policy: BasePolicy, enable_expensive_vis: bool=True):
        print(f"\n=== Started UmiPromptingRunner run for task \"{self.task_name}\" ===")
        assert policy.supports_prompting(), "UmiPromptingRunner only supports prompting models"

        # Verify task if task should be in training dataset
        training_task_names = get_training_task_names_from_training_split_info(policy.get_training_split_info())
        if self.is_eval_dataset:
            assert self.task_name not in training_task_names, f"task \"{self.task_name}\" should not be in training dataset for eval tasks"
        else:
            assert self.task_name in training_task_names, f"task \"{self.task_name}\" should be in training dataset"

        device = policy.device
        dtype = policy.dtype

        vis_out_dir = os.path.join(self.output_dir, "vis")
        os.makedirs(vis_out_dir, exist_ok=True)

        policy.reset(action_exec_horizon=self.exec_action_horizon)

        # Create dataset - it will return both prompt and rollout observations
        rollout_dataset = UmiTaskDataset(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            seed=0,
            sample_type='task',
            only_prompt=False,
            training_split_info=policy.get_training_split_info() if not self.is_eval_dataset else None,
            val_ratio=0,
            cache_dir=self.cache_dir,
            pose_repr=self.pose_repr,
            action_padding=True
        )

        # find the starting index for a non-error-correction demo of the given task_name
        task_name_to_non_ec_indices = rollout_dataset.get_unique_task_name_to_dataset_indices(exclude_error_correction=True)
        assert self.task_name in task_name_to_non_ec_indices, (
            f"task '{self.task_name}' not found in dataset or has no non-error-correction demos"
        )
        start_idx = task_name_to_non_ec_indices[self.task_name][0]
        self.rollout_idx = start_idx

        # Load prompt from dataset
        start_time = time.time()
            
        # Get prompt from dataset - the prompt is included in the observation dictionary
        prompt_sample = rollout_dataset[self.rollout_idx]
        prompt_dict = prompt_sample['obs']['prompt']
        
        # Save numpy version for visualization if needed
        prompt_dict_numpy = None
        if self.vis_prompt:
            prompt_dict_numpy = move_batch_to_numpy(prompt_dict)
            
        # Convert numpy arrays to tensors and add batch dimension and move to device
        prompt_dict = add_batch_dim(prompt_dict)
        prompt_dict = move_batch_to_device(prompt_dict, device=device)
            
        policy.prompt(prompt_dict)

        end_time = time.time()
        print(f"Finished prompting model! Time taken: {end_time - start_time} seconds")
        
        # Visualize prompt if enabled
        if self.vis_prompt and prompt_dict_numpy is not None:
            prompt_vis_path = os.path.join(vis_out_dir, f"prompt_{self.task_name.replace(' ', '_')}.mp4")
            vis_prompt(prompt_dict_numpy, prompt_vis_path)
            print(f"Saved prompt visualization to {prompt_vis_path}")

        # This makes all subsequent samples from the dataset not include the prompt
        rollout_dataset.set_ignore_prompt(True)

        # Get the task/episode info from the initial rollout sample to verify we stay in the same demo segment
        initial_sample = rollout_dataset[self.rollout_idx]
        initial_task_idx = initial_sample['metadata']['task_idx'].item()
        initial_episode_idx = initial_sample['metadata']['episode_idx'].item()
        
        # Initialize attention logger if needed
        if self.vis_attention_map:
            attn_logger = PromptAttentionLogger(policy, env_index=0)

        # Store observations for video creation
        stored_obs_frames = {key: [] for key in self.rgb_keys} if self.vis_attention_map else {}

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Rollout \"{self.task_name}\"", 
            leave=False, mininterval=self.tqdm_interval_sec)

        # Iterate through time steps
        # At each downsampled step, use the dataset to get receding observations
        # Then selectively run inference every exec_action_horizon steps
        # and store the attention map for visualization
        for t in range(self.max_steps):
            # The rollout_idx is the starting point of the demo
            # The dataset contains observations at the full 60Hz (camera) frequency
            # so we downsample as is done in the real-world deployment
            dataset_idx = self.rollout_idx + t * self.obs_down_sample_steps
            
            # Check if we've exceeded the available samples
            if dataset_idx >= len(rollout_dataset):
                break
            
            # Get sample from dataset (already processed as torch tensors)
            sample = rollout_dataset[dataset_idx]
            
            # Verify we're still in the same task/episode
            current_task_idx = sample['metadata']['task_idx'].item()
            current_episode_idx = sample['metadata']['episode_idx'].item()
            if current_task_idx != initial_task_idx or current_episode_idx != initial_episode_idx:
                break
            
            # Extract receding observations from the sample
            # The dataset returns obs as a dictionary with torch tensors
            # We add a batch dimension to the observations for the policy to process
            obs_dict = add_batch_dim(sample['obs'])
            obs_dict = move_batch_to_device(obs_dict, device=device)

            # Store frames for video creation (from the most recent observation)
            if self.vis_attention_map:
                for key in self.rgb_keys:
                    # Get the last timestep (most recent) 
                    frame = obs_dict[key][0, -1].detach().cpu().numpy()
                    if len(frame.shape) == 3:  # (C, H, W)
                        frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
                    stored_obs_frames[key].append(frame)

            # We only run inference at exec_action_horizon steps with receding observations as input
            # and store the attention map for visualization
            with torch.inference_mode():
                kwargs = dict()
                if self.vis_attention_map and (t % self.exec_action_horizon == 0):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = True
                    action_dict = policy.predict_action(obs_dict, **kwargs)
                    attn_logger.log(action_dict)                

            # Update progress bar
            pbar.update(1)

        pbar.close()
        policy.reset(action_exec_horizon=self.exec_action_horizon)

        # Create video from stored observations for visualization
        # rollout_fps is the fps of the rollout video, camera is downsampled from 60Hz by obs_down_sample_steps
        rollout_fps = 60 / self.obs_down_sample_steps
        
        all_video_paths = [None]
        if self.vis_attention_map and len(stored_obs_frames) > 0:
            main_rgb_keys = sorted([k for k in self.rgb_keys if '_main_rgb' in k])
            video_keys = main_rgb_keys if main_rgb_keys else self.rgb_keys[:1]
            if all(k in stored_obs_frames for k in video_keys):
                video_path = os.path.join(vis_out_dir, f"rollout_{self.task_name.replace(' ', '_')}.mp4")
                n_frames = len(stored_obs_frames[video_keys[0]])
                processed_frames = []
                for i in range(n_frames):
                    side_by_side = []
                    for key in video_keys:
                        frame = stored_obs_frames[key][i]
                        if frame.dtype == np.float32 or frame.dtype == np.float64:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        side_by_side.append(frame)
                    processed_frames.append(np.concatenate(side_by_side, axis=1))

                if len(processed_frames) > 0:
                    with imageio.get_writer(video_path, fps=rollout_fps, codec='libx264') as writer:
                        for frame in processed_frames:
                            writer.append_data(frame)
                    all_video_paths[0] = video_path

        log_data = dict()

        # Visualize attention map if needed
        if self.vis_attention_map and self.use_prompting:
            prompt_for_vis = remove_batch_dim(prompt_dict, index_to_keep=0)
            prompt_for_vis = move_batch_to_numpy(prompt_for_vis)

            # prefer to show ultrawide views in the attention vis, but default to showing main camera if ultrawide is not in the prompt. Sorting ensures that "left" comes before "right" if both are present
            # exclude ignore_by_policy keys since they won't be present in prompt['obs']
            _non_goal_keys = [key for key in self.rgb_keys if key != 'goal_image' and not self.shape_meta['obs'].get(key, {}).get('ignore_by_policy', False)]
            _ultrawide_keys = [key for key in _non_goal_keys if 'ultrawide' in key]
            rgb_keys = sorted(_ultrawide_keys if _ultrawide_keys else [key for key in _non_goal_keys if 'main' in key])
            
            vis_demo_name = f"umi_{self.task_name.replace(' ', '_')}"
            attention_and_rollout_vis_path = os.path.join(vis_out_dir, f"attention_and_rollout_{vis_demo_name}.mp4")

            # Use the created rollout video for visualization
            rollout_video_path = all_video_paths[0]
            
            attn_logger.vis(
                attention_and_rollout_vis_path,
                prompt_for_vis,
                rollout_video_path,
                rgb_keys,
                steps_per_render=1,
                exec_action_horizon=self.exec_action_horizon,
                fps=rollout_fps,  # Match the actual rollout video FPS (1.67 fps = one frame per policy inference)
                version=self.attention_map_version,
                save_weights=self.save_attention_weights,
            )

            print(f'Saved attention and rollout visualization to {attention_and_rollout_vis_path}')

            # Log to wandb if video was created
            if rollout_video_path is not None:
                split_prefix = 'test' if self.is_eval_dataset else 'train'
                prefix = f'{split_prefix}/attention_map/umi_task_{self.task_name}'
                sim_video = wandb.Video(attention_and_rollout_vis_path, format="mp4")
                log_data[prefix] = sim_video

        print(f"=== Finished UmiPromptingRunner run for task \"{self.task_name}\" ===\n")

        return log_data


