import os
from typing import Dict, List, Optional, Tuple
import cv2
import torch
import numpy as np
import copy

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import SequenceSampler, get_train_mask, get_training_split_info_from_train_mask
from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.train_network.model.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
    array_to_stats,
)
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset
from behavior_prompting.train_network.utils.draw_util import get_target_drawing_image_for_task_idx
from behavior_prompting.train_network.utils.dataset_util import prepare_only_task_names

class DrawImageDataset(BaseDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: Optional[str]=None,
            replay_buffer: Optional[ReplayBuffer]=None,
            seed=42,
            val_ratio=0.0,
            sample_type='task',
            max_segments=-1,
            action_padding=False,
            only_prompt:bool=False,
            only_goal_image:bool=False,
            training_split_info: Optional[Dict[str, bool]]=None,
            only_task_names: Optional[List[str]]=None,
            max_tasks: Optional[int]=None,
            num_training_demos_per_task: Optional[int]=None,
            receding_obs_augmentation_enabled: Optional[bool]=False,
            receding_obs_augmentation_probability: Optional[float]=0.2,
            receding_obs_augmentation_min_shapes: Optional[int]=1,
            receding_obs_augmentation_max_shapes: Optional[int]=3,
            receding_obs_augmentation_debug: Optional[bool]=False
            ):
        
        assert dataset_path is not None or replay_buffer is not None, 'either dataset_path or replay_buffer must be provided'

        if replay_buffer is not None:
            assert dataset_path is None, 'dataset_path and replay_buffer cannot both be provided'
            self.replay_buffer = replay_buffer
        else:
            assert dataset_path.endswith('.zarr'), 'dataset_path must be a .zarr folder, not a .zarr.zip file'
            self.replay_buffer = ReplayBuffer.create_from_path(dataset_path)

        only_task_names = prepare_only_task_names(self.replay_buffer, only_task_names, max_tasks=max_tasks)
        
        train_mask = get_train_mask(self.replay_buffer, sample_type, val_ratio, training_split_info, seed)

        # select only the task names in the replay buffer that are in the only_task_names list if provided
        buffer_task_names = self.replay_buffer.task_names[:]
        buffer_task_names_mask = np.isin(buffer_task_names, only_task_names)
        train_mask = train_mask & buffer_task_names_mask

        # Limit train mask to num_training_demos_per_task if specified
        if num_training_demos_per_task is not None and training_split_info is None:
            for task_name in only_task_names:
                task_indices = np.where((self.replay_buffer.task_names[:] == task_name) & train_mask)[0]
                assert len(task_indices) >= num_training_demos_per_task, f"Not enough training demos for task {task_name}"
                if len(task_indices) > num_training_demos_per_task:
                    train_mask[task_indices] = False # remove all train demos for this task temporarily
                    rng = np.random.RandomState(seed)
                    selected_indices = rng.choice(task_indices, size=num_training_demos_per_task, replace=False)
                    train_mask[selected_indices] = True

        # Extract keys from shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb" and attr.get('in_replay_buffer', True):
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
        
        # Check if goal image key is present
        self.use_prompting = shape_meta['use_prompting']
        self.use_goal_image = 'goal_image' in obs_shape_meta and not self.use_prompting

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.sample_type = sample_type
        self.max_segments = max_segments
        self.action_padding = action_padding
        self.only_prompt = only_prompt
        self.only_task_names = only_task_names
        self.only_goal_image = only_goal_image
        self.receding_obs_augmentation_enabled = receding_obs_augmentation_enabled
        self.receding_obs_augmentation_probability = receding_obs_augmentation_probability
        self.receding_obs_augmentation_min_shapes = receding_obs_augmentation_min_shapes
        self.receding_obs_augmentation_max_shapes = receding_obs_augmentation_max_shapes
        self.receding_obs_augmentation_debug = receding_obs_augmentation_debug

        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': self.action_padding,
            'sample_type': self.sample_type,
            'only_prompt': self.only_prompt,
            'only_goal_image': self.only_goal_image,
            'max_segments': self.max_segments,
            'seed': seed
        }

        sampler = SequenceSampler(
            mask=self.train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        new_train_mask = ~self.train_mask
        # select only the task names in the replay buffer that are in the only_task_names list if provided
        buffer_task_names = self.replay_buffer.task_names[:]
        buffer_task_names_mask = np.isin(buffer_task_names, self.only_task_names)
        new_train_mask = new_train_mask & buffer_task_names_mask

        val_set.sampler = SequenceSampler(
            mask=new_train_mask,
            **self.sampler_kwargs
        )

        val_set.train_mask = new_train_mask
        val_set.receding_obs_augmentation_enabled = False
        return val_set

    def get_normalizer(self, **kwargs) -> Normalizer:
        normalizer = Normalizer()

        # Action normalizer - use range normalization to scale to [-1, 1] like PushT
        stat = array_to_stats(self.replay_buffer.data['action'])
        normalizer['action'] = get_range_normalizer_from_stat(stat)

        # Agent position normalizer - use range normalization
        stat = array_to_stats(self.replay_buffer.data['agent_pos'])
        normalizer['agent_pos'] = get_range_normalizer_from_stat(stat)
        
        # Pen down normalizer - use range normalization to scale to [-1, 1] (will be previously 0 to 1)
        stat = array_to_stats(self.replay_buffer.data['pen_down'])
        normalizer['pen_down'] = get_range_normalizer_from_stat(stat)

        # Image normalizer (0-1 range)
        normalizer['image'] = get_image_identity_normalizer()

        # Goal image normalizer (0-1 range)
        if self.use_goal_image:
            normalizer['goal_image'] = get_image_identity_normalizer()

        prompt_normalizer = copy.deepcopy(normalizer)
        prompt_normalizer.set_prompt_normalizer(None)

        normalizer.set_prompt_normalizer(prompt_normalizer) # use the same normalizer for prompt and receding obs

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        
        def prepare_obs_dict(data):
            # Handle image data
            if 'image' in data:
                # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0,1]
                image = data['image'].astype(np.float32) / 255.0
                image = np.moveaxis(image, -1, 1)  # Move channel dimension
                data['image'] = image

            # Handle agent position
            if 'agent_pos' in data:
                data['agent_pos'] = data['agent_pos'].astype(np.float32)
            
            # Handle pen down state
            if 'pen_down' in data:
                data['pen_down'] = data['pen_down'].astype(np.float32)

        current_obs_present = any(key in data for key in self.rgb_keys + self.lowdim_keys)

        if current_obs_present:
            if self.receding_obs_augmentation_enabled:
                data['image'] = self._overlay_lines_on_image(data['image'])

            prepare_obs_dict(data)
        if 'prompt' in data:
            prepare_obs_dict(data['prompt']['obs'])

        if 'goal_image' in data:
            # data['goal_image'] is (1, H, W, C)
            data['goal_image'] = data['goal_image'].astype(np.float32) / 255.0
            data['goal_image'] = np.moveaxis(data['goal_image'], -1, 1) # (1, H, W, C) -> (1, C, H, W)

        # Convert to torch tensors
        # action and metadata
        action = data.pop('action', None)
        metadata = data.pop('metadata', {})

        # convert to torch
        torch_data = {
            "obs": dict_apply(data, torch.from_numpy),
            "metadata": metadata
        }
        if action is not None:
            torch_data['action'] = torch.from_numpy(action.astype(np.float32))
        return torch_data

    def _overlay_lines_on_image(self, image: np.ndarray) -> np.ndarray:
        """
        Takes image of shape (T, H, W, C) 0-255 uint8 and returns a new image with same shape/dtype but with random lines overlayed on top of the image.
        This is to simulate the case where the agent has made a mistake on the drawing, but we still want to keep drawing correctly despite a mistake in the view.
        The same random lines are overlayed across all timesteps to make the error show up consistently in the history of the observation.
        """
        if np.random.random() < self.receding_obs_augmentation_probability:
            assert type(image) == np.ndarray, f'image must be a numpy array, got {type(image)}' # make sure it's numpy array instead of being directly linked to the dataset so we don't accidentally modify the dataset
            num_shapes = np.random.randint(self.receding_obs_augmentation_min_shapes, self.receding_obs_augmentation_max_shapes + 1)
            for _ in range(num_shapes):
                shape_type = np.random.choice(['line', 'oval'])
                if shape_type == 'line':
                    start_x = np.random.randint(0, image.shape[2])
                    start_y = np.random.randint(0, image.shape[1])
                    end_x = np.random.randint(0, image.shape[2])
                    end_y = np.random.randint(0, image.shape[1])
                    for t in range(image.shape[0]):
                        image[t] = cv2.line(image[t], (start_x, start_y), (end_x, end_y), (0, 0, 255), 4)
                elif shape_type == 'oval':
                    center_x = np.random.randint(0, image.shape[2])
                    center_y = np.random.randint(0, image.shape[1])
                    # Sample oval size as proportions of image dimensions
                    prop_x = np.random.uniform(0.1, 0.8)
                    prop_y = np.random.uniform(0.1, 0.8)
                    axes_x = int(image.shape[2] * prop_x / 2)  # Convert to semi-axis
                    axes_y = int(image.shape[1] * prop_y / 2)  # Convert to semi-axis
                    angle = np.random.randint(0, 180)
                    # Sample a portion of the ellipse (arc)
                    start_angle = np.random.randint(0, 360)
                    arc_length = np.random.randint(30, 270)  # Arc length between 30 and 270 degrees
                    end_angle = (start_angle + arc_length) % 360
                    for t in range(image.shape[0]):
                        image[t] = cv2.ellipse(image[t], (center_x, center_y), (axes_x, axes_y), angle, start_angle, end_angle, (0, 0, 255), 4)

            if self.receding_obs_augmentation_debug:
                random_id = np.random.randint(0, 1000000)
                out_dir = f'tmp_augmentation'
                os.makedirs(out_dir, exist_ok=True)
                for t in range(image.shape[0]):
                    # Save image to disk with random identifier and timestep
                    path = f'{out_dir}/tmp_{random_id}_{t}.png'
                    cv2.imwrite(path, image[t, ::, ::, ::-1])
                    print(f'Saved augmented image to {path}')

        return image
    
    def get_target_drawing_image_for_task_idx(self, task_idx: int) -> Tuple[np.ndarray, float]:
        return get_target_drawing_image_for_task_idx(self.replay_buffer, task_idx)

    def shuffle_data_ordering(self, seed: int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def is_multi_task(self) -> bool:
        return True

    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        return self.sampler.get_unique_task_name_to_dataset_indices()

    def get_training_split_info(self) -> Dict[str, bool]:
        return get_training_split_info_from_train_mask(self.replay_buffer, self.sample_type, self.train_mask)

    def set_ignore_prompt(self, ignore_prompt: bool):
        self.sampler.set_ignore_prompt(ignore_prompt)

    def get_ignore_prompt(self) -> bool:
        return self.sampler.get_ignore_prompt()
