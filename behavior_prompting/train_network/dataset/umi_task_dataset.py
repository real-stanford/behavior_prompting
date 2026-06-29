import copy
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from behavior_prompting.train_network.model.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from behavior_prompting.train_network.common.pose_repr_util import convert_pose_mat_rep
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.replay_buffer_util import load_replay_buffer_lmdb
from behavior_prompting.train_network.common.sampler import get_train_mask, get_training_split_info_from_train_mask
from behavior_prompting.train_network.common.sampler import SequenceSampler
from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.common.pose_util import pose_to_mat, mat_to_pose10d
from transformers import CLIPTokenizer
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset

    
class UmiTaskDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: Optional[str]=None,
        replay_buffer: Optional[ReplayBuffer]=None,
        text_encoder_model_name: Optional[str]=None,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=True,
        action_padding_error_correction: bool=False,
        only_prompt:bool=False,
        only_goal_image:bool=False,
        training_split_info: Optional[Dict[str, bool]]=None,
        temporally_independent_normalization: bool=False,
        max_normalizer_iterations: Optional[int]=None,
        seed: int=42,
        val_ratio: float=0.0,
        include_labels: bool=False,
        sample_type: str='task',
        max_segments: int=-1,
        prompt_relative_to: str='last', # 'last' or 'start',
        allow_zip_file: bool=False, # during training you need to unzip the dataset to ensure multiple processes can read from it properly, but if you only plan to have one worker reading from it (e.g. for evaluation) then you can set this to True and save disk space by not unzipping the dataset,
        **kwargs
    ):
        # TODO: Need to update this to not used fixed language embeddings as it does not work too well (many different language tasks have similar embeddings making it hard for the policy to distinguish between them). Instead you should update this tofinetune the language encoding as I do for libero
        assert dataset_path is not None or replay_buffer is not None, 'either dataset_path or replay_buffer must be provided'

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'relative')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'relative')

        if replay_buffer is None:
            assert dataset_path is not None, 'dataset_path and replay_buffer cannot both be provided'
            if cache_dir is None:
                assert allow_zip_file or dataset_path.endswith('.zarr'), 'dataset_path must be a .zarr folder, not a .zarr.zip file'
                replay_buffer = ReplayBuffer.create_from_path(dataset_path)
            else:
                replay_buffer = load_replay_buffer_lmdb(dataset_path, cache_dir)
        
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        language_keys = list()
        for key, attr in shape_meta['obs'].items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            is_language = attr.get('is_language', False)
            if is_language:
                language_keys.append(key)
                lowdim_keys.append(key)
                continue
            elif type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

        self.robot_prefixes = sorted([
            key[:-len('_eef_pos')]
            for key, attr in shape_meta['obs'].items()
            if key.endswith('_eef_pos') and 'wrt' not in key
        ])
        assert len(self.robot_prefixes) == self.num_robot

        n_segments = replay_buffer.n_episodes if sample_type == 'episode' else replay_buffer.n_tasks
        if max_segments > 0:
            n_segments = min(n_segments, max_segments)

        train_mask = get_train_mask(replay_buffer, sample_type, val_ratio, training_split_info, seed)
        
        # compute language embedding
        if language_keys:
            for key in language_keys:
                assert shape_meta['obs'][key]['horizon'] == 1, f'{key} horizon must be 1'
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
        else:
            self.clip_tokenizer = None
            
        # goal image
        self.use_goal_image = 'goal_image' in shape_meta['obs']
        assert not (only_goal_image and not self.use_goal_image), 'cannot set only_goal_image to True if goal_image is not in shape_meta'
        self.only_goal_image = only_goal_image

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.language_keys = language_keys
        self.train_mask = train_mask
        self.action_padding = action_padding
        self.action_padding_error_correction = action_padding_error_correction
        self.temporally_independent_normalization = temporally_independent_normalization
        self.max_normalizer_iterations = max_normalizer_iterations
        self.include_labels = include_labels
        self.sample_type = sample_type
        self.use_prompting=self.shape_meta.use_prompting
        self.max_segments = max_segments
        self.only_prompt = only_prompt
        self.prompt_relative_to = prompt_relative_to

        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': self.action_padding,
            'action_padding_error_correction': self.action_padding_error_correction,
            'sample_type': self.sample_type,
            'only_prompt': self.only_prompt,
            'only_goal_image': self.only_goal_image,
            'max_segments': self.max_segments,
            'include_labels': self.include_labels,
            'seed': seed
        }

        sampler = SequenceSampler(
            mask=train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.train_mask = ~self.train_mask
        
        val_set.sampler = SequenceSampler(
            mask=val_set.train_mask,
            **self.sampler_kwargs
        )

        return val_set
    
    def get_normalizer(self, **kwargs) -> Normalizer:
        assert not self.only_goal_image and not self.only_prompt

        normalizer = Normalizer()

        """Compute the normalizer for the receding obs"""

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        previous_ignore_rgb = self.sampler.ignore_rgb_is_applied
        previous_ignore_prompt = self.sampler.get_ignore_prompt()
        self.sampler.ignore_rgb(True)
        self.sampler.set_ignore_prompt(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        tqdm_total = self.max_normalizer_iterations if self.max_normalizer_iterations is not None else None
        for i, batch in enumerate(tqdm(dataloader, total=tqdm_total, desc='iterating dataset to get normalization')):
            if self.max_normalizer_iterations is not None and i >= self.max_normalizer_iterations:
                break
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(previous_ignore_rgb)
        self.sampler.set_ignore_prompt(previous_ignore_prompt)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            if self.max_normalizer_iterations is None:
                assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            if key in self.language_keys:
                continue
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # language: skip normalization (token IDs are not normalized)

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()

        if self.use_goal_image:
            normalizer['goal_image'] = get_image_identity_normalizer()

        """Compute the normalizer for the prompt"""
        # for now the prompt normalizer is the same as the receding normalizer but only with keys that we expect to be in the prompt. The action normalizer is currently on supported if we are not using temporally independent normalization.
        assert not self.temporally_independent_normalization, 'temporally independent normalization is not supported for prompt since the action trajectory in the prompt chunks might not match the action trajectory in the action prediction horizon'
        prompt_normalizer = Normalizer()
        
        # prompt uses same rgb and action normalizer as receding
        for key in self.rgb_keys + ['action']:
            prompt_normalizer[key] = normalizer[key]

        # prompt uses same gripper width normalizer as receding (though by default we do not include gripper width in the prompt)
        for prefix in self.robot_prefixes:
            prompt_normalizer[f'{prefix}_gripper_width'] = normalizer[f'{prefix}_gripper_width']

        # prompt uses same between gripper relative pose normalizer as receding
        for prefix in self.robot_prefixes:
            for other_prefix in self.robot_prefixes:
                if prefix == other_prefix:
                    continue
                for wrt_key in [f'{prefix}_eef_pos_wrt_{other_prefix}', f'{prefix}_eef_rot_axis_angle_wrt_{other_prefix}']:
                    if wrt_key in self.lowdim_keys:
                        prompt_normalizer[wrt_key] = normalizer[wrt_key]

        normalizer.set_prompt_normalizer(prompt_normalizer)

        return normalizer
    
    def shuffle_data_ordering(self, seed:int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def is_multi_task(self) -> bool:
        return self.sample_type == 'task'

    def get_unique_task_name_to_dataset_indices(self, exclude_error_correction: bool = False) -> Dict[str, list[int]]:
        return self.sampler.get_unique_task_name_to_dataset_indices(exclude_error_correction=exclude_error_correction)
    
    def get_training_split_info(self) -> Dict[str, bool]:
        return get_training_split_info_from_train_mask(self.replay_buffer, self.sample_type, self.train_mask)
    
    def set_ignore_prompt(self, ignore_prompt: bool):
        self.sampler.set_ignore_prompt(ignore_prompt)

    def get_ignore_prompt(self) -> bool:
        return self.sampler.get_ignore_prompt()

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if self.only_goal_image:
            data['obs'] = dict()
            data['obs']['goal_image'] = np.moveaxis(data.pop('goal_image'), -1, 1).astype(np.float32) / 255.
            torch_data = {
                'obs': dict_apply(data['obs'], torch.from_numpy),
                'metadata': data['metadata']
            }
            return torch_data
        
        obs_dict = dict()

        def prepare_obs_dict(data, dest_dict):
            # Handle image data
            for key in self.rgb_keys + ['goal_image']:
                if not key in data:
                    continue
                # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0,1]
                image = data[key].astype(np.float32) / 255.0
                image = np.moveaxis(image, -1, 1)  # Move channel dimension
                dest_dict[key] = image
                del data[key]

            for key in self.lowdim_keys:
                if self.shape_meta['obs'][key].get('in_replay_buffer', True) and key not in self.language_keys and key in data:
                    dest_dict[key] = data[key].astype(np.float32)
                    del data[key]

        receding_present = any(key in data for key in self.rgb_keys + self.lowdim_keys)
        prompt_present = 'prompt' in data

        metadata = dict()

        if receding_present:
            """
            The additional processing for receding data is to:
            - add language encoding if present
            - compute relative pose between eefs
            - compute relative pose with respect to the episode or task start pose
            - compute relative eef pose for the history with respect to the current eef pose
            - compute relative action trajectory with respect to the current eef pose
            """
            prepare_obs_dict(data, obs_dict)

            # language encoding
            for key in self.language_keys:
                if key == 'task_language':
                    task_name = self.replay_buffer.task_names[data['metadata']['task_idx']]
                    tokens = self.clip_tokenizer(
                        task_name,
                        padding='max_length',
                        truncation=True,
                        max_length=77,
                        return_tensors='np'
                    )
                    obs_dict[key] = tokens['input_ids'].astype(np.int64)  # (1, 77)
                else:
                    raise NotImplementedError(f'language key `{key}` not supported')
            
            # generate relative pose between two ees
            for prefix in self.robot_prefixes:
                # convert pose to mat
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'{prefix}_eef_pos'],
                    obs_dict[f'{prefix}_eef_rot_axis_angle']
                ], axis=-1))
                for other_prefix in self.robot_prefixes:
                    if prefix == other_prefix:
                        continue
                    if not f'{prefix}_eef_pos_wrt_{other_prefix}' in self.lowdim_keys:
                        continue
                    other_pose_mat = pose_to_mat(np.concatenate([
                        obs_dict[f'{other_prefix}_eef_pos'],
                        obs_dict[f'{other_prefix}_eef_rot_axis_angle']
                    ], axis=-1))
                    rel_obs_pose_mat = convert_pose_mat_rep(
                        pose_mat,
                        base_pose_mat=other_pose_mat[-1],
                        pose_rep='relative',
                        backward=False)
                    rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                    obs_dict[f'{prefix}_eef_pos_wrt_{other_prefix}'] = rel_obs_pose[:,:3]
                    obs_dict[f'{prefix}_eef_rot_axis_angle_wrt_{other_prefix}'] = rel_obs_pose[:,3:]
                    
            # generate relative pose with respect to episode or task start
            for prefix in self.robot_prefixes:
                if (f'{prefix}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                    (f'{prefix}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                    continue

                # convert pose to mat
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'{prefix}_eef_pos'],
                    obs_dict[f'{prefix}_eef_rot_axis_angle']
                ], axis=-1))

                # get start pose for the start of the episode/task
                if self.sample_type == 'episode':
                    # though this is stored in the replay buffer (demo_start_pose, demo_end_pose) we instead just compute it ourselves by looking at the first episode entry. This is done because it's just easier to maintain consistency with task sampling in which case there is no (task_start_pose, task_end_pose)
                    episode_idx = data['metadata']['episode_idx']
                    episode_data_start_idx = self.replay_buffer.episode_ends[episode_idx] - self.replay_buffer.episode_lengths[episode_idx]
                    start_pos = self.replay_buffer[f'{prefix}_eef_pos'][episode_data_start_idx]
                    start_rot_axis_angle = self.replay_buffer[f'{prefix}_eef_rot_axis_angle'][episode_data_start_idx]
                    start_pose = np.concatenate([start_pos, start_rot_axis_angle])
                elif self.sample_type == 'task':
                    # not explicitly stored in replay buffer, but we can just compute it
                    task_idx = data['metadata']['task_idx']
                    task_data_start_idx = self.replay_buffer.task_data_ends[task_idx] - self.replay_buffer.task_lengths[task_idx]
                    start_pos = self.replay_buffer[f'{prefix}_eef_pos'][task_data_start_idx]
                    start_rot_axis_angle = self.replay_buffer[f'{prefix}_eef_rot_axis_angle'][task_data_start_idx]
                    start_pose = np.concatenate([start_pos, start_rot_axis_angle])

                # HACK: add noise to episode start pose
                start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
                start_pose_mat = pose_to_mat(start_pose)
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=start_pose_mat,
                    pose_rep='relative',
                    backward=False)

                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)

                # we do not typically want eef pos wrt start, just eef rotation wrt start
                # if f'{prefix}_eef_pos_wrt_start' in self.shape_meta['obs']:
                #     obs_dict[f'{prefix}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
                if f'{prefix}_eef_rot_axis_angle_wrt_start' in self.shape_meta['obs']:
                    obs_dict[f'{prefix}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

            del_keys = list()
            for key in obs_dict:
                if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                    del_keys.append(key)
            for key in del_keys:
                del obs_dict[key]

            # compute action and eef pose in requested (likely relative) representation (starting from absolute)
            actions = list()
            for robot_idx, prefix in enumerate(self.robot_prefixes):
                # convert pose to mat
                pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'{prefix}_eef_pos'],
                    obs_dict[f'{prefix}_eef_rot_axis_angle']
                ], axis=-1))
                action_mat = pose_to_mat(data['action'][...,7 * robot_idx: 7 * robot_idx + 6])

                # solve relative obs
                obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=pose_mat[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                action_pose_mat = convert_pose_mat_rep(
                    action_mat,
                    base_pose_mat=pose_mat[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)

                # convert pose to pos + rot6d representation
                obs_pose = mat_to_pose10d(obs_pose_mat)
                action_pose = mat_to_pose10d(action_pose_mat)

                action_gripper = data['action'][..., 7 * robot_idx + 6: 7 * robot_idx + 7]
                actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

                # generate data
                obs_dict[f'{prefix}_eef_pos'] = obs_pose[:,:3]
                obs_dict[f'{prefix}_eef_rot_axis_angle'] = obs_pose[:,3:]
                
            data['action'] = np.concatenate(actions, axis=-1).astype(np.float32)

            # additional metadata for debugging
            metadata['episode_idx'] = torch.tensor(data['metadata']['episode_idx'], dtype=torch.int64)

            # add task_idx if using task sampling
            if self.sample_type == 'task':
                metadata['task_idx'] = torch.tensor(data['metadata']['task_idx'], dtype=torch.int64)
            if self.use_goal_image:
                metadata['goal_image_task_idx'] = torch.tensor(data['metadata']['goal_image_task_idx'], dtype=torch.int64)

        if prompt_present:
            """
            The additional processing for prompt data is to:
            - compute relative action trajectory with respect to the eef pose at the start of each prompt chunk
            - compute relative pose between eefs at each prompt chunk
            - TODO: theoretically we could also support putting eef rotation with respect to start as prompt attributes as well
            """
            obs_dict['prompt'] = {
                'obs': dict(),
                'metadata': data['prompt']['metadata']
            }
            prompt_dict = obs_dict['prompt']
            prepare_obs_dict(data['prompt']['obs'], prompt_dict['obs'])

            # compute action and eef pose in requested (likely relative) representation (starting from absolute). Also handles conversion from 7d action to 10d action
            actions = list()
            prompt_pose_mats = {}  # {prefix: (T, 4, 4)} saved before popping for cross-arm computation

            for robot_idx, prefix in enumerate(self.robot_prefixes):
                # convert pose to mat
                pose_mat = pose_to_mat(np.concatenate([
                    prompt_dict['obs'][f'{prefix}_eef_pos'],
                    prompt_dict['obs'][f'{prefix}_eef_rot_axis_angle']
                ], axis=-1)) # (T, 4, 4)

                prompt_pose_mats[prefix] = pose_mat  # save full (T, 4, 4) before any slicing

                if self.prompt_relative_to == 'last':
                    pass
                elif self.prompt_relative_to == 'start':
                    pose_mat = np.expand_dims(pose_mat[0], axis=0) # (1, 4, 4)
                else:
                    raise ValueError(f'prompt_relative_to must be one of `last` or `start`, got {self.prompt_relative_to}')

                prompt_actions = data['prompt']['action'][...,7 * robot_idx: 7 * robot_idx + 6] # (T, chunk_n_actions, 6)
                T, chunk_n_actions, _ = prompt_actions.shape
                action_mat = pose_to_mat(prompt_actions.reshape(T*chunk_n_actions, 6)).reshape(T, chunk_n_actions, 4, 4) # (T, chunk_n_actions, 4, 4)

                # solve relative obs; specifically for each timestep of the pose_mat (current pose) we want to multiply the sequence/chunk of actions from the corresponding timestep of the action_mat. So the idea is we have a new base pose for each timestep and all the intermediate actions are relative to that base pose.
                action_pose_mat = convert_pose_mat_rep(
                    action_mat, # (T, chunk_n_actions, 4, 4)
                    base_pose_mat=np.expand_dims(pose_mat, axis=1), # (T, 1, 4, 4) for last, (1, 1, 4, 4) for start
                    pose_rep=self.obs_pose_repr,
                    backward=False)

                # convert pose to pos + rot6d representation
                action_pose = mat_to_pose10d(action_pose_mat) # (T, chunk_n_actions, 9)

                action_gripper = data['prompt']['action'][..., 7 * robot_idx + 6: 7 * robot_idx + 7]
                actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

                # the eef_pos and eef_rot_axis_angle in the prompt are only needed for computing relative action within each chunk. We now remove them from the prompt
                assert self.shape_meta['obs'][f'{prefix}_eef_pos']['prompt_type'] == 'ignore'
                assert self.shape_meta['obs'][f'{prefix}_eef_rot_axis_angle']['prompt_type'] == 'ignore'
                prompt_dict['obs'].pop(f'{prefix}_eef_pos')
                prompt_dict['obs'].pop(f'{prefix}_eef_rot_axis_angle')

            # compute cross-arm relative poses for prompt (per prompt timestep, using saved pose mats)
            for prefix in self.robot_prefixes:
                for other_prefix in self.robot_prefixes:
                    if prefix == other_prefix:
                        continue
                    wrt_pos_key = f'{prefix}_eef_pos_wrt_{other_prefix}'
                    wrt_rot_key = f'{prefix}_eef_rot_axis_angle_wrt_{other_prefix}'
                    if wrt_pos_key not in self.lowdim_keys:
                        continue
                    # per-timestep: (T,4,4) @ (T,4,4) via numpy batched matmul
                    rel_pose_mat = convert_pose_mat_rep(
                        prompt_pose_mats[prefix],          # (T, 4, 4)
                        base_pose_mat=prompt_pose_mats[other_prefix],  # (T, 4, 4) per-timestep base
                        pose_rep=self.obs_pose_repr,
                        backward=False)                    # (T, 4, 4)
                    rel_pose = mat_to_pose10d(rel_pose_mat)  # (T, 9)
                    prompt_dict['obs'][wrt_pos_key] = rel_pose[:, :3]
                    prompt_dict['obs'][wrt_rot_key] = rel_pose[:, 3:]

            prompt_dict['action'] = np.concatenate(actions, axis=-1)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'metadata': metadata
        }

        if 'action' in data:
            torch_data['action'] = torch.from_numpy(data['action'].astype(np.float32))

        if self.include_labels:
            torch_data['labels'] = dict_apply(data['labels'], torch.from_numpy)

        return torch_data
