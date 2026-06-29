"""Modified from UMI codebase to support sampling by tasks and/or with prompting. Sequence prompting implementation adapted from ICRT."""

from typing import Dict, Optional, List, Union
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker
from behavior_prompting.common.np_util import add_batch_dim, remove_batch_dim

def get_train_mask(replay_buffer: ReplayBuffer, sample_type: str, val_ratio: float, training_split_info: Optional[Dict[str, bool]]=None, seed: int=0) -> np.ndarray:
    if training_split_info is None:
        val_mask = get_val_mask(
            replay_buffer=replay_buffer,
            sample_type=sample_type,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
    else:
        train_mask = get_train_mask_from_training_split_info(replay_buffer, training_split_info, sample_type)

    return train_mask

def get_val_mask(replay_buffer: ReplayBuffer, sample_type, val_ratio, seed=0):
    """Gets the validation mask according to the replay_buffer and sample_type. Applies the val_ratio per task to ensure that the validation set will contain at least 1 episode for each task if using task sampling"""
    assert val_ratio >= 0 and val_ratio <= 1, 'val_ratio must be between 0 and 1'

    if sample_type == 'episode':
        n_segments = replay_buffer.n_episodes
        val_mask = np.zeros(n_segments, dtype=bool)
        if val_ratio <= 0:
            return val_mask

        n_val = min(max(1, round(n_segments * val_ratio)), n_segments)
        rng = np.random.default_rng(seed=seed)
        val_idxs = rng.choice(n_segments, size=n_val, replace=False)
        val_mask[val_idxs] = True

        if val_ratio > 0:
            assert np.any(val_mask), 'at least one episode must be selected for validation if val_ratio > 0'
        if val_ratio < 1:
            assert np.any(~val_mask), 'at least one episode must be selected for training if val_ratio < 1'

        return val_mask
    elif sample_type == 'task':
        n_segments = replay_buffer.n_tasks
        val_mask = np.zeros(n_segments, dtype=bool)
        if val_ratio <= 0:
            return val_mask
        
        unique_task_names = np.unique(replay_buffer.task_names)
        for task_name in unique_task_names:
            task_indices = np.where(replay_buffer.task_names[:] == task_name)[0]

            # Only select val demos from non-EC tasks so both splits always have at least one full demo.
            # EC demos always stay in train since they are partial recovery demos, not full task executions.
            val_candidate_indices = task_indices
            is_ec = replay_buffer.is_task_error_correction
            if is_ec is not None:
                non_ec_indices = task_indices[~is_ec[task_indices].astype(bool)]
                assert len(non_ec_indices) > 0, (
                    f"Task '{task_name}' has only EC demos — no full demo available for val split"
                )
                val_candidate_indices = non_ec_indices

            n_segments_cur_task = len(val_candidate_indices)

            n_val = min(max(1, round(n_segments_cur_task * val_ratio)), n_segments_cur_task)

            if val_ratio > 0:
                assert n_val > 0, 'at least one episode must be selected for validation if val_ratio > 0'
            if val_ratio < 1:
                assert n_val < n_segments_cur_task, (
                    f"Task '{task_name}': val_ratio < 1 requires at least 1 non-error-correction "
                    "remaining to put in train."
                )

            rng = np.random.default_rng(seed=seed)
            val_idxs = rng.choice(val_candidate_indices, size=n_val, replace=False)
            val_mask[val_idxs] = True

        return val_mask

def get_training_split_info_from_train_mask(replay_buffer: ReplayBuffer, sample_type: str, train_mask: np.ndarray) -> Dict[str, bool]:
    """the training split dictionary is a format that contains information about each dataset entry (as identified by demonstration name and task name) and indicates whether it's in the training split. It's a good format to identify which tasks are in the training split in a portable format."""
    if sample_type == 'episode':
        assert replay_buffer.n_episodes == len(train_mask), 'the length of the train mask must match the number of episodes'
        result = {}
        for i in range(len(replay_buffer.episode_ends)):
            result[replay_buffer.episode_names[i] + ":"] = train_mask[i]
        return result
    elif sample_type == 'task':
        result = {}
        task_to_episode_idxs = replay_buffer.get_task_to_episode_idxs()
        for i in range(replay_buffer.n_tasks):
            episode_idx = task_to_episode_idxs[i]
            episode_name = replay_buffer.episode_names[episode_idx]
            task_name = replay_buffer.task_names[i]
            result[episode_name + ":" + task_name] = train_mask[i]
        return result

def get_train_mask_from_training_split_info(replay_buffer: ReplayBuffer, training_split_info: Dict[str, bool], sample_type: str) -> np.ndarray:
    """Given a training split info dictionary, this function returns a mask indicating which dataset entries are in the training split. Notably you can pass a replay buffer that only contains a a subset of the entries in training_split_info and it will just select from dataset entries only present in replay_buffer."""
    if sample_type == 'episode':
        result = np.zeros(replay_buffer.n_episodes, dtype=bool)
        for i in range(replay_buffer.n_episodes):
            episode_name = replay_buffer.episode_names[i]
            task_name = ""
            result[i] = training_split_info[episode_name + ":" + task_name]
        return result
    elif sample_type == 'task':
        result = np.zeros(replay_buffer.n_tasks, dtype=bool)
        task_to_episode_idxs = replay_buffer.get_task_to_episode_idxs()

        # TODO: this is pretty slow when n_tasks is large, potentially there is some way to cache the result if we generate it many times like if we do rollout many times on the same dataset for different tasks

        for i in range(replay_buffer.n_tasks):
            episode_idx = task_to_episode_idxs[i]
            episode_name = replay_buffer.episode_names[episode_idx]
            task_name = replay_buffer.task_names[i]
            result[i] = training_split_info[episode_name + ":" + task_name]
        return result

def get_training_task_names_from_training_split_info(training_split_info: Dict[str, bool]) -> List[str]:
    """Returns a list of task names that are in the training split. This is useful for checking if a task is in the training split."""
    result = []
    for key, is_in_training_split in training_split_info.items():
        if is_in_training_split:
            new_value = key.split(":")[1]
            assert new_value != "", "task name cannot be empty, check that you are actually using task sampling"
            result.append(new_value)
    return list(set(result))

def convert_multi_step(data : np.ndarray, num_pred_steps: int):
    """Chunk data for predicting data `num_pred_steps` steps into the future.
    The resulting data have shape (batch, data.shape[-2] - (num_pred_steps - 1), num_pred_steps, action_dim)
    For example: chunk_data([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
            [a_4, a_5, a_5],
            [a_5, a_5, a_5],
        ]
    adapted from https://github.com/octo-models/octo/blob/7480a2a90160122b7a02459fc6f56ceefa501ebf/octo/model/components/action_heads.py#L59
    """
    assert (
        data.ndim == 2
    ), f"Expected data to have shape (seq length, action_dim), but got shape {data.shape}"
    window_size = data.shape[0]
    chunk_window_size = window_size

    curr_step = np.arange(chunk_window_size)
    action_offset = np.arange(num_pred_steps)
    chunk_indices = np.minimum(curr_step[:, None] + action_offset[None, :], np.array(chunk_window_size - 1))
    return data[chunk_indices]

class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        action_padding_error_correction: bool=False,
        sample_type='episode', # 'episode' or 'task'
        only_prompt:Optional[bool]=False,
        only_goal_image:Optional[bool]=False,
        max_segments: int=-1,
        include_labels: bool=False,
        action_key: str='action',
        seed:int=0
    ):
        assert sample_type in ['episode', 'task']

        # label keys
        if include_labels:
            assert sample_type == 'task', 'labels only supported for task sampling'
            labels_shape_meta = shape_meta.get('labels', {})
            labels_keys = list(labels_shape_meta.keys())
        else:
            labels_keys = []
        
        # obs keys
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb" and attr.get('in_replay_buffer', True):
                rgb_keys.append(key)
            elif type == "low_dim" and attr.get('in_replay_buffer', True):
                lowdim_keys.append(key)

        # horizon and down sample steps
        key_horizon = dict()
        key_down_sample_steps = dict()
        for key, attr in obs_shape_meta.items():
            is_in_replay_buffer = attr.get('in_replay_buffer', True)
            if not is_in_replay_buffer:
                continue
            key_horizon[key] = shape_meta['obs'][key]['horizon']
            key_down_sample_steps[key] = shape_meta['obs'][key]['down_sample_steps']

        key_horizon['action'] = shape_meta['action']['horizon']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        # prediction horizon
        num_pred_steps = shape_meta['action']['horizon']
        
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        self.mask = mask
        self.action_padding = action_padding
        self.action_padding_error_correction = action_padding_error_correction
        self.sample_type = sample_type
        self.only_prompt = only_prompt
        self.only_goal_image = only_goal_image
        self.ignore_prompt = False
        self.labels_keys = labels_keys
        self.max_segments = max_segments
        self.action_key = action_key
        self.num_pred_steps = num_pred_steps

        self.use_prompting = shape_meta.get('use_prompting', False)
        self.prompt_sequence_length = shape_meta.get('prompt_sequence_length', -1)
        self.sequence_prompting_min_steps_per_task = shape_meta.get('sequence_prompting_min_steps_per_task', None)
        self.max_prompt_full_demos = shape_meta.get('max_prompt_full_demos', -1)
        self.prompt_sample_mode = shape_meta.get('prompt_sample_mode', None)

        if self.only_goal_image:
            # if you request only_goal_image then we can disable prompting because we assume that you just want goal images. This is useful in evaluation settings where you might need goal images even if you have a prompting based policy
            self.use_prompting = False
        if self.only_prompt:
            assert self.use_prompting, 'only_prompt is only supported when use_prompting is True'

        self.use_goal_image = 'goal_image' in shape_meta['obs'] and (not shape_meta['obs']['goal_image'].get('ignore_by_policy', False) or self.only_goal_image) # if you use only_goal_image mode then you can still retrieve goal images from the sampler despite ignore_by_policy being True
        if self.use_goal_image:
            self.goal_image_key = shape_meta['obs']['goal_image']['key']
            is_ec = replay_buffer.is_task_error_correction
            if is_ec is not None and np.any(is_ec):
                raise NotImplementedError(
                    "Goal image sampling is not yet supported for datasets that contain error correction demonstrations. This is because we would need to add some way to figure out what a proper goal image is for an error correction demonstration since these demos are only partial task executions."
                )
        
        if self.only_prompt:
            assert self.sample_type == 'task', 'only_prompt and only_goal_image are only supported for task sampling'

        self.prepare_replay_buffer()

        if self.sample_type == 'task':
            self.task_to_episode_idxs = self.replay_buffer.get_task_to_episode_idxs()

        self.setup_indices(seed)

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizer

        if self.use_prompting:
            self.prompt_action_chunker = PromptActionChunker(shape_meta)
        
    def prepare_replay_buffer(self):
        self.prepared_replay_buffer = dict()

        # action
        if self.action_key in self.replay_buffer:
            action = self.replay_buffer[self.action_key]
        else:
            # default action construction (used in UMI)
            # TODO: probably should move this UMI specific logic outside of the sampler
            robot_prefixes = sorted([
                key[:-len('_eef_pos')]
                for key in self.lowdim_keys
                if key.endswith('_eef_pos') and 'wrt' not in key
            ])
            actions = list()
            for prefix in robot_prefixes:
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'{prefix}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            action = np.concatenate(actions, axis=-1)
        self.prepared_replay_buffer['action'] = action

        # lowdim, rgb, and labels
        for key in self.lowdim_keys:
            self.prepared_replay_buffer[key] = self.replay_buffer[key]
        for key in self.rgb_keys:
            self.prepared_replay_buffer[key] = self.replay_buffer[key]
        for key in self.labels_keys:
            self.prepared_replay_buffer[key] = self.replay_buffer.labels[key]

    def setup_indices(self, seed:int=0):
        # sets up the dataset sampling indices
        if self.use_prompting:
            if self.prompt_sample_mode == 'sequence':
                self.setup_indices_with_sequence_prompting(seed)
            elif self.prompt_sample_mode == 'pair':
                self.setup_indices_with_pair_prompting(seed)
            else:
                raise ValueError(f'prompt_sample_mode {self.prompt_sample_mode} not implemented')
        else:
            self.setup_indices_without_prompting()
    
    def setup_indices_without_prompting(self):
        if self.only_goal_image:
            valid_indices = np.where(self.mask)[0]
            self.indices = np.arange(len(valid_indices))
            self.goal_image_segment_indices = valid_indices
            return

        segment_ends = self.replay_buffer.episode_ends[:] if self.sample_type == 'episode' else self.replay_buffer.task_data_ends[:]

        # create indices, including (current_within_segment_idx, episode_idx, task_idx)
        start_of_segment_indices = {}
        end_of_segment_indices = {}
        indices = list()
        for i in range(len(segment_ends)):
            if len(start_of_segment_indices) == self.max_segments:
                break
            
            if self.mask is not None and not self.mask[i]:
                # skip episode/task
                continue
            end_data_idx = segment_ends[i]
            if self.sample_type == 'task':
                segment_length = self.replay_buffer.task_lengths[i]
                start_data_idx = end_data_idx - segment_length
            else:
                start_data_idx = 0 if i == 0 else segment_ends[i-1]
                segment_length = end_data_idx - start_data_idx

            if self.sample_type == 'task':
                task_idx = i
                episode_idx = self.task_to_episode_idxs[task_idx]
            else:
                task_idx = None
                episode_idx = i

            start_of_segment_indices[i] = len(indices)
            is_ec_segment = (
                self.sample_type == 'task'
                and self.replay_buffer.is_task_error_correction is not None
                and self.replay_buffer.is_task_error_correction[task_idx]
            )
            use_action_padding_for_this_segment = self.action_padding_error_correction if is_ec_segment else self.action_padding
            for current_data_idx in range(start_data_idx, end_data_idx):
                if not use_action_padding_for_this_segment and end_data_idx < current_data_idx + (self.key_horizon['action'] - 1) * self.key_down_sample_steps['action'] + 1:
                    # if action padding is not enabled and the action horizon would extend beyond the end of the segment, then we skip this index since it would not be a valid sample
                    continue
                
                current_within_segment_idx = current_data_idx - start_data_idx
                
                indices.append((current_within_segment_idx, episode_idx, task_idx))
            end_of_segment_indices[i] = len(indices)

        if len(indices) > 0:
            self.indices = np.array(indices)
        else:
            # support case where there are no indices (ex: empty validation set)
            self.indices = np.zeros((0, 3))

    def setup_indices_with_sequence_prompting(self, seed:int):
        if self.sample_type != 'task':
            raise NotImplementedError('prompting currently only supported for task sampling')
        
        # - `steps` is a reordering of the data so that identical task data is next to each other (this is done through an index mapping rather than sorting the data)
        # - `indices` contains indices into the `steps` array for which there is a region of length `self.prompt_sequence_length` that is valid
        # - so to step through the data, we can just sample from `indices` and then use the `steps` array to get the actual data indices we should access
        steps = []
        indices = []

        rng = np.random.RandomState(seed=seed)
        task_names = self.replay_buffer.task_names
        verb_names = np.unique(task_names)
        num_tasks_by_verb = {}
        num_tasks_by_verb_after_padding = {}

        sample_sequence_length = self.prompt_sequence_length + self.num_pred_steps - 1

        # go task by task through the data and build up the steps array
        # we define verb the the shared task name across all matching tasks
        for verb_name in verb_names:
            # get the task indices that correspond to this task name and within the mask
            mask = self.mask if self.mask is not None else 1
            task_indices = np.where((self.replay_buffer.task_names[:] == verb_name)*mask)[0]

            # Exclude EC tasks from sequence prompting entirely — they cannot be safely placed
            # in the prompt region given the continuous-stream structure.
            if self.replay_buffer.is_task_error_correction is not None:
                task_indices = task_indices[~self.replay_buffer.is_task_error_correction[task_indices].astype(bool)]

            if len(task_indices) == 0:
                # no tasks found for this verb, so just skip this verb
                continue

            assert len(task_indices) >= 2, f"at least 2 demos of a single task (task=\"{verb_name}\") are required to sample with prompting. Currently there is {len(task_indices)} demos."

            num_tasks_by_verb[verb_name] = len(task_indices)

            # find the total length of all these tasks
            original_task_indices = task_indices.copy()
            total_steps_for_verb = sum(self.replay_buffer.task_lengths[task_indices])

            # it's possible that the number of total steps for the verb is less than the minimum steps per task. In this case, we need to pad the verb by repeating tasks until we reach the minimum steps per task. Note that this may result in the same demonstration put multiple times in the prompt or multiple times in the rollout region or in both the prompt and the rollout region 
            while total_steps_for_verb < self.sequence_prompting_min_steps_per_task:
                task_indices = np.concatenate([task_indices, original_task_indices])
                total_steps_for_verb = sum(self.replay_buffer.task_lengths[task_indices])

            num_tasks_by_verb_after_padding[verb_name] = len(task_indices)
            # randomize the ordering of the task indices
            rng.shuffle(task_indices)
            assert self.prompt_sequence_length + self.num_pred_steps <= total_steps_for_verb

            verb_step_idx = 0

            # go through every task index with this same task name and build up the steps array
            for task_idx in task_indices:
                # go through every valid index in the task
                segment_length = self.replay_buffer.task_lengths[task_idx]

                end_data_idx = self.replay_buffer.task_data_ends[task_idx]                
                start_data_idx = end_data_idx - segment_length
                episode_idx = self.task_to_episode_idxs[task_idx]
                
                for current_data_idx in range(start_data_idx, end_data_idx):                    
                    current_within_segment_idx = current_data_idx - start_data_idx
                    steps.append((current_within_segment_idx, episode_idx, task_idx))
                    
                    if verb_step_idx + self.prompt_sequence_length + self.num_pred_steps <= total_steps_for_verb:
                        # if we are within the prompt sequence length including our action horizon, then we can add this index to `indices`
                        indices.append(len(steps) - 1)
                    verb_step_idx += 1
        
        steps = np.array(steps)

        # need to also check that if we sampled each index there would be enough room to fit a the non-prompt region (which is at minimum including at least one full demonstration in the rollout region) and we need to compute exactly what that mask position would be
        indices_to_keep = []
        mask_positions = []
        eos_positions_all = []
        num_full_prompts = []
        num_full_rollouts = []
        percent_prompt = []
        for start_step_index in indices:
            end_step_index = start_step_index + sample_sequence_length
            sampled_steps = steps[start_step_index:end_step_index]
            current_within_segment_idx = sampled_steps[:, 0]
            task_indices = sampled_steps[:, 2]
            task_lengths = self.replay_buffer.task_lengths[task_indices]

            eos_positions = list(np.where(current_within_segment_idx == (task_lengths - 1))[0]) # eos positions are where the within segment index equals the task length - 1
            eos_positions_copy = eos_positions.copy()

            if current_within_segment_idx[0] != 0: # if we are not starting at a full sequence, then we ignore the first eos position since we require at least one full demonstration in the prompt
                eos_positions = eos_positions[1:]

            # we need at least one full rollout in the non-prompt region. Remove any eos positions that would violate this condition
            while len(eos_positions) > 0:
                eos_position = eos_positions[-1] # pick the last eos position
                start_of_rollout_section_index = eos_position + 1 # the first index of the rollout section is immediately after the EOS of the prompt since the eos_position is the last index of the prompt (inclusive)
                if start_of_rollout_section_index == sample_sequence_length:
                    eos_positions.remove(eos_position)
                    continue

                length_of_first_rollout = task_lengths[start_of_rollout_section_index] # length of the full rollout; we need to keep at least this many steps in the sample
                if eos_position + length_of_first_rollout + 1 > self.prompt_sequence_length:
                    eos_positions.remove(eos_position)
                else:
                    break
            
            if len(eos_positions) > 0:
                # TODO: right now we just choose one such mask position from the set of valid mask positions. But in theory we could sample from all of them (so for each position in steps there would be different dataset samples corresponding to the same set of demonstrations, but different amounts rollouts set to be the prompt)

                eos_positions = eos_positions[:self.max_prompt_full_demos] # allow at most self.max_prompt_full_demos full demonstrations to be used as the prompt

                mask_position = rng.choice(eos_positions) # this choses at least 1 full sequence (assuming that at least one whole sequence can fit within the prompt length)

                indices_to_keep.append(start_step_index)
                mask_positions.append(mask_position)
                eos_positions_all.append(eos_positions_copy)

                # for statistics
                num_full_prompts.append((current_within_segment_idx[:mask_position + 1] == 0).sum()) # the number of start segment indices that are 0 are the number of full prompts that are included
                num_full_rollouts.append(len(eos_positions_copy[eos_positions_copy.index(mask_position)+1:])) # the number of EOS positions that are after the mask position are the number of full rollouts that are included
                percent_prompt.append(mask_position / self.prompt_sequence_length) # the percent of the prompt that is included in the prompt region

        average_task_length = np.mean(self.replay_buffer.task_lengths)
        print("\n--- start sampler statistics ---")
        print(f'Retained {len(indices_to_keep)} out of {len(indices)} indices for prompting ({100*len(indices_to_keep)/len(indices)}% kept). If this is a small proportion, then perhaps the task demonstrations are on average too long compared to the prompt sequence length (since the sequence contains full prompt and full rollout, the prompt sequence length should be at least twice the size of the longest demonstration.\nAverage task length across all recorded demos and tasks: {average_task_length} steps. Sampled sequence length: {self.prompt_sequence_length} (the average task length is {100*average_task_length/self.prompt_sequence_length}% of the sequence length).\nAverage number of full prompts: {np.mean(num_full_prompts)}\nAverage number of full rollouts: {np.mean(num_full_rollouts)}\nAverage percent of sequence that is the prompt: {100*np.mean(percent_prompt)}%')
        for verb in num_tasks_by_verb:
            print(f'Number of demos for task \"{verb}\" is {num_tasks_by_verb[verb]} (after padding: {num_tasks_by_verb_after_padding[verb]})')
        print('--- end sampler statistics ---\n')

        self.steps = steps
        self.indices = indices_to_keep
        self.mask_positions = mask_positions
        self.eos_positions_all = eos_positions_all

    def setup_indices_with_pair_prompting(self, seed:int):
        # the approach we take for pair prompting is to sample a pair consisting of (prompt, receding horizon rollout section)
        # we use the non-prompt sampler to sample the rollout section and then load an associated prompt of the same task from the replay buffer

        if self.only_prompt:
            valid_indices = np.where(self.mask)[0]
            if self.replay_buffer.is_task_error_correction is not None:
                non_ec_mask = ~self.replay_buffer.is_task_error_correction[valid_indices].astype(bool)
                non_ec_indices = valid_indices[non_ec_mask]
                assert len(non_ec_indices) > 0, (
                    "No non-error-correction demonstrations found in the prompt-only dataset split."
                )
                valid_indices = non_ec_indices
            self.indices = np.arange(len(valid_indices))
            self.prompt_indices = valid_indices
            return
        
        # first sample the dataset as if we didn't use prompting
        self.setup_indices_without_prompting()

        # now find an associated prompting for each index
        rng = np.random.default_rng(seed=seed)
        prompt_indices = np.zeros(len(self.indices), dtype='int')
        num_indices_processed = 0
        unique_task_name_to_dataset_indices = self.get_unique_task_name_to_dataset_indices()
        for unique_task_name, dataset_indices in unique_task_name_to_dataset_indices.items():
            cur_dataset_task_indices = np.where(self.replay_buffer.task_names[:] == unique_task_name)[0] # these are the valid prompts we can sample from for this task
            cur_dataset_task_indices = cur_dataset_task_indices[self.mask[cur_dataset_task_indices]]
            # Prompts must be full (non-EC) demonstrations — error correction demos show only
            # partial recovery, not a complete task execution.
            if self.replay_buffer.is_task_error_correction is not None:
                non_ec_mask = ~self.replay_buffer.is_task_error_correction[cur_dataset_task_indices].astype(bool)
                non_ec_indices = cur_dataset_task_indices[non_ec_mask]
                assert len(non_ec_indices) > 0, (
                    f"Task '{unique_task_name}' has no non-error-correction demonstrations in this split. "
                    "At least one full demonstration is required to use as a prompt."
                )
                cur_dataset_task_indices = non_ec_indices
            cur_prompt_task_indices = rng.choice(cur_dataset_task_indices, size=len(dataset_indices), replace=True)
            prompt_indices[dataset_indices] = cur_prompt_task_indices
            num_indices_processed += len(dataset_indices)
        
        assert num_indices_processed == len(self.indices), 'every index in the dataset should have a corresponding prompt'
        self.prompt_indices = prompt_indices

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx:int):
        """
        Format depends on sampling mode (the main idea is that rollout data is always in the top layer and prompt data is under 'prompt' to ensure consistency of the format; also putting rollout data in the top layer ensures compatibility with baseline diffusion policy models):
        non-prompting:
        {
            'metadata': {
                'task_idx': int,
                'episode_idx': int,
                'goal_image_task_idx': int, # if using goal image
            },
            **obs_keys,
            'action', # for rollout
            'labels': dict
        }
        only goal image:
        {
            'metadata': {
                'goal_image_task_idx': int,
            },
            'goal_image': (1, C, H, W) # float32
        }
        sequence prompting:
        {
            'prompt': {
                'metadata': {
                    'task_indices': list[int],
                    'episode_indices': list[int],
                    'prompt_mask': np.ndarray,
                    'eos': np.ndarray
                },
                'obs': {**obs_keys},
                'action', # (T, chunk, num_pred_steps, action_dim)
                'labels': dict, [optional],
            },
            'action': from prompt but without chunk but with downsampling (T [of which the first T_prompt are prompt actions (see action_mask below for this split)], num_pred_steps, action_dim),
            'metadata': {
                'action_mask': int # location of first non-prompt action
            }
        }
        pair prompting:
        {
            'metadata': {
                'task_idx': int, # for rollout
                'episode_idx': int, # for rollout
            },
            'prompt': {
                'metadata': {
                    'task_indices': list[int] of length 1,
                    'episode_indices': list[int] of length 1,
                    'mask': np.ndarray, # this will get added when using `collate_prompts` dataloader collation. True indicates padding/invalid action and False indicates valid action
                },
                'obs': {**obs_keys},
                'action',
                'labels': dict [optional],
            },
            **obs_keys: for rollout,
            'action': for rollout,
            'labels': for rollout [optional]
        }
        """
        if self.use_prompting:
            if self.prompt_sample_mode == 'sequence':
                return self.sample_sequence_with_sequence_prompting(idx)
            elif self.prompt_sample_mode == 'pair':
                return self.sample_sequence_with_pair_prompting(idx)
            else:
                raise ValueError(f'prompt_sample_mode {self.prompt_sample_mode} not implemented')
        else:
            return self.sample_sequence_without_prompting(idx)

    def sample_sequence_without_prompting(self, idx:int):
        """see `sample_sequence` for format"""

        if self.only_goal_image:
            segment_idx = self.goal_image_segment_indices[idx]
            goal_image = self._get_goal_image(segment_idx)
            
            result = {
                'metadata': {},
                'goal_image': goal_image
            }

            if self.sample_type == 'task':
                result['metadata']['goal_image_task_idx'] = segment_idx
            else:
                assert self.sample_type == 'episode'
                result['metadata']['goal_image_episode_idx'] = segment_idx

            return result

        current_within_segment_idx, episode_idx, task_idx = self.indices[idx]
        result = {
            'metadata': {}
        }

        if self.sample_type == 'task':
            task_length = self.replay_buffer.task_lengths[task_idx]
            result['metadata']['task_idx'] = task_idx
            
            # data idx
            end_data_idx = self.replay_buffer.task_data_ends[task_idx]
            start_data_idx = end_data_idx - task_length
            current_data_idx = start_data_idx + current_within_segment_idx

            # task idx
            start_labels_idx = self.replay_buffer.task_labels_ends[task_idx] - task_length
            end_labels_idx = self.replay_buffer.task_labels_ends[task_idx]
            current_labels_idx = start_labels_idx + current_within_segment_idx
        else: # episode
            data_slice = self.replay_buffer.get_episode_slice(episode_idx)
            start_data_idx, end_data_idx = data_slice.start, data_slice.stop
            current_data_idx = start_data_idx + current_within_segment_idx

        result['metadata']['episode_idx'] = episode_idx

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.prepared_replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            is_key_upsampled = self.replay_buffer.is_key_upsampled(key)
            
            if key in self.rgb_keys:
                if is_key_upsampled:
                    # if key is upsampled, then we need to handle the horizon differently. In particular, we apply the horizon and downsampling at the frequency of the observed data.
                    # so if we have 10Hz ultrawide data (and 60Hz other data), then a horizon of 2 will find the two most recent ultrawide frames (note these frames will be different)
                    # this is different than just duplicating the 10Hz data to make it 60Hz and then getting the last 2 frames becuase in that case the two frames would likely be the same, while in our implementation we ensure those are two different frames since we operate on the sampling frequency of 10Hz (rather than 60Hz upsampled stream of the data)
                    upsample_current_data_idx = self.replay_buffer.map_upsample_index(key, current_data_idx)
                    upsample_start_data_idx = self.replay_buffer.map_upsample_index(key, start_data_idx)
                    num_valid = min(this_horizon, (upsample_current_data_idx - upsample_start_data_idx) // this_downsample_steps + 1)
                    upsample_slice_start = upsample_current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[upsample_slice_start: upsample_current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
                else:
                    num_valid = min(this_horizon, (current_data_idx - start_data_idx) // this_downsample_steps + 1)
                    slice_start = current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[slice_start: current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
            else:
                assert not is_key_upsampled, 'upsampling not yet supported for low dim data'
                idx_to_sample = np.array(
                    [current_data_idx - idx * this_downsample_steps for idx in range(this_horizon)])
                idx_to_sample = idx_to_sample[::-1] # flip to make smallest idx come first now
                idx_to_sample = [x for x in idx_to_sample if x >= start_data_idx]

                output = input_arr[idx_to_sample]

            # for all observation types pad with the first value
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
                
            result[key] = output

        # action
        input_arr = self.prepared_replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_data_idx, current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_data_idx: slice_end: action_down_sample_steps]
        # solve padding
        is_ec_task = (
            self.sample_type == 'task'
            and self.replay_buffer.is_task_error_correction is not None
            and self.replay_buffer.is_task_error_correction[task_idx]
        )
        use_action_padding_for_this_segment = self.action_padding_error_correction if is_ec_task else self.action_padding
        if not use_action_padding_for_this_segment:
            assert output.shape[0] == action_horizon, 'a complete action horizon should have been available. This means that the setup indices for some reason allowed this index to be included even though we are not able to sample a full action horizon'
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0) # TODO: ideally each dataset should have it's own padding logic that is dependent on the action format as well as the action representation (delta vs absolute vs relative), but in practice just repeating the last action is probably fine
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        # labels
        if self.sample_type == 'task' and self.labels_keys:
            result['labels'] = {}
            for key in self.labels_keys:
                result['labels'][key] = self.prepared_replay_buffer[key][current_labels_idx]

        # goal image
        if self.use_goal_image:
            assert self.sample_type == 'task', 'goal image during training is only supported for task sampling currently'
            task_name = self.replay_buffer.task_names[task_idx]

            # select a goal image at random from the replay buffer within the training mask
            matching_task_locations = self.replay_buffer.task_names[:] == task_name
            if self.replay_buffer.meta.get('goal_img_frame_idx') is not None:
                goal_img_frame_idx = self.replay_buffer.meta.get('goal_img_frame_idx')[:]  # load as numpy array
                train_task_locations = (matching_task_locations & self.mask) & (goal_img_frame_idx != -1)
            else:
                train_task_locations = matching_task_locations & self.mask
            
            goal_image_task_idx = np.random.choice(np.where(train_task_locations)[0])
            goal_image = self._get_goal_image(goal_image_task_idx)
            result['goal_image'] = goal_image
            result['metadata']['goal_image_task_idx'] = goal_image_task_idx

        return result
    
    def sample_sequence_with_sequence_prompting(self, idx:int):
        """
        see `sample_sequence` for format

        TODO: could improve performance by using slicing instead of indexing with a list of ascending ordered indices since we index the arrays in order not randomly
        """
        if self.sample_type != 'task':
            raise NotImplementedError('prompting currently only supported for task sampling')
        
        result = {
            'metadata': {},
            'prompt': {
                'metadata': {},
            }
        }
        prompt_result = result['prompt']

        sample_sequence_length = self.prompt_sequence_length + self.num_pred_steps - 1

        """construct the indices into the replay buffer"""
        # extract a chunk of length `self.prompt_sequence_length` from the `steps` array
        start_step_index = self.indices[idx]
        mask_position = self.mask_positions[idx] # index of the EOS of the prompt region
        eos_positions = self.eos_positions_all[idx] # EOS for each demonstration
        assert mask_position in eos_positions, 'the mask position should be one of the eos positions since the mask ends at an eos position'
        end_step_index = start_step_index + sample_sequence_length
        sampled_steps = self.steps[start_step_index:end_step_index]

        # find the corresponding indices for these steps into the replay buffer
        current_within_segment_indices = sampled_steps[:, 0]
        sampled_task_indices = sampled_steps[:, 2]
        task_lengths = self.replay_buffer.task_lengths[sampled_task_indices]

        # data idx
        end_data_indices = self.replay_buffer.task_data_ends[sampled_task_indices]
        start_data_indices = end_data_indices - task_lengths
        sampled_data_indices = start_data_indices + current_within_segment_indices
        assert len(sampled_data_indices) == len(sampled_steps)

        # task idx
        end_labels_indices = self.replay_buffer.task_labels_ends[sampled_task_indices]
        start_labels_indices = end_labels_indices - task_lengths
        sampled_labels_indices = start_labels_indices + current_within_segment_indices
        assert len(sampled_labels_indices) == len(sampled_steps)

        # get the unique task indices in order that they appear in the sequence
        _, idx = np.unique(sampled_task_indices, return_index=True)
        result['prompt']['metadata']['task_indices'] = sampled_task_indices[np.sort(idx)]

        # get the episode indices associated with the task indices
        result['prompt']['metadata']['episode_indices'] = self.task_to_episode_idxs[result['prompt']['metadata']['task_indices']]
      
        eos = np.zeros((len(sampled_steps)), dtype='bool')
        eos[eos_positions] = 1

        """pull the requested data from the replay buffer"""
        prompt_result.update(self.sample_prompting_data(sampled_data_indices, sampled_labels_indices, eos, multi_step=True, trim_to_sequence_length=True))

        # validate the data
        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys
        for key in obs_keys:
            assert prompt_result['obs'][key].shape[0] == self.prompt_sequence_length
        assert prompt_result['action'].shape[0] == self.prompt_sequence_length

        # prompt mask
        prompt_mask = np.zeros((self.prompt_sequence_length,), dtype='bool')
        prompt_mask[mask_position+1:] = 1 # if 1 then not prompt
        prompt_result['metadata']['prompt_mask'] = prompt_mask

        # eos
        prompt_result['metadata']['eos'] = eos[:self.prompt_sequence_length] # eos previously included additional steps with the num_pred_steps so we need to crop it

        # if only_prompt is true, then we only return the prompt part of the data
        if self.only_prompt:
            for key in obs_keys:
                prompt_result['obs'][key] = prompt_result['obs'][key][:mask_position+1]
            
            prompt_result['action'] = prompt_result['action'][:mask_position+1]

            for key in ['prompt_mask', 'eos']:
                prompt_result['metadata'][key] = prompt_result['metadata'][key][:mask_position+1]

            last_sequence_idx = eos_positions.index(mask_position) # index into the task related lists
            for key in ['task_indices', 'episode_indices']:
                prompt_result['metadata'][key] = prompt_result['metadata'][key][:last_sequence_idx+1]

        # chunk the action
        prompt_result = add_batch_dim(prompt_result)
        prompt_result = self.prompt_action_chunker.chunk_prompt(prompt_result)
        prompt_result = remove_batch_dim(prompt_result)
        result['prompt'] = prompt_result
        result['action'] = result['prompt']['action'][:, 0] # (T, chunk, num_pred_steps, action_dim) -> (T, num_pred_steps, action_dim)

        non_prompt_locations = np.where(prompt_result['metadata']['prompt_mask'])[0]
        if len(non_prompt_locations) > 0:
            result['metadata']['action_mask'] = non_prompt_locations[0] # the first non-prompt location
        else:
            result['metadata']['action_mask'] = len(prompt_result['metadata']['prompt_mask']) # case of whole sequence is prompt, so the first non-prompt location is at the end

        return result

    def sample_sequence_with_pair_prompting(self, idx:int):
        """load the the pair consisting of (prompt, receding horizon rollout section)
        see `sample_sequence` for format
        """
        if self.sample_type != 'task':
            raise NotImplementedError('prompting currently only supported for task sampling')

        """load rollout section"""
        if self.only_prompt:
            result = {}
        else:
            result = self.sample_sequence_without_prompting(idx)

        if self.ignore_prompt:
            assert not self.only_prompt, 'if only_prompt is true, then we cannot ignore the prompt'
            return result

        """load the prompt"""
        result['prompt'] = {}
        prompt_result = result['prompt']
        prompt_task_idx = self.prompt_indices[idx]

        task_length = self.replay_buffer.task_lengths[prompt_task_idx]
        end_data_index = self.replay_buffer.task_data_ends[prompt_task_idx]
        start_data_index = end_data_index - task_length
        prompt_data_indices = slice(start_data_index, end_data_index)

        end_labels_index = self.replay_buffer.task_labels_ends[prompt_task_idx]
        start_labels_index = end_labels_index - task_length
        prompt_labels_indices = slice(start_labels_index, end_labels_index)

        # load the prompt data — obs pre-downsampled by chunk_n so chunk_prompt only needs to reshape actions
        prompt_result.update(self.sample_prompting_data(prompt_data_indices, prompt_labels_indices, None, multi_step=False, trim_to_sequence_length=False, downsample_obs_by_chunk=True))
        prompt_result['metadata'] = {}
        prompt_result['metadata']['task_indices'] = np.array([prompt_task_idx])
        prompt_result['metadata']['episode_indices'] = np.array([self.task_to_episode_idxs[prompt_task_idx]])

        prompt_result = add_batch_dim(prompt_result)
        prompt_result = self.prompt_action_chunker.chunk_prompt(prompt_result, obs_predownsampled=True)
        prompt_result = remove_batch_dim(prompt_result)
        result['prompt'] = prompt_result

        return result

    def sample_prompting_data(self, sampled_data_indices: Union[slice, np.ndarray], sampled_labels_indices: Union[slice, np.ndarray], eos: Optional[np.ndarray]=None, multi_step: bool=False, trim_to_sequence_length: bool=False, downsample_obs_by_chunk: bool=False):
        """
        Samples data from the replay buffer for the given indices for sampling prompts.
        The sampled data can be a single prompt or a sequence of prompts.
        If you want to do multi-step predictions, then you need to pass in the eos array, otherwise the eos array is not needed.
        format:
        {
            'obs': {**obs_keys},
            'action',
            'labels': dict
        }

        downsample_obs_by_chunk: if True, obs/proprio are loaded at 1/prompt_chunk_n_actions the normal rate so that
        chunk_prompt (with obs_predownsampled=True) skips the obs stride and only reshapes actions. The purpose of this is to dramatically improve IO efficiency: if we are going to chunk the prompt anyways no point in loading a bunch of observations that we will immediately discard.
        """
        result = {
            'obs': {}
        }

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        down_sample_steps = self.key_down_sample_steps['action'] # we downsampled all data by the action downsample steps
        if isinstance(sampled_data_indices, np.ndarray):
            # less efficient pathway but more general if you are not sampling contiguous segments in the replay buffer
            sampled_data_indices = sampled_data_indices[::down_sample_steps]
            sampled_labels_indices = sampled_labels_indices[::down_sample_steps]
        else:
            # more efficient pathway if you are sampling contiguous segments in the replay buffer
            sampled_data_indices = slice(sampled_data_indices.start, sampled_data_indices.stop, down_sample_steps)
            sampled_labels_indices = slice(sampled_labels_indices.start, sampled_labels_indices.stop, down_sample_steps)

        if downsample_obs_by_chunk:
            chunk_n = self.shape_meta['prompt_chunk_n_actions']
            if isinstance(sampled_data_indices, np.ndarray):
                obs_sampled_data_indices = sampled_data_indices[::chunk_n]
            else:
                obs_sampled_data_indices = slice(sampled_data_indices.start, sampled_data_indices.stop, sampled_data_indices.step * chunk_n)
        else:
            obs_sampled_data_indices = sampled_data_indices

        # observation
        for key in obs_keys:
            input_arr = self.prepared_replay_buffer[key]
            is_key_upsampled = self.replay_buffer.is_key_upsampled(key)

            if is_key_upsampled:
                adjusted_sampled_data_indices = self.replay_buffer.map_upsample_index(key, obs_sampled_data_indices) # convert indices from main data rate to lower data rate at which the data is actually stored
                output = input_arr[adjusted_sampled_data_indices]
            else:
                output = input_arr[obs_sampled_data_indices]

            # for prompting models the proprioceptive observations are converted in a prediction horizon since the prompting also outputs a horizon of proprioception predictions. For non proprioceptive observations like images, there is no prediction horizon so we just trim off the extra data
            prompt_type = self.shape_meta['obs'][key].get('sampler_prompt_type', self.shape_meta['obs'][key]['prompt_type'])
            if prompt_type == 'proprioception':
                if multi_step:
                    output = self._convert_multi_step(output, eos)
            elif prompt_type == 'observation':
                pass
            else:
                assert prompt_type == 'ignore'
                continue

            if trim_to_sequence_length:
                # we do this because when we sample a sequence of demonstrations together for sequence prompting the final length will likely not exactly match the prompt sequence length and instead will go over, so we need to trim it to the prompt sequence length
                assert output.shape[0] >= self.prompt_sequence_length, 'the output should be at least as long as the prompt sequence length'
                output = output[:self.prompt_sequence_length]
                
            result['obs'][key] = output

        # action
        input_arr = self.prepared_replay_buffer['action']
        output = input_arr[sampled_data_indices]
        if multi_step:
            output = self._convert_multi_step(output, eos)
        if trim_to_sequence_length:
            output = output[:self.prompt_sequence_length]
        result['action'] = output

        # labels
        if len(self.labels_keys) > 0:
            result['labels'] = {}
        for key in self.labels_keys:    
            result['labels'][key] = self.prepared_replay_buffer[key][sampled_labels_indices]

        return result
    
    def _convert_multi_step(self, data: np.ndarray, eos: np.ndarray):
        """
        Convert a sequence which consists of multiple tasks stitched together into a sequence of multi-step predictions. We need EOS here because we don't want to have a horizon that pulls actions from a different task. We only want to pull action horizon if there is room within the current task.
        `data` is a numpy array of shape (seq_length + self.num_pred_steps - 1, data_dim)
        `eos` is a numpy array of shape (seq_length + self.num_pred_steps - 1, ) where True if at the last step of the task

        output is numpy array of shape (seq_length, self.num_pred_steps, data_dim)
        """
        sample_sequence_length = data.shape[0]
        assert eos.shape[0] == sample_sequence_length

        # iterate through each chunk of the same segment within the data
        chunk_ranges = [0] + list(np.where(eos)[0]+1) + [sample_sequence_length]
        data_chunked = []
        for i in range(1, len(chunk_ranges)):
            start = chunk_ranges[i-1]
            end = chunk_ranges[i]
            data_segment = data[start:end] # (chunk T, data_dim)
            multi_step_data = convert_multi_step(data_segment, self.num_pred_steps) # (chunk T, num_pred_steps, data_dim)
            data_chunked.append(multi_step_data)
        
        output = np.concatenate(data_chunked, axis=0) # (seq_length + self.num_pred_steps -1 , num_pred_steps, data_dim)
        return output
    
    def _get_goal_image(self, segment_idx: int) -> np.ndarray:
        """The goal image is the last image of the task in the replay buffer"""
        if self.sample_type == 'task':
            goal_image_frame_idx = self.replay_buffer.task_data_ends[segment_idx] - 1 # end of segment
            if self.replay_buffer.meta.get('goal_img_frame_idx') is not None:
                meta_frame_idx = int(self.replay_buffer.meta['goal_img_frame_idx'][segment_idx])
                # -1 means "invalid / use last frame of this segment"; otherwise use the stored frame index
                assert meta_frame_idx >= 0, 'goal image frame index is invalid, update your dataset to either set a valid goal image frame index or remove this demonstration from your dataset. Segment index: {segment_idx}'
                goal_image_frame_idx = meta_frame_idx
        else:
            assert self.sample_type == 'episode'
            assert self.replay_buffer.meta.get('goal_img_frame_idx') is None, 'goal image frame index is not yet supported for episode sampling'
            goal_image_frame_idx = self.replay_buffer.episode_ends[segment_idx] - 1 # end of episode
        goal_image = self.replay_buffer.data[self.goal_image_key][goal_image_frame_idx]
        goal_image = np.expand_dims(goal_image, axis=0) # insert horizon dimension
        return goal_image

    def shuffle_data_ordering(self, seed:int):
        self.setup_indices(seed)

    def requires_epoch_shuffle(self) -> bool:
        criteria1 = self.use_prompting and self.prompt_sample_mode == 'sequence' # sequence prompting requires shuffling the ordering of trajectories within the sequences
        criteria2 = self.use_prompting and self.prompt_sample_mode == 'pair' and not self.ignore_prompt # the shuffle is needed for pair prompting because we pair prompts with receding sections, but if we are ignoring the prompt then we don't need to shuffle
        return criteria1 or criteria2

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply

    def get_unique_task_name_to_dataset_indices(self, exclude_error_correction: bool = False) -> Dict[str, list[int]]:
        """Returns a Dict[str, list[int]] mapping from unique task names to the dataset indices that correspond to that task name. This is useful if you want to know which specific indices into this sampler correspond to a specific task name. This is only supported for task sampling.

        Args:
            exclude_error_correction: if True, only return dataset indices whose underlying task is not an error correction demonstration.
        """
        assert self.sample_type == 'task', 'unique task name to dataset indices is only supported for task sampling'
        task_names = self.replay_buffer.task_names
        unique_task_names = sorted(list(set(task_names)))
        task_idx_to_unique_task_idx = []
        for task_name in task_names:
            task_idx_to_unique_task_idx.append(unique_task_names.index(task_name))
        task_idx_to_unique_task_idx = np.array(task_idx_to_unique_task_idx)
        
        if self.use_prompting and self.prompt_sample_mode == 'sequence':
            # for prompting we have two layers of indexing. In particular we have self.indices which then maps into self.steps which is the array that contains the actual data indices and other information like the task index that we care about. In particular the self.indices refers to starting positions in self.steps. And since the task is the same across an entire prompt we can just can the task index of the starting entry.
            dataset_step_task_indices = self.steps[:, 2] # (current_within_segment_idx, episode_idx, task_idx)
            dataset_task_indices = dataset_step_task_indices[self.indices]
        elif self.use_prompting and self.prompt_sample_mode == 'pair' and self.only_prompt:
            dataset_task_indices = self.prompt_indices
        else:
            if self.only_goal_image:
                dataset_task_indices = self.goal_image_segment_indices
            else:
                # for non sequence prompting models the indices are directly the dataset indices
                dataset_task_indices = self.indices[:, 2] # (current_within_segment_idx, episode_idx, task_idx)

        if len(dataset_task_indices) == 0:
            return {}
        
        # dataset_task_indices is the task index for each entry in the dataset, we need to map it to the unique task index
        unique_task_indices = task_idx_to_unique_task_idx[dataset_task_indices]
        assert len(unique_task_indices) == len(self), 'every entry in the dataset should have a corresponding unique task index'

        is_ec = self.replay_buffer.is_task_error_correction
        ec_mask = is_ec[dataset_task_indices].astype(bool) if (exclude_error_correction and is_ec is not None) else None

        result = {}
        for unique_task_idx, unique_task_name in enumerate(unique_task_names):
            task_mask = unique_task_indices == unique_task_idx
            if not task_mask.any():
                continue
            filter_mask = task_mask & ~ec_mask if ec_mask is not None else task_mask
            indices = np.where(filter_mask)[0].tolist()
            if len(indices) == 0:
                continue
            result[unique_task_name] = indices

        return result

    def set_ignore_prompt(self, ignore_prompt: bool):
        self.ignore_prompt = ignore_prompt

    def get_ignore_prompt(self) -> bool:
        return self.ignore_prompt
