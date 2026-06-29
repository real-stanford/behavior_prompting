from typing import Dict
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from behavior_prompting.train_network.model.common.normalizer import Normalizer

class PromptActionChunker:
    """
    Chunks the action into chunks of size prompt_chunk_n_actions and removes observation and proprioception data according to this chunk size.
    Resulting format (for chunk_n_actions = 2): O,P,[A,A],O,P,[A,A],...
    To be used on numpy arrays.
    """
    def __init__(self,
                 shape_meta: dict,
                 ):
        super().__init__()
        self.shape_meta = shape_meta

        # figure out which observation keys corresponds to proprioception and observation parts of the prompt
        prompt_type_to_prompt_keys: Dict[str, list] = {
            'observation': [],
            'proprioception': [],
            'ignore': [],
        }
        for obs_key in shape_meta['obs']:
            prompt_type = self.shape_meta['obs'][obs_key].get('sampler_prompt_type', self.shape_meta['obs'][obs_key]['prompt_type'])
            assert prompt_type in ['proprioception', 'observation', 'ignore']
            if 'sampler_prompt_type' in shape_meta['obs'][obs_key]:
                assert shape_meta['obs'][obs_key]['sampler_prompt_type'] != 'ignore', \
                    f"obs key '{obs_key}' has sampler_prompt_type='ignore' — just set prompt_type to 'ignore' instead"
            if not shape_meta['obs'][obs_key].get('ignore_by_policy', False) or 'sampler_prompt_type' in shape_meta['obs'][obs_key]:
                prompt_type_to_prompt_keys[prompt_type].append(obs_key)
        
        self.prompt_type_to_prompt_keys = prompt_type_to_prompt_keys
        self.prompt_chunk_n_actions = shape_meta['prompt_chunk_n_actions']
        self.action_horizon = shape_meta['action']['horizon']
        self.pad_end_prompt_actions = shape_meta.get('pad_end_prompt_actions', 'no')
        assert self.pad_end_prompt_actions in ['no', 'zeros', 'repeat'], \
            f"Invalid pad_end_prompt_actions: {self.pad_end_prompt_actions}"

    def chunk_prompt(self, prompt: dict[str, np.ndarray], obs_predownsampled: bool = False) -> dict[str, np.ndarray]:
        """
        Chunk the prompt into chunks of size prompt_chunk_n_actions. Observations and proprioception data are downsampled, while action data is chunked and retained at the full resolution. The prompt is truncated to the nearest multiple of the chunk size. If the prompt is less than one chunk size (prompt_chunk_n_actions > sequence_length), we pad the prompt by filling the extra actions with zeros to ensure we always have at least one full chunk.

        When 'action' is provided in the prompt it's assumed that that action contains the action prediction horizon and that the shape of the action is (B, T, num_pred_steps [optional dimension], action_dim).
        For proprioception the prediction horizon is an optional dimension and the shape is (B, T, num_pred_steps [optional dimension], proprio_dim).
        For observation the shape is (B, T, num_cameras, 3, H, W) and has no prediction horizon.

        obs_predownsampled: if True, obs/proprio have already been downsampled by prompt_chunk_n_actions at load time,
        so no striding is applied here. T is derived from action shape. prompt_mask is not supported in this mode.
        """
        result = {
            'obs': {},
            'metadata': {}
        }

        # downsample all data and chunk actions
        if obs_predownsampled:
            assert 'prompt_mask' not in prompt.get('metadata', {}), \
                'prompt_mask handling not implemented for obs_predownsampled=True'
            assert 'eos' not in prompt.get('metadata', {}), \
                'eos handling not implemented for obs_predownsampled=True'
            B, T = prompt['action'].shape[:2]
        else:
            B, T = prompt['obs'][self.prompt_type_to_prompt_keys['observation'][0]].shape[:2]
        downsampled_length = T // self.prompt_chunk_n_actions
        less_than_one_chunk = downsampled_length == 0 # in the case that the chunk size is larger than the sequence length, we must ensure that there is at least one chunk
        if less_than_one_chunk:
            downsampled_length = 1

        has_partial_chunk = (T % self.prompt_chunk_n_actions) != 0
        should_include_partial_chunk = (
            self.pad_end_prompt_actions in ['zeros', 'repeat']
            and has_partial_chunk
            and not less_than_one_chunk
        )
        if should_include_partial_chunk:
            downsampled_length += 1

        trimmed_T = downsampled_length * self.prompt_chunk_n_actions # this is the length you want to keep from the prompt. Potentially it's actually longer than the prompt if you requested to pad the prompt actions
        padded_T = trimmed_T if (
            T % self.prompt_chunk_n_actions == 0
            or less_than_one_chunk
            or should_include_partial_chunk
        ) else trimmed_T + self.prompt_chunk_n_actions

        if 'metadata' in prompt and 'prompt_mask' in prompt['metadata']:
            result['metadata']['prompt_mask'] = prompt['metadata']['prompt_mask'][:, :trimmed_T:self.prompt_chunk_n_actions]

        for key in self.prompt_type_to_prompt_keys['observation'] + self.prompt_type_to_prompt_keys['proprioception']:
            if key in prompt['obs']:
                if obs_predownsampled:
                    result['obs'][key] = prompt['obs'][key][:, :downsampled_length]
                else:
                    result['obs'][key] = prompt['obs'][key][:, :trimmed_T:self.prompt_chunk_n_actions]
                # obs: (B, downsampled_length, num_cameras, 3, H, W)
                # proprio: (B, downsampled_length, num_pred_steps [optional dimension], proprio_dim)
                
        if 'action' in prompt:
            ending_dim = prompt['action'].shape[2:] # either contains (num_pred_steps, action_dim) or (action_dim, )
            trimmed_action = prompt['action'][:, :trimmed_T] # (B, <=trimmed_T, num_pred_steps [optional dimension], action_dim)
            if trimmed_action.shape[1] < trimmed_T:
                pad_len = trimmed_T - trimmed_action.shape[1]
                if self.pad_end_prompt_actions == 'repeat' and trimmed_action.shape[1] > 0 and not less_than_one_chunk:
                    pad_actions = np.repeat(trimmed_action[:, -1:, ...], pad_len, axis=1)
                else:
                    # Default and less-than-one-chunk behavior: pad missing actions with zeros.
                    pad_actions = np.zeros((B, pad_len, *ending_dim), dtype=trimmed_action.dtype)
                trimmed_action = np.concatenate([trimmed_action, pad_actions], axis=1) # (B, trimmed_T, num_pred_steps [optional dimension], action_dim)
            result['action'] = trimmed_action.reshape(B, downsampled_length, self.prompt_chunk_n_actions, *ending_dim)

        if 'metadata' in prompt and 'eos' in prompt['metadata']:
            # eos originally looks like [0, 0, 0, 1, 0, 0, 0, 1, ...] and if we downsample by 2 naively we get [0, 0, 0, 0, ...], whereas we want [0, 1, 0, 1, ...]. So we add together the eos values for each chunk (sum([0,0], sum[0,1], sum[0,0], sum([0,1])) to get the correct result.
            padded_eos = np.concatenate([prompt['metadata']['eos'], np.zeros((B, padded_T - T))], axis=1)
            result['metadata']['eos'] = (padded_eos.reshape(B, -1, self.prompt_chunk_n_actions).sum(axis=-1) > 0)
            if result['metadata']['eos'].shape[1] > downsampled_length:
                assert result['metadata']['eos'].shape[1] == downsampled_length + 1, 'we expect only 1 length longer due to padding'
                result['metadata']['eos'][:, -2] |= result['metadata']['eos'][:, -1] # handles case were eos gets cut off at the end of the sequence
                result['metadata']['eos'] = result['metadata']['eos'][:, :downsampled_length]
            assert result['metadata']['eos'].sum() == prompt['metadata']['eos'].sum(), 'number of eos in the downsampled batch is not the same as the original batch'

            # zero out actions within a chunk that are after an eos
            for batch_idx in range(B):
                eos_locations = np.where(prompt['metadata']['eos'][batch_idx])[0]
                for eos_location in eos_locations:
                    if eos_location >= trimmed_T:
                        # case where eos is after end of last chunk; in this case the last chunk is already valid since it hadn't hit the eos yet
                        continue
                    chunk_idx = eos_location // self.prompt_chunk_n_actions
                    within_chunk_idx = eos_location % self.prompt_chunk_n_actions
                    result['action'][batch_idx, chunk_idx, within_chunk_idx+1:] = 0 # TODO: assumes that zero action means no action (aka delta end effector control)

        # add any missing metadata
        if 'metadata' in prompt:
            for key in prompt['metadata']:
                if key not in result['metadata']:
                    result['metadata'][key] = prompt['metadata'][key]

        return result

def collate_prompts(batch, pad_value=0):
    """
    Custom torch dataloader collate function that does default collation if no prompts are present or if prompts have the same lengths (sequence prompting). Does special collation to pad prompts in the case of pair prompting where prompts have different lengths.
    If padding prompts, also returns a prompt_mask for the prompt fields to indicate real vs. padded values located at batch['obs']['prompt']['metadata']['mask'].
    """
    # if no prompts are present, just return the batch
    if 'prompt' not in batch[0]['obs']:
        return default_collate(batch)
    
    # if prompts have the same lengths, just return the batch; this can either occur in sequence sampling (where all the prompts have the same length) or in pair sampling (where all the prompts have varying lengths, but a given batch might have prompts of the same length)
    if all(len(sample['obs']['prompt']['action']) == len(batch[0]['obs']['prompt']['action']) for sample in batch):
        # remove the task_indices and episode_indices from the batch before collation if they have length greater than 1
        should_remove_task_indices = False
        for sample in batch:
            if 'task_indices' in sample['obs']['prompt']['metadata']:
                if sample['obs']['prompt']['metadata']['task_indices'].shape[0] != 1:
                    should_remove_task_indices = True
                    break

        if should_remove_task_indices:
            for sample in batch:
                if 'task_indices' in sample['obs']['prompt']['metadata']:
                    del sample['obs']['prompt']['metadata']['task_indices']
                if 'episode_indices' in sample['obs']['prompt']['metadata']:
                    del sample['obs']['prompt']['metadata']['episode_indices']

        collated = default_collate(batch)

        collated['obs']['prompt']['metadata']['mask'] = torch.zeros((len(batch), collated['obs']['prompt']['action'].shape[1]), dtype=torch.bool, device=collated['obs']['prompt']['action'].device)

        return collated
    
    # at this point we know that prompts have different lengths (pair prompting)
    
    # Extract all prompts from the batch
    prompts = [sample['obs']['prompt'] for sample in batch]

    def pad_dict(list_of_dicts, key):
        # Gather all tensors for this key from each dict in the list
        tensors = [d[key] for d in list_of_dicts]
        lengths = [t.shape[0] for t in tensors]
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
        return padded, lengths

    # Collate/pad prompt['obs']
    obs_keys = prompts[0]['obs'].keys()
    collated_prompt_obs = {}
    lengths = None
    for key in obs_keys:
        padded, key_lengths = pad_dict([p['obs'] for p in prompts], key)
        collated_prompt_obs[key] = padded
        if lengths is None:
            lengths = key_lengths  # Use the first key's lengths as reference

    # Collate/pad prompt['action']
    collated_prompt_action, _ = pad_dict(prompts, 'action')

    # Collate/pad prompt['labels'] if present
    collated_prompt_labels = None
    if 'labels' in prompts[0]:
        label_keys = prompts[0]['labels'].keys()
        collated_prompt_labels = {}
        for key in label_keys:
            padded, _ = pad_dict([p['labels'] for p in prompts], key)
            collated_prompt_labels[key] = padded

    # Create a single prompt_mask
    max_len = collated_prompt_action.shape[1]
    prompt_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool, device=collated_prompt_action.device)
    for i, l in enumerate(lengths):
        prompt_mask[i, l:] = True # True means ignore

    # Collate prompt['metadata'] using default_collate
    collated_prompt_metadata = default_collate([p['metadata'] for p in prompts])
    collated_prompt_metadata['mask'] = prompt_mask

    # Collate the rest of the batch using default_collate
    # Remove 'prompt' from each sample's obs before collating
    for sample in batch:
        sample['obs'] = {k: v for k, v in sample['obs'].items() if k != 'prompt'}
    collated_batch = default_collate(batch)

    # Reinsert the collated prompt and prompt_mask
    collated_prompt = {
        'obs': collated_prompt_obs,
        'action': collated_prompt_action,
        'metadata': collated_prompt_metadata
    }
    if collated_prompt_labels is not None:
        collated_prompt['labels'] = collated_prompt_labels
    collated_batch['obs']['prompt'] = collated_prompt

    return collated_batch

def normalize_obs_with_optional_prompt(obs_dict: dict, normalizer: Normalizer):
    """
    Normalizes an observation. The observation may optionally include a 'prompt' key which will be normalized along with the rest of the observation. This method handles the fact that observations can show up both within and outside of the prompt, but that in both cases the same normalization should be applied. Also if a prompt is provided, the actions will also be normalized in addition to the observations.
    """
    obs_dict = obs_dict.copy()
    prompt_present = 'prompt' in obs_dict
    if prompt_present:
        prompt = obs_dict.pop('prompt')
    
    # Special case: task_language will not be in normalizer since we are using finetuned language encoders (at this point the language observation contains token IDs, not embeddings)
    task_language_value = None
    if 'task_language' in obs_dict and 'task_language' not in normalizer.params_dict:
        task_language_value = obs_dict.pop('task_language')
    
    nobs = normalizer.normalize(obs_dict) # works if receding obs is present or not
    
    # Add task_language back without normalization if it was skipped
    if task_language_value is not None:
        nobs['task_language'] = task_language_value

    if prompt_present:
        prompt_normalizer = normalizer.get_prompt_normalizer()
        nobs['prompt'] = {
            'obs': prompt_normalizer.normalize(prompt['obs']),
            'action': prompt_normalizer['action'].normalize(prompt['action']),
            'metadata': prompt['metadata']
        }

    return nobs
