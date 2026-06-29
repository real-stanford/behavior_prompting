"""Implementation adapted from ICRT"""

import math
from typing import List, Optional, Tuple, Dict
import torch
import torchvision
import timm
import torch.nn as nn
import copy
import torch.nn.functional as F
from behavior_prompting.train_network.model.common.module_attr_mixin import ModuleAttrMixin
from behavior_prompting.train_network.common.augmentation import ImageAugmentation
from behavior_prompting.train_network.model.common.base_obs_encoder import BaseTokenizedObsEncoder
from behavior_prompting.train_network.model.common.transformer_decoder_with_attention import TransformerDecoderLayerWithAttn, TransformerDecoderWithAttn
from timm.layers.mlp import Mlp
from timm.layers.attention_pool import AttentionPoolLatent

from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder
from behavior_prompting.train_network.utils.model_util import get_optim_groups, init_weights
from behavior_prompting.train_network.utils.shape_meta_utils import get_obs_keys_from_shape_meta

class ICRTPromptObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 train_image_transforms: Optional[ImageAugmentation],
                 eval_image_transforms: Optional[ImageAugmentation],
                 vision_model_name: str):
        super().__init__()
        self.shape_meta = shape_meta

        # figure out which observation keys corresponds to proprioception and observation parts of the prompt
        prompt_type_to_prompt_keys: Dict[str, list] = {}
        rgb_keys = []
        for obs_key in shape_meta['obs']:
            prompt_type = shape_meta['obs'][obs_key]['prompt_type']
            assert prompt_type in ['proprioception', 'observation', 'ignore']
            if not shape_meta['obs'][obs_key].get('ignore_by_policy', False):
                if prompt_type not in prompt_type_to_prompt_keys:
                    prompt_type_to_prompt_keys[prompt_type] = []
                
                prompt_type_to_prompt_keys[prompt_type].append(obs_key)
                if shape_meta['obs'][obs_key]['type'] == 'rgb':
                    rgb_keys.append(obs_key)

        # sort the keys to have a consistent order when combined into the observation
        for key in prompt_type_to_prompt_keys:
            prompt_type_to_prompt_keys[key].sort()
        
        self.prompt_type_to_prompt_keys = prompt_type_to_prompt_keys

        # prepare image transforms
        self.key_train_transform_map = {}
        self.key_eval_transform_map = {}
        for key in rgb_keys:
            self.key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
            self.key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()

        # prepare image normalizer
        self.vision_model_name = vision_model_name
        pretrained_cfg = timm.get_pretrained_cfg(vision_model_name)
        model_data_config = timm.data.resolve_data_config(pretrained_cfg=vars(pretrained_cfg))
        self.image_model_normalization_transform = torchvision.transforms.Normalize(
            mean=model_data_config['mean'],
            std=model_data_config['std']
        )

        self.action_horizon = shape_meta['action']['horizon']

    def forward(self, prompt_dict):
        """
        Combines observation data into three categories: proprioception, observation, and action in preparation for input into a prompt based model. The proprioception and action data inputs has a multi step version of length num_pred_steps (add a horizon to the prediction), while the observation data consists of the camera images with no prediction horizon. The data is downsampled and chunked before this function is called, so the action data comes in chunks while the proprioception and observation data comes in downsampled versions.

        input:
            prompt_dict: dict - dictionary of torch tensors containing keys from `shape_meta` that have dimensions (B, T, *shape) where shape will have the first dimension of `num_pred_steps` for proprioception and action data.
                optionally includes `action` key which is a tensor of shape (B, T, num_action_features)
                in addition to data keys, there are two additional keys:
                - `eos` which is a tensor of shape (B, T) that indicates the end of the sequence for each batch (1 being end of sequence, 0 being not end of sequence)
                - `prompt_mask` which is a tensor of shape (B, T) that indicates the prompt mask for each batch (0 being prompt, 1 being not prompt)
        
        output:
            result: dict - combine observation data into three categories
                {
                    'proprio': np.ndarray, - (B, T_downsampled, num_pred_steps [optional dimension], num_proprio_features) for N cameras
                    'observation': np.ndarray, - (B, T_downsampled, num_cameras, 3, H, W) image observations
                    'action': np.ndarray [optional] - (B, T_downsampled, chunk_n_actions, num_pred_steps, num_action_features)
                }
        """
        prompt_dict = prompt_dict.copy()
        
        # to format into prompt we grouped data based on the prompting type (which is either proprioception, or observation) and then concatenate the data together

        # proprioception
        proprioception = []
        for key in self.prompt_type_to_prompt_keys['proprioception']:
            proprioception.append(prompt_dict['obs'][key]) # (B, T, num_pred_steps [optional dimension], cur_proprioception_dim) 
        proprioception = torch.cat(proprioception, dim=-1) # (B, T, num_pred_steps [optional dimension], proprioception_dim) 

        # observation
        observation = []
        for key in self.prompt_type_to_prompt_keys['observation']:
            img = prompt_dict['obs'][key] # (B, T, 3, H, W)
            orig_image_shape = img.shape
            B, T = img.shape[:2]
            img = img.view(B*T, *img.shape[2:])

            if self.training:
                img = self.key_train_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            img = self.image_model_normalization_transform(img) # apply image normalization
            img = img.view(orig_image_shape)
            observation.append(img.unsqueeze(2)) # (B, T, 1, 3, H, W)
        observation = torch.cat(observation, dim=2) # (B, T, num_cameras, 3, H, W)

        result = {
            'proprio': proprioception, # (B, T, num_pred_steps, proprioception_dim)
            'observation': observation, # (B, T, num_cameras, 3, H, W)
        }

        # prompt mask and weight
        if 'prompt_mask' in prompt_dict.get('metadata', []):
            result['prompt_mask'] = prompt_dict['metadata']['prompt_mask'] # (B, T)
            result['weight_mask'] = None
        
        # action
        if 'action' in prompt_dict:
            result['action'] = prompt_dict['action'] # (B, T, num_pred_steps, action_dim)

        return result
    
    @property
    def num_cameras(self):
        num_cameras = 0
        for obs_key in self.shape_meta['obs']:
            if self.shape_meta['obs'][obs_key]['type'] == 'rgb' and self.shape_meta['obs'][obs_key]['prompt_type'] == 'observation':
                num_cameras += 1
        return num_cameras
    
    @property
    def proprio_dim(self):
        proprio_dim = 0
        for obs_key in self.shape_meta['obs']:
            if self.shape_meta['obs'][obs_key]['prompt_type'] == 'proprioception':
                cur_shape = self.shape_meta['obs'][obs_key]['shape']
                assert len(cur_shape) == 1
                proprio_dim += cur_shape[0]
        return proprio_dim

class PairPromptTransformerTokenizer(ModuleAttrMixin):
    """
    PairPromptTransformerTokenizer embeds a prompt output from PromptObsEncoder into a single embedding sequence. The main idea is that each observation modality is treated as a separate token, and the actions are treated as a separate token. These tokens are then interleaved properly in the format: receding obs 1, ..., receding obs N, prompt obs 1, prompt action 1, prompt obs 2, prompt action 2, ..., prompt obs N, prompt action N.

    Returns a prompt embedding of shape (B, T, n_emb) where T = obs_horizon*receding_num_obs_modalities (for current state) + P*prompt_num_obs_modalities (for prompt states) + P (for prompt actions). P is the number of timesteps in the prompt.
    Obs and actions are interleaved in the prompt and are preceded by the current observation tokens.
    """
    def __init__(self,
                 shape_meta: dict,
                 obs_encoder: TransformerObsEncoder,
                 n_emb: int,
                 concat_prompt_and_receding: bool,
                 only_use_last_step_of_prompt: bool=False,
                 cut_first_n_steps_of_prompt: int=0,
                 ignore_prompt_obs: bool=False,
                 ignore_prompt_proprio: bool=False,
                 ignore_prompt_action: bool=False,
                 merge_prompt_tokens: str='none', # none, obs, obs_and_action
                 prompt_attention_pool: Optional[AttentionPoolLatent]=None,
                 use_pool_modality_pos_embed: bool=True,
                 separate_prompt_and_receding_obs_encoders: bool=False
                 ):
        super().__init__()
        self.shape_meta = shape_meta
        self.separate_prompt_and_receding_obs_encoders = separate_prompt_and_receding_obs_encoders
        if separate_prompt_and_receding_obs_encoders:
            self.prompt_obs_encoder = obs_encoder
            self.receding_obs_encoder = copy.deepcopy(obs_encoder)
        else:
            self.prompt_obs_encoder = obs_encoder
            self.receding_obs_encoder = obs_encoder

        self.n_emb = n_emb
        self.concat_prompt_and_receding = concat_prompt_and_receding
        self.only_use_last_step_of_prompt = only_use_last_step_of_prompt
        self.cut_first_n_steps_of_prompt = cut_first_n_steps_of_prompt
        self.ignore_prompt_obs = ignore_prompt_obs
        self.ignore_prompt_proprio = ignore_prompt_proprio
        self.ignore_prompt_action = ignore_prompt_action
        self.merge_prompt_tokens = merge_prompt_tokens
        self.prompt_attention_pool = None if merge_prompt_tokens == 'none' else prompt_attention_pool
        self.prompt_has_no_obs_or_proprio = ignore_prompt_obs and ignore_prompt_proprio

        # obs keys
        obs_keys = get_obs_keys_from_shape_meta(self.shape_meta, for_policy=True, skip_prompt_proprio=self.ignore_prompt_proprio, skip_prompt_observation=self.ignore_prompt_obs)
        self.current_obs_rgb_keys = obs_keys['current_obs_rgb']
        self.current_obs_low_dim_keys = obs_keys['current_obs_low_dim']
        self.current_obs_keys = obs_keys['current_obs_all']
        self.prompt_rgb_keys = obs_keys['prompt_rgb']
        self.prompt_low_dim_keys = obs_keys['prompt_low_dim']
        self.prompt_obs_keys = obs_keys['prompt_all']
        self.prompt_current_obs_rgb_keys = obs_keys['prompt_current_obs_rgb']
        self.prompt_current_obs_low_dim_keys = obs_keys['prompt_current_obs_low_dim']
        self.prompt_current_obs_keys = obs_keys['prompt_current_obs_all']
        self.prompt_current_obs_has_same_keys_as_current_obs = set(self.prompt_current_obs_keys) == set(self.current_obs_keys)

        if not self.prompt_current_obs_has_same_keys_as_current_obs and not self.concat_prompt_and_receding and not self.separate_prompt_and_receding_obs_encoders:
            receding_obs_names = self.receding_obs_encoder.get_obs_names(rgb_keys=self.current_obs_rgb_keys, low_dim_keys=self.current_obs_low_dim_keys)
            prompt_receding_obs_names = self.prompt_obs_encoder.get_obs_names(rgb_keys=self.prompt_current_obs_rgb_keys, low_dim_keys=self.prompt_current_obs_low_dim_keys)
            self.prompt_current_obs_subset_indices = [receding_obs_names.index(key) for key in prompt_receding_obs_names] # these specify how to go from an observation generated for the current obs and select out the modalities that should be present to form the prompt current obs
        
        # If we duplicate the obs encoder, freeze any prompt-encoder modalities that are never used.
        if self.separate_prompt_and_receding_obs_encoders:
            used_prompt_keys = set(self.prompt_obs_keys) | set(self.prompt_current_obs_keys)
            # All modules keyed by obs name live in key_model_map/key_projection_map
            all_prompt_keys = set(self.prompt_obs_encoder.key_model_map.keys()) | set(self.prompt_obs_encoder.key_projection_map.keys())
            unused_prompt_keys = all_prompt_keys - used_prompt_keys
            for key in unused_prompt_keys:
                if key in self.prompt_obs_encoder.key_model_map:
                    for p in self.prompt_obs_encoder.key_model_map[key].parameters():
                        p.requires_grad = False
                if key in self.prompt_obs_encoder.key_projection_map:
                    for p in self.prompt_obs_encoder.key_projection_map[key].parameters():
                        p.requires_grad = False
            # do the same for the receding obs encoder
            all_receding_keys = set(self.receding_obs_encoder.key_model_map.keys()) | set(self.receding_obs_encoder.key_projection_map.keys())
            unused_receding_keys = all_receding_keys - set(self.current_obs_keys)
            for key in unused_receding_keys:
                if key in self.receding_obs_encoder.key_model_map:
                    for p in self.receding_obs_encoder.key_model_map[key].parameters():
                        p.requires_grad = False
                if key in self.receding_obs_encoder.key_projection_map:
                    for p in self.receding_obs_encoder.key_projection_map[key].parameters():
                        p.requires_grad = False
        # some sanity checks
        assert self.merge_prompt_tokens in ['none', 'obs', 'obs_and_action'], f'invalid merge_prompt_tokens: {self.merge_prompt_tokens}'

        if self.merge_prompt_tokens == 'obs' or self.merge_prompt_tokens == 'obs_and_action':
            assert self.prompt_attention_pool is not None, 'prompt_attention_pool must be provided if merge_prompt_tokens is obs or obs_and_action'

        assert not (self.ignore_prompt_obs and self.ignore_prompt_proprio and self.ignore_prompt_action), 'at least one of prompt obs, proprio, or action must not be ignored'

        if self.only_use_last_step_of_prompt and self.concat_prompt_and_receding:
            raise NotImplementedError('if keep_only_last_prompt_timestamp is True, then concat_prompt_and_receding must be False. It is implemented, but not tested.')

        sample_obs_key = self.current_obs_keys[0]
        self.receding_obs_horizon = self.shape_meta['obs'][sample_obs_key]['horizon']
        for key in self.current_obs_keys:
            assert self.shape_meta['obs'][key]['horizon'] == self.receding_obs_horizon, 'the current implementation requires that all receding observation keys have the same horizon'

        if self.concat_prompt_and_receding:
            assert self.prompt_current_obs_has_same_keys_as_current_obs, 'the prompt receding obs keys must have the same keys as the receding current obs keys if the receding and prompt are concatenated. This is because the receding obs is not separately represented for the prompt, so we just have one representation of the receding obs for the prompt and for the normal receding obs.'

        # action encoding
        assert not self.prompt_obs_encoder.concat_time_dimension
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_chunk_n_actions = shape_meta['prompt_chunk_n_actions']
        if not self.ignore_prompt_action:
            self.action_proj = Mlp(in_features=action_dim * action_chunk_n_actions, out_features=self.n_emb)

        # compute the number of modalities in the receding obs
        self.num_modalities_current_obs = self.receding_obs_encoder.output_shape(rgb_keys=self.current_obs_rgb_keys, low_dim_keys=self.current_obs_low_dim_keys)[2] # (B, T, num_modalities_receding_obs, D)
        assert self.num_modalities_current_obs == len(self.current_obs_keys)

        # compute the number of modalities in the prompt
        self.num_input_modalities_prompt_obs = 0 if self.prompt_has_no_obs_or_proprio else self.prompt_obs_encoder.output_shape(rgb_keys=self.prompt_rgb_keys, low_dim_keys=self.prompt_low_dim_keys)[2] # (B, T, num_input_modalities_prompt_obs, D)
        assert self.num_input_modalities_prompt_obs == len(self.prompt_obs_keys)
        self.num_input_modalities_prompt = self.num_input_modalities_prompt_obs + (0 if self.ignore_prompt_action else 1)

        # compute the number of modalities in the prompt receding obs
        self.num_modalities_prompt_current_obs = self.prompt_obs_encoder.output_shape(rgb_keys=self.prompt_current_obs_rgb_keys, low_dim_keys=self.prompt_current_obs_low_dim_keys)[2] # (B, T, num_modalities_prompt_current_obs, D)
        assert self.num_modalities_prompt_current_obs == len(self.prompt_current_obs_keys)
        
        if self.merge_prompt_tokens == 'obs':
            if self.ignore_prompt_action:
                self.num_output_modalities_prompt = 1 # just obs as a single token
            elif self.prompt_has_no_obs_or_proprio:
                self.num_output_modalities_prompt = 1 # just action
            else:
                self.num_output_modalities_prompt = 2 # obs as a single token and action
        elif self.merge_prompt_tokens == 'obs_and_action':
            self.num_output_modalities_prompt = 1 # obs and action together as a single token
        elif self.merge_prompt_tokens == 'none':
            self.num_output_modalities_prompt = self.num_input_modalities_prompt # separate obs per modality and action

        if use_pool_modality_pos_embed and merge_prompt_tokens == 'obs':
            self.pool_modality_pos_embed = nn.Parameter(torch.randn(self.num_input_modalities_prompt_obs, n_emb))
        elif use_pool_modality_pos_embed and merge_prompt_tokens == 'obs_and_action':
            self.pool_modality_pos_embed = nn.Parameter(torch.randn(self.num_input_modalities_prompt, n_emb))
        else:
            self.pool_modality_pos_embed = None

        self.init_weights()
    
    def forward(self, obs_dict: Dict):
        """Supports inputs where the prompt and current obs have different batch sizes. In this case, the prompt is expanded to the same batch size as the current obs. If concat_prompt_and_receding is True, then the receding obs is concatenated with the prompt and the prompt mask is expanded accordingly. If concat_prompt_and_receding is False, then the receding obs and the prompt are returned separately with a prompt_mask that is the same size as the prompt. Also supports inputs that have either only prompt or only receding obs in the case that concat_prompt_and_receding is False."""
        obs_dict = obs_dict.copy()

        prompt_present = 'prompt' in obs_dict
        receding_present = any(key in obs_dict for key in self.current_obs_keys)
        if receding_present:
            assert all(key in obs_dict for key in self.current_obs_keys)
        else:
            assert not any(key in obs_dict for key in self.current_obs_keys)

        if self.concat_prompt_and_receding:
            assert prompt_present and receding_present, 'if concat_prompt_and_receding is True, then both prompt and receding obs must be present'
        else:
            assert prompt_present or receding_present, 'if concat_prompt_and_receding is False, then either prompt or receding obs must be present, or both'

        # it's possible that the prompt has a different batch size than the current obs in the case where we provide a single prompt for all samples in the batch

        """Encode the prompt and receding obs, based on which are present"""
        if prompt_present:
            prompt = obs_dict['prompt']
            obs_dict.pop('prompt')

            if self.prompt_has_no_obs_or_proprio:
                prompt_B, prompt_len, _, _ = prompt['action'].shape
                prompt_tokens = torch.zeros((prompt_B, prompt_len, 0, self.n_emb), device=prompt['action'].device)
            else:
                prompt_tokens, _ = self.prompt_obs_encoder(prompt['obs'], rgb_keys=self.prompt_rgb_keys, low_dim_keys=self.prompt_low_dim_keys, is_prompt=True) # (prompt_B, prompt_len, num_modalities_prompt_obs, D)
            prompt_B, prompt_len, num_output_modalities_prompt_obs, D = prompt_tokens.shape
            assert num_output_modalities_prompt_obs == self.num_input_modalities_prompt_obs

            # actions
            if not self.ignore_prompt_action:
                prompt_actions = prompt['action'] # (prompt_B, prompt_len, chunk_n_actions, action_dim)
                prompt_actions = prompt_actions.reshape(prompt_B, prompt_len, -1) # (prompt_B, prompt_len, chunk_n_actions * action_dim)
                prompt_actions_embed = self.action_proj(prompt_actions) # (prompt_B, prompt_len, D)
                prompt_actions_embed = prompt_actions_embed.unsqueeze(2) # (prompt_B, prompt_len, 1, D)

                # put obs and action modalities in order along token dimension for prompt
                prompt_tokens = torch.cat([prompt_tokens, prompt_actions_embed], dim=2) # (prompt_B, prompt_len, num_input_prompt_modalities, D)

            # flatten modalities along time dimension
            prompt_tokens = prompt_tokens.view(prompt_B, prompt_len * self.num_input_modalities_prompt, D) # (prompt_B, prompt_len * num_input_prompt_modalities, D)

        if receding_present:
            receding_tokens, _ = self.receding_obs_encoder(obs_dict, rgb_keys=self.current_obs_rgb_keys, low_dim_keys=self.current_obs_low_dim_keys) # (receding_B, receding_len, num_modalities_receding_obs, D)
            receding_B, receding_len, num_modalities_receding_obs, D = receding_tokens.shape
            assert num_modalities_receding_obs == self.num_modalities_current_obs

            if not self.concat_prompt_and_receding:
                if self.separate_prompt_and_receding_obs_encoders:
                    prompt_receding_tokens, _ = self.prompt_obs_encoder(obs_dict, rgb_keys=self.prompt_current_obs_rgb_keys, low_dim_keys=self.prompt_current_obs_low_dim_keys)
                else:
                    if self.prompt_current_obs_has_same_keys_as_current_obs:
                        prompt_receding_tokens = receding_tokens
                    else:
                        # currently we have encoded the receding obs, but in this case the prompt receding obs has a subset of the keys in the normal receding obs, so we need to select out the particular observation modalities that are present in the prompt receding obs
                        #TODO: this assumes the prompt receding obs is a subset of the receding obs, but we should support non-overlapping keys as well
                        assert set(self.prompt_current_obs_keys).issubset(set(self.current_obs_keys)), 'the prompt receding obs keys must be a subset of the receding current obs keys.'
                        prompt_receding_tokens = receding_tokens[:, :, self.prompt_current_obs_subset_indices, :] # (receding_B, receding_len, num_modalities_prompt_current_obs, D)

            # put obs modalities in order along token dimension for receding obs
            receding_tokens = receding_tokens.view(receding_B, receding_len * num_modalities_receding_obs, D) # (receding_B, receding_len * num_modalities_receding_obs, D)
            if not self.concat_prompt_and_receding:
                prompt_receding_tokens = prompt_receding_tokens.view(receding_B, receding_len * self.num_modalities_prompt_current_obs, D) # (receding_B, receding_len * num_modalities_prompt_current_obs, D)

        if prompt_present and receding_present:
            assert prompt_B == receding_B or prompt_B == 1, 'prompt must have the same batch size as the current obs or be a single prompt for all samples in the batch'
        
        """Prepare the prompt mask (also handle only_use_last_step_of_prompt)"""
        prompt_mask = None
        if prompt_present:
            # prepare the mask
            if 'mask' not in prompt['metadata']:
                assert prompt_B == 1, 'if batch size greater than 1, mask must be provided'
                prompt_mask = None

                if self.only_use_last_step_of_prompt:
                    # since there is no mask we just select the last timestep for the single prompt provided
                    prompt_tokens = prompt_tokens[:, -self.num_input_modalities_prompt:] # (prompt_B, num_input_modalities_prompt, D)
                    prompt_len = 1
            else:
                prompt_mask = prompt['metadata']['mask'] # (prompt_B, prompt_len)

                if self.only_use_last_step_of_prompt:
                    # TODO: this could be more efficient both in terms of the for loop and in terms of only doing obs_encoder for the single timestep instead of trimming it down later as is done here, but since this is an ablation we are not going to worry about it for now
                    # select the last valid timestep for each sample in the batch
                    last_timestep_prompt_tokens = []
                    for i in range(prompt_B):
                        if torch.any(prompt_mask[i]):
                            loc = torch.where(prompt_mask[i])[0][-1] - 1
                        else:
                            loc = prompt_len - 1
                        loc = loc * self.num_input_modalities_prompt
                        prompt_section = prompt_tokens[i, loc:loc + self.num_input_modalities_prompt] # (num_input_modalities_prompt, D)
                        last_timestep_prompt_tokens.append(prompt_section)
                    last_timestep_prompt_tokens = torch.stack(last_timestep_prompt_tokens, dim=0) # (prompt_B, num_input_modalities_prompt, D)
                    prompt_tokens = last_timestep_prompt_tokens
                    prompt_mask = None # no need for mask anymore since prompt only length 1 and it's all valid
                    prompt_len = 1
                else:
                    prompt_mask = torch.repeat_interleave(prompt_mask, self.num_input_modalities_prompt, dim=1) # repeat each column because there are separate tokens for each obs modality and the actions

        """Merge the prompt tokens if needed"""
        if prompt_present:
            if self.merge_prompt_tokens == 'obs':
                # final format of prompt tokens is (pooled obs 0, action 0, pooled obs 1, action 1, ...)

                # extract the obs and action tokens from the prompt tokens
                prompt_tokens = prompt_tokens.view(prompt_B, prompt_len, self.num_input_modalities_prompt, D) # (prompt_B, prompt_len * self.num_input_modalities_prompt, D) -> (prompt_B, prompt_len, self.num_input_modalities_prompt, D)
                if not self.prompt_has_no_obs_or_proprio:
                    prompt_obs_tokens = prompt_tokens[:, :, :self.num_input_modalities_prompt_obs, :] # (prompt_B, prompt_len, self.num_input_modalities_prompt, D)

                    # use attention pool to merge the prompt obs tokens
                    prompt_obs_tokens = prompt_obs_tokens.view(prompt_B*prompt_len, self.num_input_modalities_prompt_obs, D) # (prompt_B*prompt_len, self.num_input_modalities_prompt, D); batch dimension now contains original batch dimension and prompt length, so that time dimension is just the observation modalities
                    if self.pool_modality_pos_embed is not None:
                        prompt_obs_tokens = prompt_obs_tokens + self.pool_modality_pos_embed
                    pooled_obs_tokens = self.prompt_attention_pool(prompt_obs_tokens) # (prompt_B*prompt_len, D)
                    pooled_obs_tokens = pooled_obs_tokens.view(prompt_B, prompt_len, D) # (prompt_B, prompt_len, D)

                # update the prompt mask (since all obs merged into a single token we just have one entry in the mask per prompt timestep)
                if prompt_mask is not None:
                    prompt_mask = prompt_mask[:, ::self.num_input_modalities_prompt] # (prompt_B, prompt_len * self.num_input_modalities_prompt) -> (prompt_B, prompt_len)

                if self.ignore_prompt_action:     
                    prompt_tokens = pooled_obs_tokens
                else:
                    prompt_action_tokens = prompt_tokens[:, :, -1:, :] # (prompt_B, prompt_len, 1, D)
                    if self.prompt_has_no_obs_or_proprio:
                        prompt_tokens = prompt_action_tokens.squeeze(2) # (prompt_B, prompt_len, 1, D) -> (prompt_B, prompt_len, D)
                    else:
                        pooled_obs_tokens = pooled_obs_tokens.unsqueeze(2) # (prompt_B, prompt_len, 1, D)
                        prompt_tokens = torch.cat([pooled_obs_tokens, prompt_action_tokens], dim=2) # (prompt_B, prompt_len, 2, D)
                        prompt_tokens = prompt_tokens.view(prompt_B, prompt_len * 2, D) # (prompt_B, prompt_len * 2, D)

                        # update the prompt mask (each timestep has obs and action, so we need to repeat the mask for each timestep)
                        if prompt_mask is not None:
                            prompt_mask = torch.repeat_interleave(prompt_mask, 2, dim=1) # (prompt_B, prompt_len) -> (prompt_B, prompt_len * 2)
            elif self.merge_prompt_tokens == 'obs_and_action':
                # final format of prompt tokens is (pooled obs 0, action 0, pooled obs 1, action 1, ...)

                # extract the obs and action tokens from the prompt tokens
                prompt_tokens = prompt_tokens.view(prompt_B, prompt_len, self.num_input_modalities_prompt, D) # (prompt_B, prompt_len * self.num_input_modalities_prompt, D) -> (prompt_B, prompt_len, self.num_input_modalities_prompt, D)

                # use attention pool to merge the prompt obs tokens
                prompt_tokens = prompt_tokens.view(prompt_B*prompt_len, self.num_input_modalities_prompt, D) # (prompt_B*prompt_len, self.num_input_modalities_prompt, D); batch dimension now contains original batch dimension and prompt length, so that time dimension is just the observation+action modalities
                if self.pool_modality_pos_embed is not None:
                    prompt_tokens = prompt_tokens + self.pool_modality_pos_embed
                prompt_tokens = self.prompt_attention_pool(prompt_tokens) # (prompt_B*prompt_len, D)
                prompt_tokens = prompt_tokens.view(prompt_B, prompt_len, D) # (prompt_B, prompt_len, D)

                # update the prompt mask
                if prompt_mask is not None:
                    prompt_mask = prompt_mask[:, ::self.num_input_modalities_prompt] # (prompt_B, prompt_len * self.num_input_modalities_prompt) -> (prompt_B, prompt_len)

        """Cut the first n steps of the prompt if needed"""
        if self.cut_first_n_steps_of_prompt > 0 and prompt_present:
            prompt_tokens = prompt_tokens[:, self.cut_first_n_steps_of_prompt*self.num_output_modalities_prompt:]
            if prompt_mask is not None:
                prompt_mask = prompt_mask[:, self.cut_first_n_steps_of_prompt*self.num_output_modalities_prompt:]

        """Concatenate the prompt and receding obs if needed"""
        if self.concat_prompt_and_receding:
            prompt_mask = torch.cat([torch.zeros((prompt_B, receding_len * self.num_modalities_prompt_current_obs), device=prompt_mask.device, dtype=torch.bool), prompt_mask], dim=1) # (prompt_B, receding_len * num_modalities_prompt_current_obs + prompt_len * self.num_output_modalities_prompt)

            if prompt_B == 1 and prompt_B != receding_B:
                # we need to expand the prompt data to the same batch size as the current obs. Expand doesn't increase memory usage, but the cat operator later on does, so this is not more memory efficient to use a single prompt paired with all samples in the batch
                prompt_tokens = prompt_tokens.expand(receding_B, -1, -1)

            # concatenate receding obs and prompt obs and actions
            obs_and_prompt_tokens = torch.cat([receding_tokens, prompt_tokens], dim=1) # (receding_B, receding_len * num_modalities_prompt_current_obs + prompt_len * self.num_output_modalities_prompt, D)

            if prompt_mask is not None:
                if prompt_B == 1 and prompt_B != receding_B:
                    prompt_mask = prompt_mask.expand(receding_B, -1)
                assert prompt_mask.shape[:2] == obs_and_prompt_tokens.shape[:2]
            return obs_and_prompt_tokens, prompt_mask
        else:
            assert prompt_mask is None or prompt_mask.shape[:2] == prompt_tokens.shape[:2]
            if prompt_present and receding_present:
                # both prompt and receding obs are present
                return prompt_receding_tokens, receding_tokens, prompt_tokens, prompt_mask
            elif prompt_present:
                # only prompt is present
                return prompt_tokens, prompt_mask
            else:
                # only receding obs is present
                return prompt_receding_tokens, receding_tokens
    
    def get_max_token_count(self):
        prompt_chunk_n_actions = self.shape_meta['prompt_chunk_n_actions']
        if self.only_use_last_step_of_prompt:
            max_prompt_len = 1
        else:
            max_prompt_len = math.ceil(self.shape_meta['max_sequence_length'] / prompt_chunk_n_actions)

        max_prompt_tokens = max_prompt_len * self.num_output_modalities_prompt
        num_current_obs_tokens = self.receding_obs_horizon * self.num_modalities_current_obs
        num_prompt_current_obs_tokens = self.receding_obs_horizon * self.num_modalities_prompt_current_obs
        return max_prompt_tokens, num_current_obs_tokens, num_prompt_current_obs_tokens

    def get_obs_names(self, prompt_len: int) -> List[str]:
        max_prompt_tokens, num_current_obs_tokens, num_prompt_current_obs_tokens = self.get_max_token_count()
        
        current_obs_modality_names = self.receding_obs_encoder.get_obs_names(rgb_keys=self.current_obs_rgb_keys, low_dim_keys=self.current_obs_low_dim_keys)
        prompt_obs_modality_names = self.prompt_obs_encoder.get_obs_names(rgb_keys=self.prompt_rgb_keys, low_dim_keys=self.prompt_low_dim_keys)
        prompt_current_obs_modality_names = self.prompt_obs_encoder.get_obs_names(rgb_keys=self.prompt_current_obs_rgb_keys, low_dim_keys=self.prompt_current_obs_low_dim_keys)

        prompt_chunk_n_actions = self.shape_meta['prompt_chunk_n_actions']

        # current obs names
        current_obs_names = []
        for i in range(self.receding_obs_horizon):
            for obs_name in current_obs_modality_names:
                current_obs_names.append(f'{obs_name} T-{self.receding_obs_horizon - i - 1}')
        assert len(current_obs_names) == num_current_obs_tokens

        # prompt current obs names
        prompt_current_obs_names = []
        for i in range(self.receding_obs_horizon):
            for obs_name in prompt_current_obs_modality_names:
                prompt_current_obs_names.append(f'{obs_name} T-{self.receding_obs_horizon - i - 1}')
        assert len(prompt_current_obs_names) == num_prompt_current_obs_tokens

        # figure out if proprio and obs prompt types are in use
        if self.ignore_prompt_proprio:
            prompt_obs_types = 'obs'
        elif self.ignore_prompt_obs:
            prompt_obs_types = 'proprio'
        elif not self.prompt_has_no_obs_or_proprio:
            prompt_obs_types = 'obs+proprio'
        
        # prompt obs names
        prompt_obs_names = []
        for i in range(self.cut_first_n_steps_of_prompt, prompt_len):
            action_range = f'{i*prompt_chunk_n_actions}->{i*prompt_chunk_n_actions + prompt_chunk_n_actions - 1}'
            if self.merge_prompt_tokens == 'obs':
                if not self.prompt_has_no_obs_or_proprio:
                    prompt_obs_names.append(f'{prompt_obs_types} {i}')

                if not self.ignore_prompt_action:
                    prompt_obs_names.append(f'action {action_range}')
            elif self.merge_prompt_tokens == 'none':
                for obs_name in prompt_obs_modality_names:
                    prompt_obs_names.append(f'{obs_name} {i}')

                if not self.ignore_prompt_action:
                    prompt_obs_names.append(f'action {action_range}')
            elif self.merge_prompt_tokens == 'obs_and_action':
                if self.ignore_prompt_action:
                    prompt_obs_names.append(f'{prompt_obs_types} {i}')
                elif self.prompt_has_no_obs_or_proprio:
                    prompt_obs_names.append(f'action {action_range}')
                else:
                    prompt_obs_names.append(f'{prompt_obs_types} {i} + action {action_range}')
        assert len(prompt_obs_names) == (prompt_len - self.cut_first_n_steps_of_prompt) * self.num_output_modalities_prompt

        if self.only_use_last_step_of_prompt:
            prompt_obs_names = prompt_obs_names[-self.num_output_modalities_prompt:]
        
        assert len(prompt_obs_names) <= max_prompt_tokens

        return current_obs_names, prompt_obs_names, prompt_current_obs_names
    
    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        optim_groups = []
        
        # obs encoder
        optim_groups.extend(self.receding_obs_encoder.get_optim_groups(lr=lr, weight_decay=weight_decay))
        if self.separate_prompt_and_receding_obs_encoders:
            optim_groups.extend(self.prompt_obs_encoder.get_optim_groups(lr=lr, weight_decay=weight_decay))

        # action proj
        if not self.ignore_prompt_action:
            optim_groups.append({
                "params": self.action_proj.parameters(),
                "weight_decay": weight_decay,
                "lr": lr
            })

        # prompt attention pool
        if self.prompt_attention_pool is not None:
            optim_groups.append({
                "params": self.prompt_attention_pool.parameters(),
                "weight_decay": weight_decay,
                "lr": lr
            })

        # modality positional embedding for attention pool
        if self.pool_modality_pos_embed is not None:
            optim_groups.append({
                "params": [self.pool_modality_pos_embed],
                "weight_decay": 0.0,
                "lr": lr
            })

        return optim_groups
    
    def init_weights(self):
        # obs encoder will handle its own weight initialization
        # we ignore weight initializations for the action proj
        # prompt attention pool will handle its own weight initialization
        pass

    def reset(self):
        self.receding_obs_encoder.reset()
        if self.separate_prompt_and_receding_obs_encoders:
            self.prompt_obs_encoder.reset()

class PairPromptObsEncoder(BaseTokenizedObsEncoder):
    """
    Two different uses:
    1) prompt_with_obs_decoder_enabled=True:
        - encode the prompt and receding obs
        - optionally encode the prompt with a transformer encoder
        - cross attend the receding obs and the prompt with transformer decoder to produce encoded prompt tokens
        - concatenate the receding obs and the encoded prompt tokens
    2) prompt_with_obs_decoder_enabled=False:
        - encode the prompt
        - encode the receding obs
        - concatenate the receding obs and the encoded prompt tokens
    """
    def __init__(self,
                 shape_meta: dict,
                 obs_encoder: PairPromptTransformerTokenizer,
                 prompt_encoder_enabled: bool=False,
                 prompt_encoder_n_layer: int=8,
                 prompt_encoder_n_head: int=8,
                 prompt_with_obs_decoder_enabled: bool=True,
                 prompt_with_obs_decoder_n_layer: int=4,
                 prompt_with_obs_decoder_n_head: int=8,
                 ignore_prompt: bool=False,
                 flatten_output: bool=False,
                 n_emb=768,
                 p_drop_attn=0.1,
                 p_drop_prompt_tokens=0.0,
                 use_attention_sink: bool=False,
                 num_attention_sink_tokens: int = 4,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.prompt_encoder_enabled = prompt_encoder_enabled
        self.prompt_encoder_n_layer = prompt_encoder_n_layer
        self.prompt_encoder_n_head = prompt_encoder_n_head
        self.prompt_with_obs_decoder_enabled = prompt_with_obs_decoder_enabled
        self.prompt_with_obs_decoder_n_layer = prompt_with_obs_decoder_n_layer
        self.prompt_with_obs_decoder_n_head = prompt_with_obs_decoder_n_head
        self.ignore_prompt = ignore_prompt
        self.flatten_output = flatten_output
        self.p_drop_prompt_tokens = p_drop_prompt_tokens
        # Only meaningful when we actually do prompt cross-attention.
        self.attention_sink_enabled = use_attention_sink and prompt_with_obs_decoder_enabled
        self.num_attention_sink_tokens = int(num_attention_sink_tokens)
        if self.attention_sink_enabled and self.num_attention_sink_tokens < 1:
            raise ValueError(
                f"num_attention_sink_tokens must be >= 1 when use_attention_sink is enabled, got {self.num_attention_sink_tokens}"
            )

        self.is_prompted = False
        self.prompt_dict = None
        self.prompt_encoder_enabled = prompt_encoder_enabled
        self.prompt_with_obs_decoder_enabled = prompt_with_obs_decoder_enabled
        self.current_obs_keys = get_obs_keys_from_shape_meta(shape_meta, for_policy=True)['current_obs_all']

        if self.ignore_prompt and not self.prompt_with_obs_decoder_enabled:
            raise NotImplementedError('`ignore_prompt` is currently only supported when `prompt_with_obs_decoder_enabled` is True')

        if self.prompt_encoder_enabled:
            assert self.prompt_with_obs_decoder_enabled, '`prompt_encoder_enabled` requires `prompt_with_obs_decoder_enabled` to be True'

        if self.flatten_output:
            assert self.prompt_with_obs_decoder_enabled, '`flatten_output` requires `prompt_with_obs_decoder_enabled` to be True as otherwise the output would have variable length'

        if self.p_drop_prompt_tokens > 0:
            self.dropped_token_embedding = nn.Parameter(torch.zeros(1, 1, n_emb))  # learned "missing frame" embedding when tokens are dropped

        if prompt_encoder_enabled:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=prompt_encoder_n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                batch_first=True,
                norm_first=True # important for stability
            )
            self.prompt_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=prompt_encoder_n_layer
            )

        if prompt_with_obs_decoder_enabled:
            decoder_layer = TransformerDecoderLayerWithAttn(
                d_model=n_emb,
                nhead=prompt_with_obs_decoder_n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.prompt_with_obs_decoder = TransformerDecoderWithAttn(
                decoder_layer=decoder_layer,
                num_layers=prompt_with_obs_decoder_n_layer
            )

            max_prompt_tokens, num_current_obs_tokens, num_prompt_current_obs_tokens = self.obs_encoder.get_max_token_count()
            self.prompt_pos_emb = nn.Parameter(torch.randn((1, max_prompt_tokens, n_emb)))
            self.prompt_current_obs_pos_emb = nn.Parameter(torch.randn((1, num_prompt_current_obs_tokens, n_emb)))
            if self.attention_sink_enabled:
                # Learnable attention-sink tokens prepended to prompt/memory for cross-attention.
                # They are removed from the returned attention weights for visualization/analysis.
                self.attention_sink_tokens = nn.Parameter(
                    torch.zeros(1, self.num_attention_sink_tokens, n_emb)
                )
        else:
            assert obs_encoder.concat_prompt_and_receding, '`obs_encoder.concat_prompt_and_receding` must be True when `prompt_with_obs_decoder_enabled` is False'

        self.init_weights()

    def init_weights(self) -> None:
        if self.prompt_encoder_enabled:
            self.prompt_encoder.apply(init_weights)
        
        if self.prompt_with_obs_decoder_enabled:
            self.prompt_with_obs_decoder.apply(init_weights)

            torch.nn.init.normal_(self.prompt_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.prompt_current_obs_pos_emb, mean=0.0, std=0.02)
            if self.attention_sink_enabled:
                torch.nn.init.normal_(self.attention_sink_tokens, mean=0.0, std=0.02)

    def forward(self, obs_dict: Dict[str, torch.Tensor], need_weights: bool=False, average_attn_weights: bool=False, *args, **kwargs) -> Tuple[torch.Tensor, Optional[Dict]]:
        if self.is_prompted:
            assert 'prompt' not in obs_dict, 'prompt should not be provided in the obs_dict after `prompt` is called. Call `reset` to reset the policy to remove the prompt.'

            if not self.prompt_with_obs_decoder_enabled:
                obs_dict['prompt'] = self.prompt_dict
        
        cross_attn_weights = None

        # process input
        if self.prompt_with_obs_decoder_enabled:
            prompt_present = 'prompt' in obs_dict
            receding_present = any(key in obs_dict for key in self.current_obs_keys)
            if receding_present:
                assert all(key in obs_dict for key in self.current_obs_keys)
            else:
                assert not any(key in obs_dict for key in self.current_obs_keys)

            def prepare_prompt_tokens(prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                """Apply dropout, positional embedding, and optional prompt encoder."""
                prompt_tokens = self.drop_tokens(prompt_tokens)
                assert prompt_tokens.shape[1] <= self.prompt_pos_emb.shape[1], f"prompt length ({prompt_tokens.shape[1]}) is greater than prompt position embedding length ({self.prompt_pos_emb.shape[1]})"
                prompt_tokens = prompt_tokens + self.prompt_pos_emb[:, :prompt_tokens.shape[1], :]

                if self.prompt_encoder_enabled:
                    prompt_tokens = self.prompt_encoder(prompt_tokens)
                return prompt_tokens

            if prompt_present and receding_present:
                # both prompt and receding obs are present so we encode both (this is used during training in `compute_loss`)
                prompt_receding_tokens, receding_tokens, prompt_tokens, prompt_mask = self.obs_encoder(obs_dict)
                prompt_tokens = prepare_prompt_tokens(prompt_tokens)
            elif prompt_present:
                # only prompt is present so we encode the prompt (this is used when we call `prompt`)
                prompt_tokens, prompt_mask = self.obs_encoder(obs_dict)
                prompt_tokens = prepare_prompt_tokens(prompt_tokens)
                return prompt_tokens, prompt_mask
            elif receding_present:
                # only receding is present so we encode only the receding and then used the cached encoded prompt (this is used when we call `predict_action` after calling `prompt`)
                prompt_receding_tokens, receding_tokens = self.obs_encoder(obs_dict)
                assert self.is_prompted
                prompt_tokens = self.prompt_tokens_cache
                prompt_mask = self.prompt_mask_cache

            # include the prompt, so we need to cross attend the prompt receding tokens and the prompt tokens and then concatenate that result with the non-prompt receding tokens
            prompt_B = prompt_tokens.shape[0]
            receding_B = receding_tokens.shape[0]
            if self.attention_sink_enabled:
                # Prepend sinks before any potential batch expand so this concat allocates at
                # prompt batch size (often 1), preserving expand's memory advantage.
                sink = self.attention_sink_tokens.expand(
                    prompt_tokens.shape[0], self.num_attention_sink_tokens, -1
                )  # (B, num_attention_sink_tokens, D)
                prompt_tokens = torch.cat([sink, prompt_tokens], dim=1)

                if prompt_mask is not None:
                    sink_mask = torch.zeros(
                        (prompt_mask.shape[0], self.num_attention_sink_tokens),
                        device=prompt_mask.device,
                        dtype=prompt_mask.dtype,
                    )
                    prompt_mask = torch.cat([sink_mask, prompt_mask], dim=1)

            if prompt_B == 1 and prompt_B != receding_B:
                # we use `expand` to save memory by creating a view along the batch dimension. We have to match the batch dimension for the cross attention to work, but using `expand` doesn't increase memory usage.
                prompt_tokens = prompt_tokens.expand(receding_B, -1, -1)
                if prompt_mask is not None:
                    prompt_mask = prompt_mask.expand(receding_B, -1)

            ret = self.prompt_with_obs_decoder(
                tgt=prompt_receding_tokens + self.prompt_current_obs_pos_emb,
                memory=prompt_tokens,
                memory_key_padding_mask=prompt_mask,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights
            ) # we merge the receding and prompt tokens by cross attending the receding tokens and the prompt tokens

            if need_weights:
                encoded_prompt_tokens, cross_attn_weights = ret
                if self.attention_sink_enabled:
                    # Remove sink token columns from returned attention weights.
                    cross_attn_weights = cross_attn_weights[..., self.num_attention_sink_tokens:]
            else:
                encoded_prompt_tokens = ret

            if self.ignore_prompt:
                # do not include the prompt, only include the receding tokens
                receding_and_prompt_tokens = receding_tokens + 0*encoded_prompt_tokens # we have to include the encoded_prompt_tokens so that accelerate doesn't complain about not all model parameters being used
                # TODO: this might actually error now that it's not guaranteed that the receding tokens and the encoded_prompt_tokens have the same num modality dimension. Easy fix is just select out on single item from encoded_prompt_tokens and then multiply by 0 and add it
            else:
                # concatenate the receding tokens and the encoded_prompt_tokens so the action diffusion head will have direct access to the receding tokens as well as indirect access to the prompt (through the obs-with-prompt decoder)
                receding_and_prompt_tokens = torch.cat([receding_tokens, encoded_prompt_tokens], dim=1)

            prompt_mask = None # once the receding tokens have attended to the prompt, the resulting output dimension is fixed and does not need masking

            if self.flatten_output:
                receding_and_prompt_tokens = receding_and_prompt_tokens.view(receding_and_prompt_tokens.shape[0], -1)
        else:
            # if we are not using the prompt with obs decoder, we also assume that the concat_prompt_and_receding is True in the base obs_encoder, so we already have the receding and prompt tokens concatenated
            receding_and_prompt_tokens, prompt_mask = self.obs_encoder(obs_dict)

        metadata = {}
        if not self.flatten_output:
            metadata['token_mask'] = prompt_mask

        if need_weights:
            metadata['prompt_with_obs_transfomer_attention_weights'] = cross_attn_weights

        return receding_and_prompt_tokens, metadata
    
    """BaseObsEncoder methods"""
    def supports_prompting(self):
        return True
    
    def prompt(self, prompt_dict: dict):
        if self.prompt_with_obs_decoder_enabled:
            # we can encode the prompt once and then used the cached result for each action prediction step
            obs_dict = {'prompt': prompt_dict}
            self.prompt_tokens_cache, self.prompt_mask_cache = self.forward(obs_dict)
        else:
            # we just save the raw prompt to be encoded later. Theoretically this could be make more efficient by encoding the prompt once and then concatenating the result with the receding obs later instead of reencoding the prompt everytime, but it's ok for now.
            self.prompt_dict = prompt_dict
        self.is_prompted = True
    
    def reset(self) -> None:
        self.prompt_dict = None
        self.is_prompted = False
        self.prompt_tokens_cache = None
        self.prompt_mask_cache = None

        self.obs_encoder.reset()

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        optim_groups = []
        
        # obs_encoder
        optim_groups.extend(self.obs_encoder.get_optim_groups(lr=lr, weight_decay=weight_decay))

        if self.p_drop_prompt_tokens > 0:
            optim_groups.append({
                "params": [self.dropped_token_embedding],
                "weight_decay": 0.0,
                "lr": lr
            })

        if self.prompt_encoder_enabled:
            # prompt encoder
            optim_groups.extend(get_optim_groups(self.prompt_encoder, lr=lr, weight_decay=weight_decay))

        if self.prompt_with_obs_decoder_enabled:
            # prompt with obs decoder
            optim_groups.extend(get_optim_groups(self.prompt_with_obs_decoder, lr=lr, weight_decay=weight_decay))

            # prompt_pos_emb, prompt_current_obs_pos_emb do not experience weight decay
            optim_groups.append({
                "params": [self.prompt_pos_emb, self.prompt_current_obs_pos_emb],
                "weight_decay": 0.0,
                "lr": lr
            })

            if self.attention_sink_enabled:
                optim_groups.append({
                    "params": [self.attention_sink_tokens],
                    "weight_decay": 0.0,
                    "lr": lr
                })

        return optim_groups

    @torch.inference_mode()
    def output_shape(self) -> Tuple[int, ...]:
        obs_shape_meta = self.shape_meta['obs']
        example_obs_dict = dict()
        example_obs_dict['prompt'] = {
            'obs': {},
            'metadata': {}
        }

        sample_prompt_len = 3
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs

            this_obs_prompt = torch.zeros(
                (1, sample_prompt_len) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict['prompt']['obs'][key] = this_obs_prompt

        prompt_chunk_n_actions = self.shape_meta['prompt_chunk_n_actions']
        this_action = torch.zeros(
            (1, sample_prompt_len, prompt_chunk_n_actions) + tuple(self.shape_meta['action']['shape']), 
            dtype=self.dtype,
            device=self.device) # (B, T, chunk_dim, action_dim)
        example_obs_dict['prompt']['action'] = this_action
 
        example_output, _ = self.forward(example_obs_dict)
        
        return example_output.shape

    """BaseTokenizedObsEncoder methods"""
    def get_max_token_count(self):
        assert not self.flatten_output, '`flatten_output` is not supported in `get_max_token_count`'
        
        max_prompt_tokens, num_current_obs_tokens, num_prompt_current_obs_tokens = self.obs_encoder.get_max_token_count()

        if self.prompt_with_obs_decoder_enabled:
            max_token_count = num_current_obs_tokens + num_prompt_current_obs_tokens # the output is the current obs and the encoded prompt (which has the dimensions of the prompt current obs)
        else:
            max_token_count = num_current_obs_tokens + max_prompt_tokens
        
        return max_token_count

    def get_output_token_names(self, obs_len: int) -> List[str]:
        assert not self.flatten_output, '`flatten_output` is not supported in `get_output_token_names`'
        
        current_obs_names, prompt_obs_names, prompt_current_obs_names = self.obs_encoder.get_obs_names(obs_len)
        if self.prompt_with_obs_decoder_enabled:
            if self.ignore_prompt:
                return current_obs_names
            else:
                encoded_prompt_dim_names = [f'encoded prompt {i}' for i in range(len(prompt_current_obs_names))]
                return current_obs_names + encoded_prompt_dim_names
        else:
            return current_obs_names + prompt_obs_names

    def get_prompt_cross_attn_dim_names(self, prompt_len: int):
        assert self.prompt_with_obs_decoder_enabled, 'get_prompt_cross_attn_dim_names is only supported when prompt_with_obs_decoder_enabled is True'

        current_obs_names, prompt_obs_names, prompt_current_obs_names = self.obs_encoder.get_obs_names(prompt_len)
        return prompt_current_obs_names, prompt_obs_names

    def drop_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Performs token level dropout. Replaces dropped tokens with a learned "missing frame" embedding."""
        if self.p_drop_prompt_tokens > 0 and self.training:
            keep_mask = (torch.rand(tokens.shape[0], tokens.shape[1], 1, device=tokens.device) > self.p_drop_prompt_tokens).float() # (B, T, 1)
            tokens = tokens * keep_mask + self.dropped_token_embedding * (1 - keep_mask)
        return tokens
