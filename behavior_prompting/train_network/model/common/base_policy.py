from typing import Dict, Optional
import torch

from behavior_prompting.train_network.model.common.module_attr_mixin import ModuleAttrMixin
from behavior_prompting.train_network.model.common.normalizer import Normalizer

class BasePolicy(ModuleAttrMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_split_info = None

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        fixed_action_prefix:
            B, Tp, Da
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def predict_action_training(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Some policies have different rollout and policy training action prediction methods. For example, prompting methods will predict actions in parallel for every observation in the training batch which consists of a long prompt+rollout, while for standard receeding horizon diffusion policy there is only a single action prediction for each observation."""

        # default behavior is with just observation as input. Past actions are not normally input into model except for prompting based policies
        return self.predict_action(obs_dict)
    
    def supports_prompting(self) -> bool:
        raise NotImplementedError

    @torch.inference_mode()
    def prompt(self, prompt_dict: Dict):
        raise NotImplementedError
    
    def num_available_actions(self) -> Optional[int]:
        """Returns the number of actions that are available to execute. Returns None if no limit is imposed."""
        return None
    
    # reset state for stateful policies
    def reset(self, action_exec_horizon=None):
        """The actual horizon of the action sequence to execute. This should be <= the horizon of the policy."""
        raise NotImplementedError

    def set_normalizer(self, normalizer: Normalizer):
        raise NotImplementedError()

    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError

    def register_training_split_info(self, training_split_info: Dict[str, bool]) -> None:
        """For prompting models it's important that we know which demonstrations were in the training split because at test time we want to prompt the model with a demonstration that we trained on to help the model stay more in distribution"""
        self._training_split_info = training_split_info
    
    def get_training_split_info(self) -> Dict[str, bool]:
        return self._training_split_info
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state['_extra_training_split_info'] = self._training_split_info
        return state

    def _apply_backward_compatibility_renames(self, state_dict: Dict):
        """
        Apply backward compatibility renames for old checkpoint formats.
        
        Handles:
        1. obs_encoder.obs_encoder.obs_encoder.* -> obs_encoder.obs_encoder.prompt_obs_encoder.* 
           and obs_encoder.obs_encoder.receding_obs_encoder.*
        2. obs_encoder.prompt_with_obs_obs_pos_emb.* -> obs_encoder.prompt_current_obs_pos_emb.*
        """
        # Backward compatibility: handle old checkpoint format
        # Old prefix: obs_encoder.obs_encoder.obs_encoder.*
        # New prefixes: obs_encoder.obs_encoder.prompt_obs_encoder.* and obs_encoder.obs_encoder.receding_obs_encoder.*
        old_obs_encoder_prefix = 'obs_encoder.obs_encoder.obs_encoder'
        keys_to_remove = []
        keys_to_add = {}
        
        for key in list(state_dict.keys()):
            if key.startswith(old_obs_encoder_prefix + '.'):
                # Extract the suffix after the old prefix
                suffix = key[len(old_obs_encoder_prefix):]
                # Create new keys for both prompt and receding encoders
                prompt_key = 'obs_encoder.obs_encoder.prompt_obs_encoder' + suffix
                receding_key = 'obs_encoder.obs_encoder.receding_obs_encoder' + suffix
                # Only add if they don't already exist
                if prompt_key not in state_dict:
                    keys_to_add[prompt_key] = state_dict[key]
                    print(f"Renamed key: {key} -> {prompt_key}")
                if receding_key not in state_dict:
                    keys_to_add[receding_key] = state_dict[key]
                    print(f"Renamed key: {key} -> {receding_key}")
                keys_to_remove.append(key)
            elif key == old_obs_encoder_prefix:
                # Handle exact match (no suffix)
                prompt_key = 'obs_encoder.obs_encoder.prompt_obs_encoder'
                receding_key = 'obs_encoder.obs_encoder.receding_obs_encoder'
                keys_to_add[prompt_key] = state_dict[key]
                keys_to_add[receding_key] = state_dict[key]
                print(f"Renamed key: {key} -> {prompt_key}")
                print(f"Renamed key: {key} -> {receding_key}")
                keys_to_remove.append(key)
        
        # Apply the changes
        state_dict.update(keys_to_add)
        for key in keys_to_remove:
            del state_dict[key]
        
        # Backward compatibility: handle renamed prompt_current_obs_pos_emb
        # Old prefix: obs_encoder.prompt_with_obs_obs_pos_emb.*
        # New prefix: obs_encoder.prompt_current_obs_pos_emb.*
        old_pos_emb_prefix = 'obs_encoder.prompt_with_obs_obs_pos_emb'
        keys_to_remove = []
        keys_to_add = {}
        
        for key in list(state_dict.keys()):
            if key.startswith(old_pos_emb_prefix + '.'):
                # Extract the suffix after the old prefix
                suffix = key[len(old_pos_emb_prefix):]
                # Create new key
                new_key = 'obs_encoder.prompt_current_obs_pos_emb' + suffix
                # Only add if it doesn't already exist
                if new_key not in state_dict:
                    keys_to_add[new_key] = state_dict[key]
                    print(f"Renamed key: {key} -> {new_key}")
                keys_to_remove.append(key)
            elif key == old_pos_emb_prefix:
                # Handle exact match (no suffix)
                new_key = 'obs_encoder.prompt_current_obs_pos_emb'
                if new_key not in state_dict:
                    keys_to_add[new_key] = state_dict[key]
                    print(f"Renamed key: {key} -> {new_key}")
                keys_to_remove.append(key)
        
        # Apply the changes
        state_dict.update(keys_to_add)
        for key in keys_to_remove:
            del state_dict[key]

    def load_state_dict(self, state_dict, strict=True):
        self._training_split_info = state_dict.pop('_extra_training_split_info', None)
        self._apply_backward_compatibility_renames(state_dict) # TODO: can eventually remove this
        return super().load_state_dict(state_dict, strict)
