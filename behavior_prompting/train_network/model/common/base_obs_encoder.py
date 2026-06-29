from typing import Dict, List, Optional, Tuple
import torch
from behavior_prompting.train_network.model.common.module_attr_mixin import ModuleAttrMixin

class BaseObsEncoder(ModuleAttrMixin):
    def supports_prompting(self) -> bool:
        return False
        
    def prompt(self, prompt_dict: dict) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass
    
    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        raise NotImplementedError
    
    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
    
    def forward(self, obs_dict: dict, *args, **kwargs) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Returns:
            encoded_obs
            metadata: dict or None
        """
        raise NotImplementedError

class BaseTokenizedObsEncoder(BaseObsEncoder):
    def get_max_token_count(self):
        raise NotImplementedError
    
    def get_output_token_names(self, obs_len: int) -> List[str]:
        raise NotImplementedError
    
    # additional requirement is that the forward method returns includes a 'token_mask' key in the metadata that is returned
