import torch
from typing import Dict, Optional

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.model.common.normalizer import Normalizer

class BaseDataset(torch.utils.data.Dataset):

    # the constructor should have the arguments:
    # - `dataset_path` to the path to the dataset
    # - `replay_buffer` which is a ReplayBuffer object (optional) so that the dataset can be created using an already loaded replay buffer instead of loading it from a path
    # - `training_split_info` which is a dictionary mapping "demonstration_name:task_name" to a boolean indicating whether the dataset is in the training split and, if provided, should be used to create the training and validation masks.

    def __init__(self,
                 dataset_path: Optional[str]=None,
                 replay_buffer: Optional[ReplayBuffer]=None,
                 training_split_info: Optional[Dict[str, bool]]=None,
                 ):
        raise NotImplementedError
    
    def get_validation_dataset(self) -> 'BaseDataset':
        raise NotImplementedError

    def get_normalizer(self, **kwargs) -> Normalizer:
        raise NotImplementedError

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()

    def shuffle_data_ordering(self, seed:int) -> None:
        """For prompt based policies, we need to shuffle the data ordering since we sample large regions of the data and thus the ordering of tasks in the dataset matters. Typically this is called once per epoch when using prompting"""
        raise NotImplementedError

    def requires_epoch_shuffle(self) -> bool:
        """Whether the dataset requires shuffling. For prompt based policies, we need to shuffle the data ordering since we sample large regions of the data and thus the ordering of tasks in the dataset matters."""
        raise NotImplementedError

    def is_multi_task(self) -> bool:
        raise NotImplementedError
    
    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        """Returns a dictionary mapping task names to dataset indices. This is used for multi-task training and evaluation."""
        raise NotImplementedError

    def get_training_split_info(self) -> Dict[str, bool]:
        """Get the dataset information. Returns a dictionary mapping "demonstration_name:task_name" to a boolean indicating whether the dataset is in the training split."""
        raise NotImplementedError

    def set_ignore_prompt(self, ignore_prompt: bool):
        """Tells the dataset whether to ignore the prompt when sampling data. This can be freely enabled or disabeld at any time based on what the user wants to do with the dataset."""
        raise NotImplementedError
    
    def get_ignore_prompt(self) -> bool:
        """Returns whether the dataset is ignoring the prompt when sampling data."""
        raise NotImplementedError
