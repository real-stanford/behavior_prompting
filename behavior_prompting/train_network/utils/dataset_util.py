import math
from torch.utils.data import Dataset, Subset
from behavior_prompting.common.replay_buffer import ReplayBuffer
import numpy as np
from typing import Optional, List

def pad_dataset_to_length(dataset: Dataset, length: int) -> Dataset:
    # pads a dataset to a given length by repeating the dataset
    if len(dataset) < length:
        num_repeats = math.ceil(length / len(dataset))
        indices = list(range(len(dataset))) * num_repeats
        indices = indices[:length]  # Truncate to exact length needed
        padded_dataset = Subset(dataset, indices)
    elif len(dataset) > length:
        # truncate the dataset
        indices = list(range(length))
        padded_dataset = Subset(dataset, indices)
    else:
        padded_dataset = dataset

    return padded_dataset

def prepare_only_task_names(replay_buffer: ReplayBuffer, only_task_names: Optional[List[str]], max_tasks: Optional[int] = None) -> List[str]:
    # TODO: potentially we want to add an option to shuffle and then select max tasks since this will always select the first tasks in the replay buffer
    if only_task_names is None:
        _, idx = np.unique(replay_buffer.task_names[:], return_index=True)
        unique_task_names = replay_buffer.task_names[np.sort(idx)]
        if max_tasks is not None:
            return unique_task_names[:max_tasks]
        else:
            return unique_task_names
    else:
        _, idx = np.unique(replay_buffer.task_names[:], return_index=True)
        unique_task_names = replay_buffer.task_names[np.sort(idx)]
        
        for i, task_name in enumerate(only_task_names):
            if type(task_name) == int:
                only_task_names[i] = unique_task_names[task_name]
        
        for task_name in only_task_names:
            assert task_name in unique_task_names, f"task {task_name} is not in the replay buffer"

        assert len(only_task_names) == len(set(only_task_names)), 'only_task_names must be a list of unique task names'
    
        if max_tasks is not None:
            assert len(only_task_names) <= max_tasks, "only_task_names must be less than or equal to max_tasks"
        
        return [x for x in unique_task_names if x in only_task_names] # maintains the order of the task names in the original list
