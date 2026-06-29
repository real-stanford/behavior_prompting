"""Custom dataloaders for prompt training"""

import math
import torch
from typing import Optional

from torch.utils.data import Dataset, DataLoader, Subset

from behavior_prompting.common.pytorch_util import add_batch_dim
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset

class BatchedByTaskDataset(Dataset):
    """
    Given a multi-task dataset, creates a Dataset that samples by task. Each sample from this dataset will be one batch of data from a single task. This will also optionally reweight the sampling to be uniform across tasks even if the dataset is not balanced if `balanced` is True.
    
    If a prompt dataset is provided, it will sample a single prompt of the same task to go with each batch from the dataset.
    """
    def __init__(self, dataset: BaseDataset, dataloader_config: dict, shape_meta: dict, balanced: bool, prompt_dataset: Optional[BaseDataset]=None):
        dataloader_config = dataloader_config.copy()
        dataloader_config.update({
            'shuffle': True,
            'num_workers': 0, # we just load one batch from each dataloader we create so we don't need to use multiple workers
            'pin_memory': False,
            'persistent_workers': False
        })
        # prefetch factor should not be set when num_workers is 0
        if 'prefetch_factor' in dataloader_config:
            dataloader_config.pop('prefetch_factor')
        self.dataset = dataset
        self.dataloader_config = dataloader_config
        self.shape_meta = shape_meta
        self.prompt_dataset = prompt_dataset
        self.balanced = balanced

        unique_task_name_to_dataset_indices = dataset.get_unique_task_name_to_dataset_indices()

        # Create dataloaders for each task
        self.datasets = {}
        for task_name, dataset_indices in unique_task_name_to_dataset_indices.items():
            self.datasets[task_name] = Subset(self.dataset, dataset_indices)

        # If a prompt_dataset is provided, we need to create separate dataloaders for the prompts per task
        if self.prompt_dataset is not None:
            unique_task_name_to_prompt_dataset_indices = self.prompt_dataset.get_unique_task_name_to_dataset_indices()

            self.prompt_datasets = {}
            for task_name, dataset_indices in unique_task_name_to_prompt_dataset_indices.items():
                self.prompt_datasets[task_name] = Subset(self.prompt_dataset, dataset_indices)

        if self.balanced:
            max_samples_per_task = max(len(self.datasets[task_name]) for task_name in self.datasets)
            
        # Map from the dataset index to the task name
        cur_index = 0
        self.dataset_index_to_task_name = {}
        for task_name in sorted(list(unique_task_name_to_dataset_indices.keys())):
            cur_len = max_samples_per_task if self.balanced else len(self.datasets[task_name])
            num_batches = math.ceil(cur_len / self.dataloader_config['batch_size'])
            for _ in range(num_batches):
                self.dataset_index_to_task_name[cur_index] = task_name
                cur_index += 1
        self.length = cur_index

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # create a new dataloader to sample a single batch from the dataset. Since shuffle is True, we will get a different batch each time we call this function. Note that with this approach we can't guarantee that every datapoint is only seen once each epoch, but it's good enough for training purposes
        if self.prompt_dataset is not None:
            assert self.dataset.get_ignore_prompt(), 'prompt dataset is provided but the non-prompt dataset is not ignoring the prompt'

        task_name = self.dataset_index_to_task_name[idx]
        dataset = self.datasets[task_name]
        dataloader = DataLoader(dataset, **self.dataloader_config)
        batch = next(iter(dataloader))

        if self.prompt_dataset is not None:
            # if we are using pair prompting, we need to sample a single prompt to go with the samples. we just randomly sample a prompt from the prompt dataset
            prompt_dataset = self.prompt_datasets[task_name]
            prompt_idx = torch.randint(0, len(prompt_dataset), (1,)).item()
            prompt = prompt_dataset[prompt_idx]['obs']['prompt']
            prompt = add_batch_dim(prompt)
            assert 'prompt' not in batch['obs'], 'prompt already exists in batch'
            batch['obs']['prompt'] = prompt

        batch['metadata']['task_name'] = task_name
        return batch
