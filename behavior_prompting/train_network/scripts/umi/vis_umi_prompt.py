#!/usr/bin/env python3

import os
import hydra
from omegaconf import DictConfig

from behavior_prompting.common.pytorch_util import move_batch_to_numpy
from behavior_prompting.train_network.dataset.umi_task_dataset import UmiTaskDataset
from behavior_prompting.train_network.utils.umi_util import vis_prompt

@hydra.main(version_base=None, config_path="../../config")
def main(cfg: DictConfig) -> None:
    # Create dataset
    print("Loading UmiTaskDataset...")
    dataset: UmiTaskDataset = hydra.utils.instantiate(cfg.task.dataset, only_prompt=True, val_ratio=0.0)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Load a sample prompt
    print("Loading a prompt sample...")
    sample_idx = cfg.task.dataset.seed % len(dataset)  # Use first sample
    prompt_data = dataset[sample_idx]['obs']['prompt']

    prompt_data = move_batch_to_numpy(prompt_data)
    
    print(f"Prompt data keys: {list(prompt_data.keys())}")
    print(f"Prompt obs keys: {list(prompt_data['obs'].keys())}")
    
    main_camera_key = next((k for k in ['camera_right_main_rgb', 'camera_left_main_rgb'] if k in prompt_data['obs']), None)
    ultrawide_camera_key = next((k for k in ['camera_right_ultrawide_rgb', 'camera_left_ultrawide_rgb'] if k in prompt_data['obs']), None)
    main_camera_shape = prompt_data['obs'][main_camera_key].shape
    ultrawide_camera_shape = prompt_data['obs'][ultrawide_camera_key].shape
    print(f"Main camera shape: {main_camera_shape}")
    print(f"Ultrawide camera shape: {ultrawide_camera_shape}")
    
    # Create output directory
    output_dir = 'tmp_vis_umi_prompt_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the prompt
    output_path = os.path.abspath(os.path.join(output_dir, 'prompt.mp4'))
    
    vis_prompt(prompt_data, output_path)
    
    print("Visualization complete! See output at: ", output_path)

if __name__ == "__main__":
    main()
