# iPhUMI

## Setup
Follow the instructions in the [iPhUMI repository](https://github.com/real-stanford/iPhUMI) for gripper build steps, data collection instructions, and generation of zarr datasets. We pick up here assuming you have generated a `.zarr.zip` dataset, or you can use the sample ones below.

### Download Sample Data
We provide sample datasets you can [download from Hugging Face](https://huggingface.co/datasets/austinpatel/iphumi_example_zarr_datasets) to get started directly with single arm and bimanual training.
```bash
cd behavior_prompting/train_network/datasets # go into the datasets folder
mkdir umi && cd umi
hf download austinpatel/iphumi_example_zarr_datasets --repo-type=dataset --local-dir ./iphumi_example_zarr_datasets
```

You can also access iPhUMI data that is uploaded to the [UMI Data Initiative](https://umi-data.github.io/).

### Unzip .zarr.zip → .zarr
The policy training expects `.zarr` instead of `.zarr.zip`.
```bash
cd behavior_prompting/scripts
python unzip_zarr.py -i <path to .zarr.zip>
```

## Training

The iPhUMI configs are structured around two base configs: [umi.yaml](../behavior_prompting/train_network/config/task/umi.yaml) and [umi_bimanual.yaml](../behavior_prompting/train_network/config/task/umi_bimanual.yaml). It's worthwile to look at these task configs and the general [config directory structure](../behavior_prompting/train_network/config/) to understand how the configuration system works.

### Single arm training

With no task conditioning (24GB GPUs):
```bash
cd behavior_prompting/train_network/
accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=umi_policy_dunet \
  task.dataset.dataset_path=datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-single.zarr \
  exp_name="example_data_single_arm" \
  group_tag="initial_experiments"
```

Supported `--config-name` values:
- [umi_policy_dunet](../behavior_prompting/train_network/config/umi_policy_dunet.yaml) (24GB) - no task conditioning
- [umi_policy_dunet_goal_image](../behavior_prompting/train_network/config/umi_policy_dunet_goal_image.yaml) (24GB) - goal image conditioning
- [umi_policy_dunet_language](../behavior_prompting/train_network/config/umi_policy_dunet_language.yaml) (24GB) - language conditioning
- [umi_policy_dunetp](../behavior_prompting/train_network/config/umi_policy_dunetp.yaml) (46GB) - behavior prompting policy

### Bimanual training
With no task conditioning (48GB GPUs):
```bash
accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=umi_bimanual_policy_dunet \
  task.dataset.dataset_path=datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-bimanual.zarr \
  exp_name="example_data_bimanual" \
  group_tag="initial_experiments"
```

Supported `--config-name` values:
- [umi_bimanual_policy_dunet](../behavior_prompting/train_network/config/umi_bimanual_policy_dunet.yaml) (48GB) - no task conditioning
- [umi_bimanual_policy_dunet_language](../behavior_prompting/train_network/config/umi_bimanual_policy_dunet_language.yaml) (48GB) - language conditioning
- [umi_bimanual_policy_dunetp](../behavior_prompting/train_network/config/umi_bimanual_policy_dunetp.yaml) (48GB) - behavior prompting policy

### Modifiers
Rather than editing base configs which will modify all UMI runs, often you have a particular experiment you want to set parameters for. You can do this by appending a modifiers file containing configuration changes to the end of the training command. Here's an example: `+modifiers/umi/laundry_folding=task_dunetp`.

### Behavior Prompting runner
When training behavior prompting policies you often want to understand which parts of the prompt that the policy is paying attention to. We have a runner [umi_prompting_runner.py](../behavior_prompting/train_network/env_runner/umi_prompting_runner.py) that will run periodically when you are training behavior prompting policies to help with this. The runner replays training demonstrations and calls the policy throughout to visualize the prompt encoder attention throughout the demonstration.

## Deploy iPhUMI policy on robot
Follow the [robot deployment](robot_deployment.md) instructions.
