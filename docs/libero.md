# LIBERO

## Setup

### Dataset
Download:
```bash
cd deps/LIBERO
python benchmark_scripts/download_libero_datasets.py # can also specify --datasets to download specific datasets
```

Move datasets to `behavior_prompting/train_network/datasets/libero`. The format shold look like:
```
datasets/libero
├── libero_10
│   ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
│   └── ...
├── libero_90
├── libero_goal
├── libero_object
└── libero_spatial
```

When launching training and evaluation, our code will convert the `.hdf5` files into `.zarr` datasets that are saved in the `behavior_prompting/train_network/cache` folder.

### Collecting Demonstrations
[collect_demonstration.py](../behavior_prompting/train_network/scripts/libero/collect_demonstration.py) collects human demonstrations using a spacemouse and saves them as HDF5 files:
```bash
cd behavior_prompting/train_network/scripts/libero
python collect_demonstration.py \
  --bddl-file PATH/TO/task.bddl \
  --directory PATH/TO/output_dir \
  --num-demonstration 1 \
  --device spacemouse
```

### LIBERO Config
Create config file:
```bash
cd ~ && mkdir .libero
cd .libero && touch config.yaml
```

Setup `config.yaml` as follows (fill in `<PATH_TO>` with absolute base path):
```
assets: <PATH_TO>/behavior_prompting/deps/LIBERO/libero/libero/assets
bddl_files: <PATH_TO>/behavior_prompting/behavior_prompting/train_network/env/libero/bddl_files
benchmark_root: <PATH_TO>/behavior_prompting/deps/LIBERO/libero/libero
datasets: <PATH_TO>/behavior_prompting/behavior_prompting/train_network/datasets/libero
init_states: <PATH_TO>/behavior_prompting/behavior_prompting/train_network/env/libero/init_files
```

## Pretrained Checkpoints

Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/libero).
```bash
hf download austinpatel/libero --repo-type=model --local-dir ./libero_checkpoints
```
- `libero_behavior_prompting.ckpt` — behavior prompting policy checkpoint
- `libero_language.ckpt` — language conditioned policy checkpoint

## Training
During training we do periodic partial evaluation by rolling out the policy in the LIBERO simulation and then we do full rollout evaluation at the end of training.

It's worthwhile to look at the [libero_defaults](../behavior_prompting/train_network/config/task/libero_defaults.yaml) config to understand how we configure LIBERO experiments.

Language conditioned policy (24GB GPUs) on all 130 LIBERO tasks:
```bash
env MUJOCO_GL=egl accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=libero_policy_dunet_language \
  task=libero_all \
  +modifiers=libero/libero_all \
  exp_name="dunet_language_defaults" \
  group_tag="initial_experiments"
```

Supported values for `--config-name`:
- [libero_policy_dunet_language](../behavior_prompting/train_network/config/libero_policy_dunet_language.yaml) (24GB) - language conditioning
- [libero_policy_dunet_goal_image](../behavior_prompting/train_network/config/libero_policy_dunet_goal_image.yaml) (24GB) - goal image conditioning
- [libero_policy_dunetp](../behavior_prompting/train_network/config/libero_policy_dunetp.yaml) (46GB) - behavior prompting policy

Supported values for `task`:
- [libero_10](../behavior_prompting/train_network/config/task/libero_10.yaml) — also use `+modifiers=libero/libero_small`
- [libero_90](../behavior_prompting/train_network/config/task/libero_90.yaml) — also use `+modifiers=libero/libero_90`
- [libero_goal](../behavior_prompting/train_network/config/task/libero_goal.yaml) — also use `+modifiers=libero/libero_small`
- [libero_object](../behavior_prompting/train_network/config/task/libero_object.yaml) — also use `+modifiers=libero/libero_small`
- [libero_spatial](../behavior_prompting/train_network/config/task/libero_spatial.yaml) — also use `+modifiers=libero/libero_small`
- [libero_all](../behavior_prompting/train_network/config/task/libero_all.yaml) — also use `+modifiers=libero/libero_all` — train on all 130 LIBERO tasks

## Evaluation
Evaluation already happens during and at the end of the training process. However, you can still run evaluation manually if you would like given a pretrained checkpoint. Evaluation is distributed across each of the GPUs provided by assigning different tasks to different GPUs.

> [!TIP]
> This rollout code builds the config from the command line rather than pulling the config from the checkpoint. So if you trained with any additional config changes specified you need to make sure they are also added to this rollout command.

```bash
env MUJOCO_GL="egl" accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=libero_policy_dunet_language \
  task=libero_all \
  +modifiers=libero/libero_all \
  +rollout=libero \
  rollout.checkpoint_path=<path to .ckpt> \
  exp_name="dunet_language_defaults_rollout" \
  group_tag="initial_experiments"
```

## Reproducing Results
To reproduce the results from the paper, use `launch_many_experiments.py` with the experiment file:
```bash
cd behavior_prompting/train_network/
python launch_many_experiments.py --runs-files experiments/libero/original_libero.sh --gpus 0,1,2,3 --seeds 0,1,2
```

Paper experiment files:
- [original_libero.sh](../behavior_prompting/train_network/experiments/libero/original_libero.sh) - original LIBERO experiments

### $\pi_{0.5}$ Evaluation
See [this doc](libero_openpi).
