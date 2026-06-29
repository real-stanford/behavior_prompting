# DrawAnything-Sim

## Setup
Dataset is on [Hugging Face](https://huggingface.co/datasets/austinpatel/drawanything_sim).
```bash
# download datasets
hf download austinpatel/drawanything_sim --repo-type=dataset --local-dir ./drawanything_sim_datasets

# move zips into datasets/draw and unzip
mv ./drawanything_sim_datasets/*.zip behavior_prompting/train_network/datasets/draw/
python behavior_prompting/scripts/unzip_zarr.py -i behavior_prompting/train_network/datasets/draw
```

Dataset contents (procedural_X_Y.zarr.zip means `X` tasks and `Y` demos per task):
- `procedural_2000_10.zarr.zip` — training dataset
- `procedural_1000_25.zarr.zip` — training dataset (ablation)
- `procedural_2000_10_parts1to3.zarr.zip` — training dataset (ablation with only 1 to 3 segments in each drawing; low complexity drawings)
- `procedural_2000_10_parts4to6.zarr.zip` — training dataset (ablation with 4-6 segments in each drawing; high complexity drawings)
- `eval_handmade.zarr.zip` — evaluation dataset with 50 tasks at 5 demos per task (human drawn)

## Collecting Drawings

### Human Drawings
[demo_draw.py](../behavior_prompting/train_network/scripts/draw/demo_draw.py) opens an interactive window for collecting human drawing demonstrations:
```bash
cd behavior_prompting/train_network/scripts/draw
python demo_draw.py -o PATH/TO/output_dir
```

### Procedural Generation
[procedural_generate_drawings.py](../behavior_prompting/train_network/scripts/draw/procedural_generate_drawings.py) generates drawings procedurally by sampling random planar stroke sequences. Each task is a unique target drawing made of segments that vary in type (lines, curves, ovals, etc.), count, length, and position/orientation.
```bash
cd behavior_prompting/train_network/scripts/draw
python procedural_generate_drawings.py \
  -o PATH/TO/output_dir \
  --num-tasks 2000 \
  --demos-per-task 10 \
  --min-parts 1 \
  --max-parts 6
```

## Pretrained Checkpoints

Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/drawanything_sim).
```bash
hf download austinpatel/drawanything_sim --repo-type=model --local-dir ./drawanything_sim_models
```
- `drawanything_sim_behavior_prompting.ckpt` — behavior prompting policy checkpoint
- `drawanything_sim_goal_image.ckpt` — goal image conditioned policy checkpoint
- `drawanything_sim_icrt.ckpt` — ICRT policy checkpoint

## Training
Here's an example of how to train a behavior prompting policy:
```bash
cd behavior_prompting/train_network
accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=draw_policy_dunetp \
  task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr \
  task.dataset.num_training_demos_per_task=5 \
  task.eval_dataset_path=datasets/draw/eval_handmade.zarr \
  exp_name="defaults" \
  group_tag="prompt_defaults" \
  training.seed=$SEED
```

Supported `--config-name` values:
- [draw_policy_dunetp](../behavior_prompting/train_network/config/draw_policy_dunetp.yaml) (46GB) - behavior prompting policy
- [draw_policy_dunet_goal_image](../behavior_prompting/train_network/config/draw_policy_dunet_goal_image.yaml) (24GB) - goal image conditioned policy
- [draw_policy_icrt](../behavior_prompting/train_network/config/draw_policy_icrt.yaml) (24GB) - ICRT baseline

See [baselines_and_ablations.sh](../behavior_prompting/train_network/experiments/draw_sim/baselines_and_ablations.sh) for all baseline and ablation training commands.

## Evaluation
Evaluation runs automatically during and at the end of training. You can also run it manually on a pretrained checkpoint:

> [!TIP]
> This rollout code builds the config from the command line rather than pulling the config from the checkpoint. Make sure any additional config changes used during training are also included here.

```bash
cd behavior_prompting/train_network
CUDA_VISIBLE_DEVICES="0" python train.py --config-name=draw_policy_dunetp \
  +rollout=draw \
  rollout.checkpoint_path=PATH/TO/drawanything_sim_behavior_prompting.ckpt \
  task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr \
  task.dataset.num_training_demos_per_task=5 \
  task.eval_dataset_path=datasets/draw/eval_handmade.zarr
```

## Reproducing Results
To reproduce the results from the paper, use `launch_many_experiments.py` with the experiment file:
```bash
cd behavior_prompting/train_network/
python launch_many_experiments.py --runs-files experiments/draw_sim/baselines_and_ablations.sh --gpus 0,1,2,3 --seeds 0,1,2
```

Paper experiment files:
- [baselines_and_ablations.sh](../behavior_prompting/train_network/experiments/draw_sim/baselines_and_ablations.sh) - all baselines and ablations

## Live Demo

Opens an interactive window where you can hand draw a prompt with your mouse and the policy will attempt to draw the corresponding drawing at a different board orientation:
```bash
cd behavior_prompting/train_network
CUDA_VISIBLE_DEVICES="0" python train.py --config-name=draw_policy_dunetp \
  rollout.checkpoint_path=PATH/TO/drawanything_sim_behavior_prompting.ckpt \
  +rollout=draw \
  task.live_demo=true \
  rollout.enable_expensive_rollout_vis=false
```

