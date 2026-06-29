# DrawAnything-Real

## Setup
Single arm whiteboard drawing experiments. Dataset is on [Hugging Face](https://huggingface.co/datasets/austinpatel/iphumi_drawinganything_real).
```bash
hf download austinpatel/iphumi_drawinganything_real --repo-type=dataset --local-dir ./iphumi_drawinganything_real
```

Dataset contents:
- We have not uploaded the raw iPhUMI data for this experiment, but reach out if you need it
- `drawanything_real_replay_buffer.zarr.zip` — training dataset

## Pretrained Checkpoints

Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/iphumi_drawanything_real).
```bash
hf download austinpatel/iphumi_drawanything_real --repo-type=model --local-dir ./iphumi_drawanything_real_models
```
- `drawanything_real_behavior_prompting_policy.ckpt` — behavior prompting policy checkpoint
- `drawanything_real_goal_image_policy.ckpt` — goal image conditioned policy checkpoint

> [!WARNING]
> These are not in-the-wild data/checkpoints, so it's likely they will not work when deployed in your environment.

## Goal Image Setup
For goal image conditioned policies, [set_goal_image_whiteboard_task.py](../behavior_prompting/train_network/scripts/umi/set_goal_image_whiteboard_task.py) uses SAM (Segment Anything Model) to detect red reference dots in each demonstration and set the goal image frame index in the replay buffer:
```bash
cd behavior_prompting/train_network/scripts/umi
python set_goal_image_whiteboard_task.py -i PATH/TO/replay_buffer.zarr
```

## Unreleased Components

> [!NOTE]
> Two components used in our pipeline are not publicly released as they are built using older robot deployment scripts: the procedural demonstration generation scripts (which replay DrawAnything-Sim trajectories on the real robot to collect real training data) and the automatic drawing evaluation using Chamfer distance scoring. Reach out if you need either of these.

## Training
Modify the single arm training command from [this doc](iphumi.md).

Modifier configs: [modifiers/umi/whiteboard_drawing/](../behavior_prompting/train_network/config/modifiers/umi/whiteboard_drawing/)
- [task_goal_image.yaml](../behavior_prompting/train_network/config/modifiers/umi/whiteboard_drawing/task_goal_image.yaml) - goal image conditioning
- [task_dunetp.yaml](../behavior_prompting/train_network/config/modifiers/umi/whiteboard_drawing/task_dunetp.yaml) - behavior prompting policy
