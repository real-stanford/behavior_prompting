# Helpers

## Models

All policies share the same `train.py` entry point and are selected via `--config-name`. The main variants are:

- `*_dunet` — base diffusion UNet policy with no task conditioning
- `*_dunet_goal_image` — goal image conditioned policy
- `*_dunet_language` — language conditioned policy
- `*_dunetp` — behavior prompting policy
- `*_icrt` — [ICRT](https://github.com/Max-Fu/icrt) baseline (in-context robot transformer)

In addition to the conditional UNet diffusion backbone (`conditional_unet1d.py`), a diffusion transformer backbone is also implemented at [model/diffusion/transformer_for_action_diffusion.py](../behavior_prompting/train_network/model/diffusion/transformer_for_action_diffusion.py) and can be swapped in by creating additional configs.

## Dataset Utilities

Scripts in [behavior_prompting/scripts/](../behavior_prompting/scripts/) for inspecting and manipulating replay buffer datasets:

**Inspect dataset:**
```bash
cd behavior_prompting/scripts

# Print summary of episodes, tasks, and shapes in a replay buffer
python print_replay_buffer.py PATH/TO/replay_buffer.zarr

# Generate a video grid of episodes from a replay buffer
python video_grid_from_replay_buffer.py -i PATH/TO/replay_buffer.zarr -o output.mp4

# Generate an image grid from a replay buffer
python image_grid_from_replay_buffer.py -i PATH/TO/replay_buffer.zarr -o output.png
```

**Manipulate dataset:**
```bash
cd behavior_prompting/scripts

# Concatenate multiple replay buffers into one
python concat_replay_buffers.py \
  -o PATH/TO/output.zarr \
  PATH/TO/buffer1.zarr PATH/TO/buffer2.zarr

# Convert a zarr directory store to a zip file (for upload/transfer)
python zip_zarr.py -i PATH/TO/dataset.zarr

# Unzip a zarr zip back to a directory store (required for training)
python unzip_zarr.py -i PATH/TO/dataset.zarr.zip
```

## Launching Many Experiments

[launch_many_experiments.py](../behavior_prompting/train_network/launch_many_experiments.py) runs a batch of training commands sequentially from a `.sh` experiment file. It uses Weights & Biases as a backend to manage run state and completex experiments, so it's able to skip runs that have already completed. This means it's safe to re-run after interruptions or run multiple instaces across servers.

```bash
cd behavior_prompting/train_network/
python launch_many_experiments.py \
  --runs-files experiments/draw_sim/baselines_and_ablations.sh \
  --gpus 0,1,2,3 \
  --seeds 0,1,2
```

Useful flags:
- `--runs-files` — one or more `.sh` experiment files (multiple files are treated as one merged list)
- `--gpus` — which GPUs to use (passed as `$GPUS` to each command)
- `--seeds` — repeat each run for each seed (passed as `$SEED`)
- `--status-only` — print which runs are done/pending without launching anything
- `--dry-run` — print the commands that would be run without executing them

## Attention Visualizations

During rollout, behavior prompting policies produce attention visualization videos showing which parts of the prompt the encoder is attending to at each timestep. These are saved to the run's `vis/` output directory and logged to Weights & Biases.

Attention visualization is enabled by default during eval but is somewhat expensive. Disable it with `rollout.enable_expensive_rollout_vis=false`.
