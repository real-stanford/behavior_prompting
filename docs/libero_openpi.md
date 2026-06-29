# $\pi_{0.5}$ LIBERO/LIBERO-Gen Evaluation

We extend the LIBERO support in [openpi](https://github.com/Physical-Intelligence/openpi) to also support LIBERO-Gen evaluation.

## Setup

### Code
Clone the `liberogen` branch on [my fork of openpi](https://github.com/austinapatel/openpi)
```bash
git clone --recurse-submodules git@github.com:austinapatel/openpi.git
# then make sure you are on liberogen branch

# Or if you already cloned the repo:
git submodule update --init --recursive
```

### Environment
If you are using a server with CUDA13, make sure to copy `uv_cuda13.toml` to `uv.toml` (in the openpi fork) before you run the uv environment setup instructions below. Effectively we just want to use `jax[cuda13]` instead of `jax[cuda12]` in that case.

```bash
cp uv_cuda13.toml uv.toml
```

Then install dependencies:
```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


If you plan to use the LIBERO evaluation scripts in openpi you will also need to setup their LIBERO environment. See `examples/libero/README.md` in the openpi repo for instructions. You only use this LIBERO environment when you are launching the evaluation script and use the main uv environment installed above for the other commands.

### Data
LIBERO-Gen matches the exact format as the original LIBERO dataset. However, openpi expects that the dataset has 256x256 image resolution in LeRobot format (rather than 128x128 in HDF5 format).

> [!WARNING]
> Unlike other scripts that do this similar LIBERO regeneration, it is important to not remove no-ops for LIBERO-Gen as some of the procedurally generated LIBERO-Gen data does not regenerate properly if no-ops are removed.

> [!NOTE]
> We show the coversion process here, but we also include already converted versions on Hugging Face (see further down) so you don't need to run these commands yourself.

To rerender the datasets at 256x256 still in HDF5 format (first make sure you have properly downloaded the original LIBERO and the new LIBERO-Gen datasets properly by following the instructions in [here](libero.md) and [here](liberogen.md)):
```bash
# start in behavior_prompting repo
cd behavior_prompting/train_network/scripts/libero
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python regenerate_libero_dataset.py \
  --libero_task_suites \
    libero_spatial_selected_combinations_view \
    libero_spatial_selected_combinations_inverse_view \
    libero_goal_chain_firststep_view \
    libero_goal_chain_secondstep_view \
    libero_goal_chain_selected_inverse_view \
    libero_goal_chain_selected_view \
    libero_10 libero_goal libero_object libero_spatial \
# you can select which specific splits are converted as desired
```

To convert the re-rendered datasets to LeRobot:
```bash
# Start in the openpi fork. This will push converted versions to your Huggingface account.
uv run examples/libero/convert_liberogen_data_to_lerobot.py --data_dir PATH/TO/tmp_libero_regenerated --vis --push_to_hub # --data_dir is the output of regenerate_libero_dataset.py

# if the above command fails to push to Hugging Face (for example for LIBERO-Gen Spatial Combination):
export HF_USERNAME=<INSERT HERE>
export HF_CACHE_LOCATION=<INSERT HERE> # path includes .cache/huggingface
rm -rf $HF_CACHE_LOCATION/lerobot/$HF_USERNAME/libero_gen_spatial_combination_train_openpi/.cache/huggingface
uv run huggingface-cli upload-large-folder \
  $HF_USERNAME/libero_gen_spatial_combination_train_openpi \
  $HF_CACHE_LOCATION/lerobot/$HF_USERNAME/libero_gen_spatial_combination_train_openpi \
  --repo-type dataset \
  --private \
  --num-workers 1
uv run huggingface-cli tag $HF_USERNAME/libero_gen_spatial_combination_train_openpi v2.1 --repo-type dataset -y
```

> [!WARNING]
> openpi expects a horizontal flip of the images in the dataset, so I match that in this LeRobot conversion. See the comment in `convert_liberogen_data_to_lerobot.py` in the openpi fork for additional details on this.

### Already converted LeRobot versions
We have converted versions of LIBERO and LIBERO-Gen to 256x256 LeRobot format on Hugging Face.

> [!NOTE]
> You do not need to pull these yourself. When you launch training they will automatically be pulled.

Training datasets:
- [libero_regenerated_openpi](https://huggingface.co/datasets/austinpatel/libero_regenerated_openpi) (openpi also provides a LeRobot version of the original LIBERO dataset, but we use our own conversion for our results)
- [libero_gen_goal_chain_train_openpi](https://huggingface.co/datasets/austinpatel/libero_gen_goal_chain_train_openpi)
- [libero_gen_goal_chain_no_secondstep_train_openpi](https://huggingface.co/datasets/austinpatel/libero_gen_goal_chain_no_secondstep_train_openpi)
- [libero_gen_spatial_combination_train_openpi](https://huggingface.co/datasets/austinpatel/libero_gen_spatial_combination_train_openpi)

Unseen datasets (unlike behavior prompting which prompts using demonstrations from the unseen tasks, openpi only needs a language description so you probably won't need to use these):
- [libero_gen_goal_chain_test_openpi](https://huggingface.co/datasets/austinpatel/libero_gen_goal_chain_test_openpi)
- [libero_gen_spatial_combination_test_openpi](https://huggingface.co/datasets/austinpatel/libero_gen_spatial_combination_test_openpi)

## Training

### Normalization statistics
We rewrote their norm stats script to make it much faster. This needs to be done before launching training.

```bash
# in openpi fork
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats_fast.py --config-name pi05_libero_lora_my_regeneration
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats_fast.py --config-name pi05_libero_gen_spatial_combination_lora
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats_fast.py --config-name pi05_libero_gen_goal_chain_lora
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats_fast.py --config-name pi05_libero_gen_goal_chain_no_secondstep_lora
```

### JAX compilation
By default JAX seems to store some compilation artifacts on `/home`, but if your `/home` is NFS networked you probably want to do this elsewhere. Put this in your `~/.bashrc`: 
```bash
export JAX_COMPILATION_CACHE_DIR=/local/.cache/jax # or equivalent for your system
```

### Launch training

We finetune $\pi_{0.5}$ with LoRA finetuning:
```bash
# in openpi fork
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_lora_my_regeneration \
  --exp-name=pi05_libero_my_regeneration_lora_seed0_v1 \
  --seed=0 --resume

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_gen_spatial_combination_lora \
  --exp-name=pi05_libero_gen_spatial_combination_lora_seed0_v1 \
  --seed=0 --resume

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_gen_goal_chain_lora \
  --exp-name=pi05_libero_gen_goal_chain_lora_seed0_v1 \
  --seed=0 --resume

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_gen_goal_chain_no_secondstep_lora \
  --exp-name=pi05_libero_gen_goal_chain_no_secondstep_lora_seed0_v1 \
  --seed=0 --resume
```

## Evaluation

### Launch policy server

Launch the appropriate server for the checkpoint you want to evaluate:
```bash
# in openpi fork
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --env LIBERO policy:checkpoint \
  --policy.config pi05_libero_lora_my_regeneration \
  --policy.dir checkpoints/pi05_libero_lora_my_regeneration/pi05_libero_my_regeneration_lora_seed0_v1/29999

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --env LIBERO policy:checkpoint \
  --policy.config pi05_libero_gen_spatial_combination_lora \
  --policy.dir checkpoints/pi05_libero_gen_spatial_combination_lora/pi05_libero_gen_spatial_combination_lora_seed0_v1/99999

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --env LIBERO policy:checkpoint \
  --policy.config pi05_libero_gen_goal_chain_lora \
  --policy.dir checkpoints/pi05_libero_gen_goal_chain_lora/pi05_libero_gen_goal_chain_lora_seed0_v1/99999

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --env LIBERO policy:checkpoint \
  --policy.config pi05_libero_gen_goal_chain_no_secondstep_lora \
  --policy.dir checkpoints/pi05_libero_gen_goal_chain_no_secondstep_lora/pi05_libero_gen_goal_chain_no_secondstep_lora_seed0_v2_no_horizontal_flip/99999
```

And then launch the corresponding LIBERO environment runner script:

> [!NOTE]
> We have updated the original `main.py` as `main_parallel.py` to support parallel evaluation which is dramatically faster.

```bash
# in openpi fork
# this is the only place where you use the special LIBERO environment detailed above in the setup instructions.
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl python examples/libero/main_parallel.py \
  --args.run_id=pi05_libero_my_regeneration_lora_seed0_v1_29999_no_horizontal_flip \
  --args.task-suite-names libero_goal libero_spatial libero_10 libero_object

MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl python examples/libero/main_parallel.py \
  --args.run_id=pi05_libero_gen_spatial_combination_lora_seed0_v1_99999 \
  --args.task-suite-names libero_spatial_selected_combinations_view libero_spatial_selected_combinations_inverse_view

MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl python examples/libero/main_parallel.py \
  --args.run_id=pi05_libero_gen_goal_chain_lora_seed0_v1_99999 \
  --args.task-suite-names libero_goal_chain_selected_view libero_goal_chain_selected_inverse_view

MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl python examples/libero/main_parallel.py \
  --args.run_id=pi05_libero_gen_goal_chain_no_secondstep_lora_seed0_v2_99999 \
  --args.task-suite-names libero_goal_chain_selected_view libero_goal_chain_selected_inverse_view
```
