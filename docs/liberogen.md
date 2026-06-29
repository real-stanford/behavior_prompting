# LIBERO-Gen

> [!NOTE]
> Here we document information specific to LIBERO-Gen, but you should also refer to the main documentation in [libero.md](libero.md).

## Setup

### Dataset

Download the LIBERO-Gen HDF5 datasets from Hugging Face for [LIBERO-Gen Combination](https://huggingface.co/datasets/austinpatel/libero_gen_spatial_combination_hdf5) and [LIBERO-Gen Chain](https://huggingface.co/datasets/austinpatel/libero_gen_goal_chain_hdf5):
```bash
cd behavior_prompting/train_network/datasets/libero

# LIBERO-Gen Chain dataset
hf download austinpatel/libero_gen_goal_chain_hdf5 --repo-type=dataset --local-dir ./libero_gen_goal_chain_hdf5
mv libero_gen_goal_chain_hdf5/demonstration_data/* .

# LIBERO-Gen combination dataset
hf download austinpatel/libero_gen_spatial_combination_hdf5 --repo-type=dataset --local-dir ./libero_gen_spatial_combination_hdf5
mv libero_gen_spatial_combination_hdf5/demonstration_data/* .
```

The corresponding `init_files` and `bddl_files` are already included in this repo under `behavior_prompting/train_network/env/libero/`, so we don't copy them over. Note that these contain entries for some tasks that do not have corresponding HDF5 data, as not all tasks had demonstrations generated successfully. The HuggingFace datasets include only the `init_files` and `bddl_files` for tasks that have corresponding HDF5 data, while this repository contains all the ones created. The training and evaluation scripts will only look at the files that have corresponding HDF5 demonstration data generated, so it's fine that there are extra init and bddl files.

The resulting folder structure should look like:
```
datasets/libero
├── libero_goal_chain_selected_view
│   ├── open_the_top_drawer_and_then_put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5
│   └── ...
├── libero_goal_chain_selected_inverse_view
├── libero_goal_chain_firststep_view
├── libero_goal_chain_secondstep_view
├── libero_spatial_selected_combinations_view
└── libero_spatial_selected_combinations_inverse_view
```

## Generating New Data

The data generation pipeline has three sequential steps: generate task BDDL files and init states, automatically collect demonstrations using motion planning, then link the generated data into the datasets folder for training.

### 1. Generating Tasks
[gen_extra_libero_envs.py](../behavior_prompting/train_network/scripts/libero/gen_extra_libero_envs.py) reads a task spec YAML (in `gen_extra_libero_envs_configs/`) and generates BDDL task description files and simulation init states. For Goal Chain, it pairs first-step actions (e.g., open a drawer) with second-step goals (e.g., place an object on a location) to create two-step tasks. For Spatial Combination, it combines pick objects and placement locations from the LIBERO-Spatial domain to create novel pick-and-place tasks.
```bash
cd behavior_prompting/train_network/scripts/libero

# this generates the tasks BDDLs and init states for LIBERO-Gen Chain
env MUJOCO_GL=osmesa python gen_extra_libero_envs.py \
  --suffix staging \
  --splits libero_goal
```
`--suffix` tags the output directories (e.g., `staging` → writes to `bddl_files_staging/` and `init_states_staging/`) so you can stage and review before overwriting the live data.

### 2. Generating Demonstration Data
[generate_demonstrations.py](../behavior_prompting/train_network/scripts/libero/generate_demonstrations.py) runs motion planning to automatically collect demonstrations for tasks given their BDDL files. It runs in parallel across many workers and writes results to a timestamped run directory that contains the demonstration data:
```bash
cd behavior_prompting/train_network/scripts/libero

# this generates demonstration data for LIBERO-Gen Chain
python generate_demonstrations.py \
  --suffix staging \
   --include-splits \
   libero_goal_chain_firststep_view \
   libero_goal_chain_secondstep_view \
   libero_goal_chain_selected_view \
   libero_goal_chain_selected_inverse_view
```
Use `--status-only` along with a `--run-dir <previous run path>` to see the results of the generation.

### 3. Linking in Demonstration Data
[link_generated_demonstrations.py](../behavior_prompting/train_network/scripts/libero/link_generated_demonstrations.py) copies (or symlinks) the generated datasets from the run directory into the libero datasets folder so they are available for training. It also clears stale training and rollout caches. Clearing caches is quite important because the training and evluations scripts convert the HDF5 files to zarr datasets and then cache those results. If we change the HDF5 data we need to clear those caches such that they are recreated.
```bash
cd behavior_prompting/train_network/scripts/libero
python link_generated_demonstrations.py \
  --run-dirs PATH/TO/run_dir
```

## Task Similarity Analysis

[check_bddl_similarity_liberogen.py](../behavior_prompting/train_network/scripts/libero/check_bddl_similarity_liberogen.py) reports how similar the unseen/test tasks are to the training tasks. This is important for understanding what the evaluation suites are actually measuring — e.g., whether the policy has seen the same pick object or first step during training, just with different goals.

For goal chain tasks (checks unseen `libero_goal_chain_selected_view` against training splits):
```bash
cd behavior_prompting/train_network/scripts/libero
python check_bddl_similarity_liberogen.py --split liberogen_chain
```

For spatial combination tasks (checks unseen `libero_spatial_selected_combinations_view` against training splits):
```bash
cd behavior_prompting/train_network/scripts/libero
python check_bddl_similarity_liberogen.py --split liberogen_combination
```

Add `--simple` for compact output that shows counts only.

## Visualization

Visualize environments from BDDL files as images:
```bash
cd behavior_prompting/train_network/scripts/libero
env MUJOCO_GL=osmesa python vis_envs.py PATH/TO/bddl_folder --grid
```

Visualize init states:
```bash
cd behavior_prompting/train_network/scripts/libero
env MUJOCO_GL=osmesa python vis_init_states.py PATH/TO/bddl_folder
```

Visualize HDF5 demonstration data:
```bash
cd behavior_prompting/train_network/scripts/libero
python vis_hdf5.py PATH/TO/demo.hdf5 --generate-video
```

## Pretrained Checkpoints

### LIBERO-Gen Goal Chain
Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/liberogen_goal_chain).
```bash
hf download austinpatel/liberogen_goal_chain --repo-type=model --local-dir ./liberogen_goal_chain_checkpoints
```
- `liberogen_goal_chain_behavior_prompting.ckpt` — behavior prompting policy checkpoint
- `liberogen_goal_chain_goal_image.ckpt` — goal image conditioned policy checkpoint
- `liberogen_goal_chain_language.ckpt` — language conditioned policy checkpoint

### LIBERO-Gen Goal Chain (No Second Step)
An ablation variant that trains **without second-step demonstrations**. Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/liberogen_goal_chain_no_secondstep).
```bash
hf download austinpatel/liberogen_goal_chain_no_secondstep --repo-type=model --local-dir ./liberogen_goal_chain_no_secondstep_checkpoints
```
- `liberogen_goal_chain_no_secondstep_behavior_prompting.ckpt` — behavior prompting policy checkpoint
- `liberogen_goal_chain_no_secondstep_goal_image.ckpt` — goal image conditioned policy checkpoint
- `liberogen_goal_chain_no_secondstep_language.ckpt` — language conditioned policy checkpoint

### LIBERO-Gen Spatial Combination
Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/liberogen_spatial_combination).
```bash
hf download austinpatel/liberogen_spatial_combination --repo-type=model --local-dir ./liberogen_spatial_combination_checkpoints
```
- `liberogen_spatial_combination_behavior_prompting.ckpt` — behavior prompting policy checkpoint
- `liberogen_spatial_combination_goal_image.ckpt` — goal image conditioned policy checkpoint
- `liberogen_spatial_combination_language.ckpt` — language conditioned policy checkpoint

## Training

Language conditioned policy on LIBERO-Gen Goal Chain (24GB GPUs):
```bash
cd behavior_prompting/train_network/
env MUJOCO_GL=egl accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 train.py \
  --config-name=libero_policy_dunet_language \
  task=liberogen_goal_chain \
  +modifiers=libero/liberogen_goal_chain \
  exp_name="dunet_language_defaults" \
  group_tag="initial_experiments"
```

Supported values for `--config-name`:
- [libero_policy_dunetp](../behavior_prompting/train_network/config/libero_policy_dunetp.yaml) (46GB) - behavior prompting policy
- [libero_policy_dunet_language](../behavior_prompting/train_network/config/libero_policy_dunet_language.yaml) (24GB) - language conditioning
- [libero_policy_dunet_goal_image](../behavior_prompting/train_network/config/libero_policy_dunet_goal_image.yaml) (24GB) - goal image conditioning

Supported values for `task`:
- [liberogen_goal_chain](../behavior_prompting/train_network/config/task/liberogen_goal_chain.yaml) — also use `+modifiers=libero/liberogen_goal_chain`
- [liberogen_goal_chain_no_secondstep](../behavior_prompting/train_network/config/task/liberogen_goal_chain_no_secondstep.yaml) — also use `+modifiers=libero/liberogen_goal_chain` (ablation: excludes second-step demos from training)
- [liberogen_spatial_combination](../behavior_prompting/train_network/config/task/liberogen_spatial_combination.yaml) — also use `+modifiers=libero/liberogen_spatial_combination`

## Reproducing Results
To reproduce the results from the paper, use `launch_many_experiments.py` with the relevant experiment file:
```bash
cd behavior_prompting/train_network/
python launch_many_experiments.py \
  --runs-files experiments/libero/liberogen_goal_chain.sh \
  --gpus 0,1,2,3 \
  --seeds 0,1,2
```

Experiment files:
- [liberogen_spatial_combination.sh](../behavior_prompting/train_network/experiments/libero/liberogen_spatial_combination.sh) - spatial combination experiments
- [liberogen_goal_chain.sh](../behavior_prompting/train_network/experiments/libero/liberogen_goal_chain.sh) - goal chain experiments
- [liberogen_goal_chain_no_secondstep_ablation.sh](../behavior_prompting/train_network/experiments/libero/liberogen_goal_chain_no_secondstep_ablation.sh) - goal chain no second step experiments

### $\pi_{0.5}$ Evaluation
See [this doc](libero_openpi).