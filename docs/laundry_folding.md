# Laundry Folding

Bimanual sweater folding experiments. We use the sweater from [this link (Sky Blue; 4-5T)](https://www.amazon.com/dp/B0FFZ718FP?ref=ppx_pop_dt_b_product_details&th=1&psc=1).

## Setup
Dataset is on [Hugging Face](https://huggingface.co/datasets/austinpatel/iphumi_bimanual_laundry_folding).
```bash
hf download austinpatel/iphumi_bimanual_laundry_folding --repo-type=dataset --local-dir ./iphumi_bimanual_laundry_folding_datasets
```

Dataset contents:
- `sweater_folding_raw_data.zip` — raw iPhUMI data for training
- `sweater_folding_replay_buffer.zarr.zip` — training dataset
- `sweater_folding_raw_data_evaluation_prompts.zip` — raw iPhUMI data for evaluation ("unseen" prompts of known tasks)
- `sweater_folding_replay_buffer_evaluation_prompts.zarr.zip` — evaluation dataset ("unseen" prompts of known tasks)

## Pretrained Checkpoints

Checkpoints are on [Hugging Face](https://huggingface.co/austinpatel/iphumi_bimanual_laundry_folding).
```bash
hf download austinpatel/iphumi_bimanual_laundry_folding --repo-type=model --local-dir ./iphumi_bimanual_laundry_folding_models
```
- `sweater_folding_behavior_prompting_policy.ckpt` — behavior prompting policy checkpoint
- `sweater_folding_language_policy.ckpt` — language conditioned policy checkpoint

> [!WARNING]
> These are not in-the-wild data/checkpoints, so it's likely they will not work when deployed in your environment.

> [!NOTE]
> If you would like to process the raw iPhUMI data yourself, use the `link_shared_iphumi_data.py` script from the iPHUMI repo (see the documentation in that repo for how to use this script). We do not include the depth information in the raw iPhUMI data for these tasks as there was a (now-resolved) bug in the depth data collection.

## Training
Modify the bimanual training command from [this doc](iphumi.md).

Modifier configs: [modifiers/umi/laundry_folding/](../behavior_prompting/train_network/config/modifiers/umi/laundry_folding/)
- [task_language.yaml](../behavior_prompting/train_network/config/modifiers/umi/laundry_folding/task_language.yaml) - language conditioning
- [task_dunetp.yaml](../behavior_prompting/train_network/config/modifiers/umi/laundry_folding/task_dunetp.yaml) - behavior prompting policy
