# all of these run on 46gb of GPU memory

## BASELINES
# dunetp (ours)
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr exp_name="defaults" group_tag="prompt_defaults" training.seed=$SEED

# goal image baseline
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunet_goal_image task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr exp_name="defaults" group_tag="goal_image_defaults" training.seed=$SEED

# icrt baseline
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_icrt task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr exp_name="icrt_defaults" group_tag="icrt_defaults" training.seed=$SEED


## ABLATIONS
# task allocation
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_1000_25.zarr task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=500 task.dataset.num_training_demos_per_task=20 exp_name="20_demos_500_tasks" group_tag="task_allocation" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_1000_25.zarr task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=1000 task.dataset.num_training_demos_per_task=10 exp_name="10_demos_1000_tasks" group_tag="task_allocation" training.seed=$SEED

# chunk size
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.shape_meta.prompt_chunk_n_actions=50 exp_name="chunk50" group_tag="chunk_size" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.shape_meta.prompt_chunk_n_actions=30 exp_name="chunk30" group_tag="chunk_size" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.shape_meta.prompt_chunk_n_actions=20 exp_name="chunk20" group_tag="chunk_size" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.shape_meta.prompt_chunk_n_actions=5 exp_name="chunk5" group_tag="chunk_size" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.shape_meta.prompt_chunk_n_actions=2 exp_name="chunk2" group_tag="chunk_size" training.seed=$SEED

# number of tasks
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=1000 training.num_epochs=60 exp_name="1000_tasks" group_tag="numtasks" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=500 training.num_epochs=80 exp_name="500_tasks" group_tag="numtasks" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=250 training.num_epochs=80 exp_name="250_tasks" group_tag="numtasks" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr task.dataset.max_tasks=125 training.num_epochs=80 exp_name="125_tasks" group_tag="numtasks" training.seed=$SEED

# merge tokens
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.merge_prompt_tokens=none exp_name="merge_none" group_tag="merge_tokens" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.merge_prompt_tokens=obs exp_name="merge_obs" group_tag="merge_tokens" training.seed=$SEED

# number of parts
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10_parts1to3.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr exp_name="parts_1to3" group_tag="num_parts" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10_parts4to6.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr exp_name="parts_4to6" group_tag="num_parts" training.seed=$SEED

# prompt info
accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_action=true exp_name="obs+proprio" group_tag="prompt_info" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_obs=true exp_name="proprio+action" group_tag="prompt_info" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_proprio=true exp_name="obs+action" group_tag="prompt_info" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_obs=true model.obs_encoder.obs_encoder.ignore_prompt_proprio=true exp_name="action" group_tag="prompt_info" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_proprio=true model.obs_encoder.obs_encoder.ignore_prompt_action=true exp_name="obs" group_tag="prompt_info" training.seed=$SEED

accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=draw_policy_dunetp task.dataset.dataset_path=datasets/draw/procedural_2000_10.zarr task.dataset.num_training_demos_per_task=5 task.eval_dataset_path=datasets/draw/eval_handmade.zarr model.obs_encoder.obs_encoder.ignore_prompt_obs=true model.obs_encoder.obs_encoder.ignore_prompt_action=true exp_name="proprio" group_tag="prompt_info" training.seed=$SEED
