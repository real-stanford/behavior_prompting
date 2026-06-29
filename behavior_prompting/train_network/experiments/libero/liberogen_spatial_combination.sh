# all of these run within 46gb of GPU memory

# dunetp
env MUJOCO_GL=egl accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=libero_policy_dunetp task=liberogen_spatial_combination +modifiers=libero/liberogen_spatial_combination exp_name="dunetp_defaults" group_tag="liberogen_spatial_combination" training.seed=$SEED

# dunet language
env MUJOCO_GL=egl accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=libero_policy_dunet_language task=liberogen_spatial_combination +modifiers=libero/liberogen_spatial_combination exp_name="dunet_language_defaults" group_tag="liberogen_spatial_combination" training.seed=$SEED

# dunet goal image
env MUJOCO_GL=egl accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=libero_policy_dunet_goal_image task=liberogen_spatial_combination +modifiers=libero/liberogen_spatial_combination exp_name="dunet_goal_image_defaults" group_tag="liberogen_spatial_combination" training.seed=$SEED
