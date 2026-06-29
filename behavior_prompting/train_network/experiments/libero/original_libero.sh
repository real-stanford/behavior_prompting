# all of these run within 46gb of GPU memory

# dunetp
env MUJOCO_GL=egl accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=libero_policy_dunetp task=libero_all +modifiers=libero/libero_all exp_name="dunetp_defaults" group_tag="libero_all" training.seed=$SEED

# dunet language
env MUJOCO_GL=egl accelerate launch --gpu_ids $GPUS --num_processes=$NUM_PROCESSES train.py --config-name=libero_policy_dunet_language task=libero_all +modifiers=libero/libero_all exp_name="dunet_language_defaults" group_tag="libero_all" training.seed=$SEED
