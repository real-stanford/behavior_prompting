# Robot Deployment

Our repository leverages the code from [real-env](https://github.com/real-stanford/real-env). Follow the detailed policy deployment instructions from that repository.

We include detached policy inference nodes that will serve policies from our respository located [here](../behavior_prompting/deployment).

## Serving a Behavior Prompting Policy
When serving behavior prompting policies you will need access to a demonstration dataset where the policy will fetch the in-context prompts from.

```bash
cd behavior_prompting/deployment
python detached_inference_real_env.py \
  -i PATH/TO/behavior_prompting_policy.ckpt \
  --dataset-path PATH/TO/prompt_demos.zarr.zip \
  --eval-dataset
```

Then run the task on the robot. The task script is responsible for requesting a certain prompt by index. Here's an example:
```bash
python iphumi_arx5_bimanual_task.py <task_name> \
  project_name=<project_name> \
  task_name="<task name>" \
  prompt_index=0
```

## Serving a Language Policy

```bash
cd behavior_prompting/deployment
python detached_inference_real_env.py \
  -i PATH/TO/language_policy.ckpt
```

## Action Replay
[detached_inference_action_replay.py](../behavior_prompting/deployment/detached_inference_action_replay.py) is an alternative inference server that replays demonstrations open-loop from a UMI replay buffer rather than running a policy. It computes delta actions between consecutive demo frames, which are frame-invariant and can be replayed as long as the robot starts in the same initial configuration as the demo.
```bash
cd behavior_prompting/deployment
python detached_inference_action_replay.py \
  --dataset-path PATH/TO/replay_buffer.zarr.zip
```
