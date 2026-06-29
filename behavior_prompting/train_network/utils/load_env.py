import os
import time
from typing import Callable, List, Optional, Dict, Any
import hydra
from libero.libero import get_libero_path
import zarr
import random
import numpy as np
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset
from behavior_prompting.train_network.env.draw.draw_env import DrawEnv
from behavior_prompting.train_network.env_runner.base_runner import BaseRunner
from behavior_prompting.train_network.utils.dataset_util import prepare_only_task_names
from behavior_prompting.train_network.env_runner.draw_runner import get_draw_env
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.scripts.draw.demo_draw import collect_demos
from accelerate import Accelerator
import torch

from behavior_prompting.train_network.utils.libero_util import get_hdf5_files, hdf5_to_split, discover_and_register_benchmarks
discover_and_register_benchmarks()
from libero.libero.benchmark import get_benchmark_dict

def _distribute_tasks(tasks, process_index: int, num_processes: int, shuffle_tasks: bool=True):
    """Distribute tasks across processes for multi-GPU evaluation using round robin assignment."""
    tasks = tasks.copy()
    if shuffle_tasks:
        random.Random(0).shuffle(tasks) # if earlier tasks are easier than later tasks and max tasks is specified, then we might end up with a biased estimate, so we shuffle the tasks to randomize the order.

    if num_processes == 1:
        return tasks
    
    # Round robin assignment: task i goes to process (i % num_processes)
    distributed_tasks = [task for i, task in enumerate(tasks) if i % num_processes == process_index]
    
    # Print task distribution info
    if len(distributed_tasks) > 0:
        task_indices = [i for i in range(len(tasks)) if i % num_processes == process_index]
        print(f"[Process {process_index}] Assigned {len(distributed_tasks)} tasks: round robin indices {task_indices[:5]}{'...' if len(task_indices) > 5 else ''} (tasks: {distributed_tasks[:3]}{'...' if len(distributed_tasks) > 3 else ''})")
    else:
        print(f"[Process {process_index}] Assigned 0 tasks (no work for this process)")
    
    return distributed_tasks

def aggregate_metrics(runner_log: Dict[str, Any], accelerator: Optional[Accelerator]=None, group_additional_categories: Optional[List[str]]=None) -> Dict[str, Any]:
    """
    Aggregate metrics and handles case of multiple processes if using accelerate. Also computes additional matrics which are mean scores.
    
    Args:
        runner_log: Dictionary of metrics from current process
        accelerator: Accelerator instance for gathering metrics
    
    Returns:
        Aggregated metrics dictionary (only valid on main process)
    """
    if accelerator is None:
        merged_log = runner_log.copy()
    else:
        runner_log = runner_log.copy()
        runner_log['dummy_value'] = 0 # I found gather_for_metrics doesn't work if one of the dictionaries is empty which can happen if no task is assigned to a process, so we insert this dummy value
        # Multi-GPU case - gather all metrics first, then compute means
        all_runner_logs = accelerator.gather_for_metrics([runner_log])
        
        if not accelerator.is_main_process:
            return {}  # Non-main processes return empty dict

        merged_log = {}
        for log in all_runner_logs:
            for k, v in log.items():
                if k in merged_log:
                    raise ValueError(f"Metric '{k}' appears in multiple processes with values {v}. This indicates improper task distribution - each metric should only appear once across all processes.")
                elif k != 'dummy_value':
                    merged_log[k] = v

    new_metrics = _compute_mean_scores(merged_log, group_additional_categories)
    merged_log.update(new_metrics)
    return merged_log

def _compute_mean_scores(runner_log: Dict[str, Any], group_additional_categories: Optional[List[str]]=None) -> Dict[str, Any]:
    """
    Compute mean scores from individual task scores.
    This should only be called after all metrics are aggregated.
    """
    new_metrics = {}

    all_catogories = [None]
    if group_additional_categories is not None:
        all_catogories.extend(group_additional_categories)
    
    # Compute overall train mean score
    for category in all_catogories:
        for split in ["train", "test"]:
            split_mean_score = {
                k: v for k, v in runner_log.items() if f"{split}/" in k and "mean_score" in k and (category is None or f"/{category}/" in k)
            }
            if split_mean_score:
                mean_score = np.mean(list(split_mean_score.values()))
                stderr_score = np.std(list(split_mean_score.values())) / np.sqrt(len(split_mean_score))
                new_metrics[f"mean_scores/{split}/{'all' if category is None else category}"] = mean_score
                new_metrics[f"stderr_scores/{split}/{'all' if category is None else category}"] = stderr_score

    return new_metrics

def load_env_runner(cfg, output_dir, dataset: Optional[BaseDataset]=None, accelerator: Optional[Accelerator]=None):
    if "libero" in cfg.task.name:
        def interleave_hdf5_ordering(hdf5_files):
            # we want to interleave the hdf5 files so that we have at least one task from each of the splits if we limit the number of tasks
            file_by_split = {}
            for file in hdf5_files:
                split = hdf5_to_split(file)
                if split not in file_by_split:
                    file_by_split[split] = []
                file_by_split[split].append(file)

            output_files = []

            splits_order = sorted(file_by_split.keys())
            while len(output_files) < len(hdf5_files):
                for split in splits_order:
                    if len(file_by_split[split]) > 0:
                        output_files.append(file_by_split[split].pop(0))
            return output_files

        seen_env_runners = []
        unseen_env_runners = []
        if cfg.task.env_runner.eval_seen:
            hdf5_files = get_hdf5_files(get_libero_path('datasets'), cfg.task.env_runner.dataset_splits, cfg.task.env_runner.include_file_filters)
            hdf5_files = interleave_hdf5_ordering(hdf5_files)

            # Limit number of tasks if max_tasks is specified
            seen_max_tasks = cfg.task.train_rollout_max_tasks
            if seen_max_tasks is not None:
                hdf5_files = hdf5_files[:seen_max_tasks]

            # Distribute files across processes
            if accelerator is not None:
                hdf5_files = _distribute_tasks(hdf5_files, accelerator.process_index, accelerator.num_processes, shuffle_tasks=False)

            for file in hdf5_files:
                # configure env
                # instead of creating all the environments at once we create them on the fly so we can save memory. This means that instead of the total memory being determined by the number of tasks, it is determined by the number of accelerate processes.
                seen_env_runners.append(lambda dataset_path=file, output_dir=output_dir, **kwargs: hydra.utils.instantiate(
                    cfg.task.env_runner, dataset_path=dataset_path, output_dir=output_dir, is_seen=True, **kwargs
                ))
        
        if cfg.task.env_runner.eval_unseen:
            unseen_dataset_splits = cfg.task.env_runner.unseen_dataset_splits
            hdf5_files = get_hdf5_files(get_libero_path('datasets'), unseen_dataset_splits, cfg.task.env_runner.unseen_include_file_filters)
            hdf5_files = interleave_hdf5_ordering(hdf5_files)

            # Limit number of tasks if max_tasks is specified
            unseen_max_tasks = cfg.task.unseen_rollout_max_tasks
            if unseen_max_tasks is not None:
                hdf5_files = hdf5_files[:unseen_max_tasks]

            # Distribute files across processes
            if accelerator is not None:
                hdf5_files = _distribute_tasks(hdf5_files, accelerator.process_index, accelerator.num_processes, shuffle_tasks=False)

            for file in hdf5_files:
                # configure env
                # instead of creating all the environments at once we create them on the fly so we can save memory. This means that instead of the total memory being determined by the number of tasks, it is determined by the number of accelerate processes.
                unseen_env_runners.append(lambda dataset_path=file, output_dir=output_dir, **kwargs: hydra.utils.instantiate(
                    cfg.task.env_runner, dataset_path=dataset_path, output_dir=output_dir, is_seen=False, **kwargs
                ))
        
        env_runners = {
            'seen': seen_env_runners,
            'unseen': unseen_env_runners
        }

        return env_runners
    elif cfg.task.name == "draw":
        # for drawing task the number of env runners corresponds to the number of different tasks in the replay buffer. Thus we need to load the replay buffer and create an env runner for each task.
        # create a single set of drawing envs which is shared by all env runners (this is because all the envs are the same across letters and we can just update the target drawing for each task). This is more efficient than creating a new set of envs for each task.
        assert (cfg.task.dataset.dataset_path is not None or cfg.task.eval_dataset_path is not None) or cfg.task.live_demo, "either dataset path must be provided or live_demo must be true"
        assert not cfg.task.live_demo or (cfg.task.dataset.dataset_path is None and cfg.task.eval_dataset_path is None), "if live_demo set then dataset path and eval dataset path must both be None"

        if not cfg.task.live_demo:
            print (f"Creating drawing envs...")
            start_time = time.time()
            env = get_draw_env(**cfg.task.env_runner)
            end_time = time.time()
            print(f"Time taken to create drawing envs: {end_time - start_time} seconds")

        # create an env runner for each task
        env_runners = {
            'train': [],
            'eval': [],
            'output_dir': output_dir
        }
        if cfg.task.dataset.dataset_path is not None:
            if dataset is not None:
                replay_buffer = dataset.replay_buffer
            else:
                print (f"Loading train replay buffer for env runners from {cfg.task.dataset.dataset_path}...")
                start_time = time.time()
                replay_buffer = ReplayBuffer.create_from_path(cfg.task.dataset.dataset_path)
                end_time = time.time()
                print(f"Time taken to load train replay buffer in load_env_runner: {end_time - start_time} seconds")

            # compute only task names
            if cfg.task.train_rollout_only_task_names is not None:
                if cfg.task.dataset.only_task_names is not None:
                    assert set(cfg.task.train_rollout_only_task_names).issubset(set(cfg.task.dataset.only_task_names)), "train_rollout_only_task_names must be a subset of dataset.only_task_names"
                train_only_task_names = cfg.task.train_rollout_only_task_names
            else:
                train_only_task_names = cfg.task.dataset.only_task_names # could be None

            # compute max tasks
            if cfg.task.train_rollout_max_tasks is not None:
                if cfg.task.dataset.max_tasks is not None:
                    assert cfg.task.train_rollout_max_tasks <= cfg.task.dataset.max_tasks, "task.train_rollout_max_tasks must be less than or equal to task.dataset.max_tasks"
                max_tasks = cfg.task.train_rollout_max_tasks
            else:
                max_tasks = cfg.task.dataset.max_tasks # could be None
            
            unique_task_names = prepare_only_task_names(replay_buffer, train_only_task_names, max_tasks=max_tasks)
            
            # Distribute train tasks across processes
            if accelerator is not None:
                unique_task_names = _distribute_tasks(unique_task_names, accelerator.process_index, accelerator.num_processes)

            for task_name in unique_task_names:
                # configure env
                env_runner: BaseRunner
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner, output_dir=output_dir, env=env, replay_buffer=replay_buffer, task_name=task_name,  is_eval_dataset=False
                )
                assert isinstance(env_runner, BaseRunner)
                env_runners['train'].append(env_runner)
        else:
            unique_task_names = []

        if cfg.task.eval_dataset_path is not None:
            print (f"Loading eval replay buffer for env runners from {cfg.task.eval_dataset_path}...")
            start_time = time.time()
            eval_replay_buffer = ReplayBuffer.copy_from_path(cfg.task.eval_dataset_path, store=zarr.MemoryStore())
            end_time = time.time()
            print(f"Time taken to load eval replay buffer in load_env_runner: {end_time - start_time} seconds")

            unique_eval_task_names = prepare_only_task_names(eval_replay_buffer, cfg.task.eval_rollout_only_task_names, max_tasks=cfg.task.eval_rollout_max_tasks)
            for task_name in unique_eval_task_names:
                assert task_name not in unique_task_names, f"task {task_name} is in both train and eval datasets"

            # Distribute eval tasks across processes
            if accelerator is not None:
                unique_eval_task_names = _distribute_tasks(unique_eval_task_names, accelerator.process_index, accelerator.num_processes)

            eval_env_runner_cfg = cfg.task.env_runner.copy()
            eval_env_runner_cfg.n_test += eval_env_runner_cfg.n_train # all environments are test environments for eval datasets
            eval_env_runner_cfg.n_train = 0
            for task_name in unique_eval_task_names:
                env_runner: BaseRunner
                env_runner = hydra.utils.instantiate(
                    eval_env_runner_cfg,  output_dir=output_dir, env=env, replay_buffer=eval_replay_buffer, task_name=task_name, is_eval_dataset=True
                )
                assert isinstance(env_runner, BaseRunner)
                env_runners['eval'].append(env_runner)
        
        return env_runners
    elif cfg.task.name == "umi_prompt" or cfg.task.name == "umi_bimanual_prompt":
        # For UMI tasks, create env runners for each task in the dataset
        # Similar to draw, but using UmiPromptingRunner
        assert accelerator is None, "We currently don't support distributed evaluation for the UMI tasks, though it could be easily added later"
        
        # create an env runner for each task
        env_runners = {
            'train': [],
            'eval': []
        }

        if cfg.task.dataset.dataset_path is not None:
            if dataset is not None:
                replay_buffer = dataset.replay_buffer
            else:
                print (f"Loading train replay buffer for env runners from {cfg.task.dataset.dataset_path}...")
                start_time = time.time()
                replay_buffer = ReplayBuffer.create_from_path(cfg.task.dataset.dataset_path)
                end_time = time.time()
                print(f"Time taken to load train replay buffer in load_env_runner: {end_time - start_time} seconds")
            
            # compute only task names
            train_only_task_names = cfg.task.train_rollout_only_task_names
            
            unique_task_names = prepare_only_task_names(replay_buffer, only_task_names=train_only_task_names)
            # select train_rollout_max_tasks from the dataset at random indices
            random.seed(cfg.task.dataset.seed + 1)
            random.shuffle(unique_task_names)
            unique_task_names = unique_task_names[:cfg.task.train_rollout_max_tasks]
            
            # Create env runners for each task
            for task_name in unique_task_names:
                env_runner: BaseRunner
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner, 
                    output_dir=output_dir, 
                    replay_buffer=replay_buffer, 
                    task_name=task_name,
                    shape_meta=cfg.task.shape_meta,
                    obs_down_sample_steps=cfg.task.obs_down_sample_steps,
                    is_eval_dataset=False
                )
                assert isinstance(env_runner, BaseRunner)
                env_runners['train'].append(env_runner)
        
        if cfg.task.eval_dataset_path is not None:
            eval_replay_buffer = ReplayBuffer.create_from_path(cfg.task.eval_dataset_path)
            unique_eval_task_names = prepare_only_task_names(eval_replay_buffer, cfg.task.eval_rollout_only_task_names, max_tasks=cfg.task.eval_rollout_max_tasks)
            if cfg.task.dataset.dataset_path is not None:
                for task_name in unique_eval_task_names:
                    assert task_name not in unique_task_names, f"task {task_name} is in both train and eval datasets"

            for task_name in unique_eval_task_names:
                env_runner: BaseRunner
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner, 
                    output_dir=output_dir, 
                    replay_buffer=eval_replay_buffer, 
                    task_name=task_name,
                    shape_meta=cfg.task.shape_meta,
                    obs_down_sample_steps=cfg.task.obs_down_sample_steps,
                    is_eval_dataset=True
                )
                assert isinstance(env_runner, BaseRunner)
                env_runners['eval'].append(env_runner)
        
        return env_runners
    else:
        # For single env runner tasks only create on main process when using accelerator
        if accelerator is None or accelerator.is_main_process:
            # configure env
            env_runner: BaseRunner
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
            assert isinstance(env_runner, BaseRunner)
            return env_runner
        else:
            # Other processes return None since there's only one env runner
            return None


def env_rollout(cfg, env_runners, policy: Optional[BasePolicy]=None, enable_expensive_vis: bool=True, accelerator: Optional[Accelerator]=None, max_tasks: Optional[int]=None, init_runner_kwargs: Optional[Dict[str, Any]]=None):
    init_runner_kwargs = init_runner_kwargs.copy() if init_runner_kwargs is not None else {}
    step_log = {}

    def handle_runners(runners, aggregate_prefix="", need_to_init_runners: bool=False, group_additional_categories: Optional[List[str]]=None):
        if max_tasks is not None:
            if accelerator is None:
                runners_to_use = runners[:max_tasks]
            else:
                max_tasks_per_runner = max_tasks // accelerator.num_processes
                if accelerator.process_index < max_tasks % accelerator.num_processes:
                    max_tasks_per_runner += 1
                runners_to_use = runners[:max_tasks_per_runner]
        else:
            runners_to_use = runners

        runner_logs = {}
        for env_runner in runners_to_use:
            assert not policy.training, "policy must be in eval mode for env rollout"

            if need_to_init_runners:
                kwargs = init_runner_kwargs if init_runner_kwargs is not None else {}
                env_runner = env_runner(**kwargs)

            runner_log = env_runner.run(policy, enable_expensive_vis)

            if need_to_init_runners:
                # we assume that if you are initializing the runner on the fly then you want to close it after the run
                env_runner.close()

            runner_logs.update(runner_log)

        # Aggregate metrics across all processes and compute mean scores
        runner_logs = aggregate_metrics(runner_logs, accelerator, group_additional_categories)
        
        # we also want metrics for mean score specifically to be in the root key of the step log so it's easy to see and for checkpoining
        root_level_metrics = {}
        aggregate_prefix_underscore = aggregate_prefix + "_" if aggregate_prefix != "" else ""
        for split in ["train", "test"]:
            key = f"mean_scores/{split}/all"
            if key in runner_logs:
                root_level_metrics[f"{aggregate_prefix_underscore}{split}_mean_score"] = runner_logs[key]

        # add the aggregate prefix to the metrics
        runner_log_with_prefix = {**root_level_metrics}
        if aggregate_prefix != "":
            for metric, value in runner_logs.items():
                runner_log_with_prefix[f"{aggregate_prefix}/{metric}"] = value
        else:
            runner_log_with_prefix.update(runner_logs)

        step_log.update(runner_log_with_prefix)

    if "libero" in cfg.task.name: # in libero we have separate env runners for each task
        group_additional_categories = list(get_benchmark_dict().keys()) # group results by libero split
        # map the accelerator device index to the GPU device id for the libero renderer
        # note that often the cuda index doesn't match the render device index and this implementation is making the assumption that the graphics indices are some permutation of the cuda indices. so cuda 4,5,6,7 could map to 7,6,5,4 graphics devices and that would be fine, but if the associated graphics are instead 0,1,2,3 then we would be allocating libero environments on GPUs that we are not using to train which is problematic and would need additional logic here to handle. can you use EGL probe to find this mapping: https://github.com/StanfordVL/egl_probe
        if os.environ.get("MUJOCO_GL", None) == "egl":
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible_devices is not None and accelerator is not None:
                cuda_visible_devices = cuda_visible_devices.split(",")
                assert accelerator.num_processes == 1 or accelerator.device.index < len(cuda_visible_devices), f"accelerator device index {accelerator.device.index} is greater than the number of CUDA visible devices {len(cuda_visible_devices)}"
                render_gpu_device_id = int(cuda_visible_devices[accelerator.device.index]) if accelerator.num_processes > 1 else -1
                init_runner_kwargs.update({
                    'render_gpu_device_id': render_gpu_device_id
                })
                torch.cuda.empty_cache() # since the libero rendering will happen in separate processes we need to free up reserved (but unused) memory left over from the training stage from this process to ensure there is enough memory for the libero renderer.
                accelerator.wait_for_everyone() # wait for all processes to clear their cache since sometimes the cuda index doesn't match the render device index
        
        handle_runners(env_runners['seen'], need_to_init_runners=True, group_additional_categories=group_additional_categories)
        handle_runners(env_runners['unseen'], need_to_init_runners=True, aggregate_prefix='unseen', group_additional_categories=group_additional_categories)
    elif cfg.task.name == "draw":
        if cfg.task.live_demo:
            # TODO: move this functionality to a separate file
            # live demo mode requires user interaction and cannot work with multiple processes
            if accelerator is not None:
                assert accelerator.num_processes == 1, f"Live demo mode requires single process, but got {accelerator.num_processes} processes. Multi-GPU evaluation is not supported for live demo mode."
            
            # live demo is used to collect a prompt from the user and then run the policy conditioned on that prompt
            
            env_runner_cfg = cfg.task.env_runner.copy()
            env_runner_cfg.n_train = 0
            env_runner_cfg.n_test = 1

            draw_env = DrawEnv(
                boundary_angle=0,
                render_mode='human',
                clock_hz=10 # the policy rollout is limited to 10Hz steps through the environment
            )
            # create the runner environment from the collect_demos environment
            env = get_draw_env(**env_runner_cfg, draw_env=draw_env, use_async_vector_env=False)

            print('\nInstructions:\nIn draw mode:\n - "q" to quit program\n - "r" to retry the drawing\n - "d" when you are done drawing\nIn live demo mode:\n - "d" to stop execution early')

            seed = 0
            while True:
                seed += 1
                draw_env.seed(seed)
                draw_env.reset(no_rotation=True)

                # call function in demo_draw.py to collect a demo from the user
                output_dir = env_runners['output_dir']

                replay_buffer = collect_demos(output_dir, single_demo=True, env=draw_env, in_memory_replay_buffer=True)

                if replay_buffer is None:
                    env.close()
                    break

                # pick a boundary angle that is not near the center so it is more obvious that board rotation happens
                boundary_angle = np.random.uniform(np.pi/8, np.pi/4)
                boundary_angle *= np.random.choice([1, -1])
                draw_env.boundary_angle = boundary_angle
                draw_env.reset()

                # create a instance of the runner
                task_name = replay_buffer.task_names[0]
                env_runner: BaseRunner = hydra.utils.instantiate(
                    env_runner_cfg,  output_dir=output_dir, env=env, replay_buffer=replay_buffer, task_name=task_name, is_eval_dataset=True, test_start_seed=seed
                )
                
                # run the env runner
                draw_env.enable_keyboard_control = True
                draw_env.put_reward_in_title = True
                handle_runners([env_runner])
                draw_env.enable_keyboard_control = False
                draw_env.put_reward_in_title = False

                draw_env.boundary_angle = 0
                draw_env.set_target_drawing(None)
        else:
            handle_runners(env_runners['train'])
            handle_runners(env_runners['eval'], aggregate_prefix='eval')
    elif cfg.task.name == "umi_prompt" or cfg.task.name == "umi_bimanual_prompt":
        # UMI tasks have multiple env runners (one per task)
        if env_runners is not None:
            handle_runners(env_runners['train'])
            handle_runners(env_runners['eval'], aggregate_prefix='eval')
    else: # For tasks that have a single env runner
        # Only main process has the env_runner, other processes have None
        if env_runners is not None:
            env_runner = env_runners
            handle_runners([env_runner])
    
    return step_log
