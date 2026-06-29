"""
This is an adaptation of the original regenerate_libero_dataset.py script that is from OpenVLA repo: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py

We have modified it to work with LIBERO-Gen datasets. One major problem with the original script is that it doesn't seed the environment properly. Seeding is neccessary as just setting the initial state is not sufficient to ensure an exact replay of the actions. This means that a number of the trajectories fail to replay successfully. This is not something we can get around for the original LIBERO datasets since the seeds are not provided, but we store in the seeds in the HDF5 for LIBERO-Gen data so that we can replay them exactly.
"""

"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out unsuccessful demonstrations. The reason for unsuccessful demonstrations is that for the original LIBERO data we do not know what seed was used to initialize the environment (using set init state with the initial state is actually NOT sufficient to get exactly the same initialization). For tasks where we do know the seed used during data generation (LIBERO-Gen), we will have a higher success rate, but it still won't be 100% since we add in some extra steps at the beginning of the episode to let the environment settle after setting the initial state, which can cause some demos that were originally successful to now fail (probably due to some very minor simulation instabilities).
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by 180 degrees because we observe that the environments return images that are upside down on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_regenerated

"""

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import h5py
import numpy as np
from behavior_prompting.train_network import fix_robosuite_log_permission_issue
fix_robosuite_log_permission_issue()
import robosuite.utils.transform_utils as T
import tqdm
from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
discover_and_register_benchmarks()
from libero.libero import benchmark

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from behavior_prompting.train_network.scripts.libero.vis_hdf5 import generate_video_from_hdf5

def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


IMAGE_RESOLUTION = 256


def _process_task_worker(args):
    """
    Worker function: processes all episodes for a single task and writes the output HDF5.
    Runs in a subprocess when using parallel workers.

    progress_queue: if provided, sends ("start", task_name, total), ("progress", task_name),
                    and ("done", task_name) messages so the main process can display per-task bars.
                    If None (single-worker mode), a local tqdm bar is used instead.

    Returns: (task_name, task_description, num_replays, num_success, metainfo_episodes)
    """
    task_id, libero_task_suite, libero_raw_data_dir, split_output_dir, max_episodes_to_replay, retain_failed, progress_queue = args

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_task_suite]()
    task = task_suite.get_task(task_id)

    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

    orig_data_path = os.path.join(libero_raw_data_dir, libero_task_suite, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]

    new_data_path = os.path.join(split_output_dir, f"{task.name}_demo.hdf5")
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    task_num_replays = 0
    task_num_success = 0
    metainfo_episodes = {}

    num_episodes = len(orig_data.keys())
    if max_episodes_to_replay is not None:
        num_episodes = min(num_episodes, max_episodes_to_replay)

    if progress_queue is not None:
        progress_queue.put(("start", task.name, num_episodes))
        episode_pbar = None
    else:
        episode_pbar = tqdm.tqdm(total=num_episodes, desc=task.name, unit="ep", leave=False)

    for i in range(num_episodes):
        # Get demo data
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        if episode_pbar is not None:
            episode_pbar.set_postfix_str(f"demo_{i}")

        # This ordering of seeding and resetting (although quite cumbersome) allows us to achieve a higher success rate of replaying the demos generated by LIBERO-Gen as it more closely follows the procedure used in `create_dataset.py`

        # Load seed if available (for consistent environment replay)
        try:
            seed = demo_data.attrs["seed"]
            env.seed(seed)
            env.reset()
        except (KeyError, ValueError):
            seed = None

        # Load seed if available (for consistent environment replay)
        try:
            seed = demo_data.attrs["seed"]
            env.seed(seed)
        except (KeyError, ValueError):
            seed = None

        # Reset environment, set initial state, and wait a few steps for environment to settle
        model_xml = env.sim.model.get_xml()
        env.env.reset_from_xml_string(model_xml)
        env.env.sim.reset()
        env.env.sim.set_state_from_flattened(orig_states[0])
        env.env.sim.forward()
        for _ in range(10):
            obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

        # Set up new data lists
        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []

        # Replay original demo actions in environment and record observations
        actions_iter = tqdm.tqdm(orig_actions, desc="  steps", unit="step", leave=False) if episode_pbar is not None else orig_actions
        for action in actions_iter:
            if states == []:
                # In the first timestep, since we're using the original initial state to initialize the environment,
                # copy the initial state (first state in episode) over from the original HDF5 to the new one
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                # For all other timesteps, get state from environment and record it
                states.append(env.sim.get_state().flatten())
                robot_states.append(
                    np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                )

            # Record original action (from demo)
            actions.append(action)

            # Record data returned by environment
            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(
                np.hstack(
                    (
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                    )
                )
            )
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

            # Execute demo action in environment
            obs, reward, done, info = env.step(action.tolist())

        # At end of episode, save replayed trajectories to new HDF5 files (only keep successes, unless --retain-failed)
        if done or retain_failed:
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            assert len(actions) == len(agentview_images)

            ep_data_grp = grp.create_group(f"demo_{i}")
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=np.stack(states))
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)
            if seed is not None:
                ep_data_grp.attrs["seed"] = seed

            if done:
                task_num_success += 1

        task_num_replays += 1
        metainfo_episodes[f"demo_{i}"] = {
            "success": bool(done),
            "initial_state": orig_states[0].tolist(),
        }

        if progress_queue is not None:
            progress_queue.put(("progress", task.name))
        else:
            episode_pbar.update(1)

    orig_data_file.close()
    new_data_file.close()

    if progress_queue is not None:
        progress_queue.put(("done", task.name))
    else:
        episode_pbar.close()

    return task.name, task_description, task_num_replays, task_num_success, metainfo_episodes


def _run_progress_listener(progress_queue, num_tasks, num_workers):
    """
    Runs in a thread in the main process. Consumes messages from progress_queue sent by worker
    processes and manages per-task tqdm episode bars.

    Bar positions:
      - positions 0, 1: reserved for the splits and tasks bars (managed outside this function)
      - positions 2 .. 2+num_workers-1: per-task episode bars (recycled as tasks complete)

    Messages from workers:
      ("start",    task_name, total_episodes) — task is starting, create its bar
      ("progress", task_name)                 — one episode done, tick its bar
      ("done",     task_name)                 — task finished, close its bar
    """
    # Pool of tqdm position slots for concurrent per-task bars
    available_positions = list(range(2, 2 + num_workers))
    bars = {}  # task_name -> (tqdm_bar, position)
    completed = 0

    while completed < num_tasks:
        msg = progress_queue.get()
        tag = msg[0]

        if tag == "start":
            _, task_name, total = msg
            pos = available_positions.pop(0) if available_positions else 2 + len(bars)
            bar = tqdm.tqdm(
                total=total,
                desc=task_name[:45],
                position=pos,
                leave=False,
                unit="ep",
                ncols=100,
            )
            bars[task_name] = (bar, pos)

        elif tag == "progress":
            _, task_name = msg
            if task_name in bars:
                bars[task_name][0].update(1)

        elif tag == "done":
            _, task_name = msg
            if task_name in bars:
                bar, pos = bars.pop(task_name)
                bar.close()
                available_positions.append(pos)
                available_positions.sort()
            completed += 1


def _write_summary(summary, libero_task_suite, summary_json_out_path):
    rates = [v["success_rate"] for v in summary.values()]
    summary_out = {
        libero_task_suite: {
            "tasks": summary,
            "aggregate": {
                "min_success_rate": float(np.min(rates)),
                "max_success_rate": float(np.max(rates)),
                "mean_success_rate": float(np.mean(rates)),
                "std_success_rate": float(np.std(rates)),
            },
        }
    }
    with open(summary_json_out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    return summary_out


def regenerate_libero_split(libero_task_suite, libero_raw_data_dir, libero_target_dir, max_tasks_per_split=None, max_episodes_to_replay=None, only_tasks=None, num_workers=1, vis=False, vis_single_episode=False, retain_failed=False):
    print(f"Regenerating {libero_task_suite} dataset!")

    split_output_dir = os.path.join(libero_target_dir, libero_task_suite + "_regenerated")
    os.makedirs(split_output_dir)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(split_output_dir, f"{libero_task_suite}_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite and collect tasks to process
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    if max_tasks_per_split is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, max_tasks_per_split)

    task_args = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if only_tasks is not None and task.name not in only_tasks:
            continue
        task_args.append((task_id, libero_task_suite, libero_raw_data_dir, split_output_dir, max_episodes_to_replay, retain_failed))

    summary = {}
    summary_json_out_path = os.path.join(split_output_dir, "summary.json")

    def _on_task_done(task_name, task_description, task_num_replays, task_num_success, metainfo_episodes, pbar):
        # Update metainfo JSON
        task_key = task_description.replace(" ", "_")
        metainfo_json_dict[task_key] = metainfo_episodes
        with open(metainfo_json_out_path, "w") as f:
            json.dump(metainfo_json_dict, f, indent=2)

        # Update summary JSON
        summary[task_name] = {
            "episodes_attempted": task_num_replays,
            "episodes_succeeded": task_num_success,
            "success_rate": task_num_success / task_num_replays if task_num_replays > 0 else 0.0,
        }
        _write_summary(summary, libero_task_suite, summary_json_out_path)

        pbar.set_postfix_str(f"{task_name}: {task_num_success}/{task_num_replays} ok")
        pbar.update(1)

        if vis:
            hdf5_path = os.path.join(split_output_dir, f"{task_name}_demo.hdf5")
            video_path = generate_video_from_hdf5(hdf5_path, single_episode=vis_single_episode)
            print(f"Saved visualization video to: {video_path}")

    with tqdm.tqdm(total=len(task_args), desc=libero_task_suite, unit="task", position=1, leave=True) as pbar:
        if num_workers == 1:
            # Single-worker: process tasks sequentially with a local episode bar per task
            for args in task_args:
                result = _process_task_worker(args + (None,))
                _on_task_done(*result, pbar)
        else:
            with Manager() as manager:
                progress_queue = manager.Queue()

                # Start listener thread to manage per-task episode bars
                listener = threading.Thread(
                    target=_run_progress_listener,
                    args=(progress_queue, len(task_args), num_workers),
                    daemon=True,
                )
                listener.start()

                task_args_with_queue = [args + (progress_queue,) for args in task_args]
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(_process_task_worker, args): args for args in task_args_with_queue}
                    for future in as_completed(futures):
                        result = future.result()
                        _on_task_done(*result, pbar)

                listener.join()

    print(f"Dataset regeneration complete! Saved new dataset at: {split_output_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    if summary:
        print(f"Saved summary JSON at: {summary_json_out_path}")
        summary_out = _write_summary(summary, libero_task_suite, summary_json_out_path)
        print(f"\n=== Summary: {libero_task_suite} ===")
        for task_name, stats in summary.items():
            print(f"  {task_name}: {stats['episodes_succeeded']}/{stats['episodes_attempted']} succeeded ({stats['success_rate']*100:.1f}%)")
        agg = summary_out[libero_task_suite]["aggregate"]
        print(f"  ---")
        print(f"  Success rate — min: {agg['min_success_rate']*100:.1f}%  max: {agg['max_success_rate']*100:.1f}%  mean: {agg['mean_success_rate']*100:.1f}%  std: {agg['std_success_rate']*100:.1f}%")


if __name__ == "__main__":
    # Parse command-line arguments
    _datasets_dir = get_libero_path("datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suites", type=str, nargs="+", default=None,
                        help="One or more LIBERO task suites to process. Example: libero_spatial libero_goal. "
                             "If not specified, all suites in --libero_raw_data_dir are processed.")
    parser.add_argument("--libero_raw_data_dir", type=str, default=_datasets_dir,
                        help=f"Path to directory containing raw HDF5 dataset. Defaults to {_datasets_dir}")
    parser.add_argument("--max_tasks_per_split", type=int, default=None,
                        help="Limit the number of HDF5 files processed per task suite (for debugging)")
    parser.add_argument("--max_episodes_to_replay", type=int, default=None,
                        help="Limit the number of episodes replayed per task (for debugging)")
    parser.add_argument("--libero_target_dir", type=str, default=None,
                        help="Path to regenerated dataset directory. Defaults to --libero_raw_data_dir with the last folder renamed to '_regenerated'")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directories without prompting")
    parser.add_argument("--only-tasks", type=str, nargs="+", default=None, metavar="TASK_NAME",
                        help="Only process these specific tasks (e.g. put_the_bowl_on_the_plate open_the_middle_drawer_of_the_cabinet). "
                             "Task names match HDF5 filenames without the _demo.hdf5 suffix.")
    parser.add_argument("--num_workers", type=int, default=20,
                        help="Number of parallel worker processes per split (default: 20)")
    parser.add_argument("--single-worker", action="store_true",
                        help="Disable multiprocessing and process tasks sequentially (useful for debugging)")
    parser.add_argument("--vis", action="store_true",
                        help="Generate a visualization video for each task after it completes")
    parser.add_argument("--vis-single-episode", action="store_true",
                        help="When --vis is set, only render the first episode per task")
    parser.add_argument("--retain-failed", action="store_true",
                        help="Save failed (done=False) episodes to the output HDF5 in addition to successes")
    args = parser.parse_args()

    _valid_task_suites = sorted(d for d in os.listdir(args.libero_raw_data_dir) if os.path.isdir(os.path.join(args.libero_raw_data_dir, d)))

    # Default to all suites if not specified
    if args.libero_task_suites is None:
        args.libero_task_suites = _valid_task_suites
        print(f"No --libero_task_suites specified. Running on all suites: {args.libero_task_suites}")
    else:
        for suite in args.libero_task_suites:
            if suite not in _valid_task_suites:
                parser.error(f"Invalid task suite '{suite}'. Valid options: {_valid_task_suites}")

    # Validate --only-tasks: each name must appear in at least one of the specified suites
    if args.only_tasks is not None:
        all_task_names = set()
        for suite in args.libero_task_suites:
            suite_dir = os.path.join(args.libero_raw_data_dir, suite)
            for fname in os.listdir(suite_dir):
                if fname.endswith("_demo.hdf5"):
                    all_task_names.add(fname[: -len("_demo.hdf5")])
        missing = [t for t in args.only_tasks if t not in all_task_names]
        if missing:
            parser.error(
                f"--only-tasks specified tasks not found in any of the given suites {args.libero_task_suites}: {missing}. "
                f"Available tasks: {sorted(all_task_names)}"
            )

    _default_target_dir = args.libero_raw_data_dir.rstrip("/") + "_regenerated"
    target_dir = args.libero_target_dir or _default_target_dir
    num_workers = 1 if args.single_worker else args.num_workers

    # Check all output dirs upfront, ask any overwrite questions, then delete — all before processing begins.
    for suite in args.libero_task_suites:
        split_output_dir = os.path.join(target_dir, suite + "_regenerated")
        if os.path.isdir(split_output_dir):
            if not args.overwrite:
                user_input = input(f"Target directory already exists at path: {split_output_dir}\nEnter 'y' to overwrite, or anything else to exit: ")
                if user_input != 'y':
                    exit()
            print(f"Deleting existing output directory: {split_output_dir}")
            shutil.rmtree(split_output_dir)

    # Process each split, with an outer progress bar
    for suite in tqdm.tqdm(args.libero_task_suites, desc="Splits", unit="split", position=0, leave=True):
        regenerate_libero_split(suite, args.libero_raw_data_dir, target_dir, args.max_tasks_per_split, args.max_episodes_to_replay, args.only_tasks, num_workers, args.vis, args.vis_single_episode, args.retain_failed)
