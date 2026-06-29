import glob
import os
import pickle
import cv2
import shutil
import numpy as np
from behavior_prompting.common.trajectory_util import vis_video_aligned_trajectories
from behavior_prompting.common.transform_util import pose_6d_to_4x4
from behavior_prompting.train_network.model.common.rotation_transformer import RotationTransformer
import imageio
from libero.libero import get_libero_path
from libero.libero import benchmark
import robomimic.utils.file_utils as FileUtils
from libero.libero.benchmark import Benchmark, register_benchmark, grab_language_from_filename, Task
from libero.libero.benchmark import BENCHMARK_MAPPING as LIBERO_BENCHMARK_MAPPING

class NewBenchmark(Benchmark):
    def __init__(self, split):
        super().__init__(task_order_index=0)
        self.name = split
        self.hdf5_files = get_hdf5_files(get_libero_path('datasets'), [split])
        self._make_benchmark()

    def _make_benchmark(self):
        task_names = [os.path.basename(x).replace('_demo.hdf5', '') for x in self.hdf5_files]
        tasks = {}
        for task in task_names:
            language = grab_language_from_filename(task + ".bddl")
            tasks[task] = Task(
                name=task,
                language=language,
                problem="Libero",
                problem_folder=self.name,
                bddl_file=f"{task}.bddl",
                init_states_file=f"{task}.pruned_init",
            )

        tasks = list(tasks.values())
        tasks.sort(key=lambda x: x.name)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

def _create_benchmark_class(split_name):
    """Factory function to create a benchmark class for a given split name."""
    class_name = split_name.upper()
    
    class BenchmarkClass(NewBenchmark):
        def __init__(self):
            super().__init__(split_name)
    
    BenchmarkClass.__name__ = class_name
    BenchmarkClass.__qualname__ = class_name
    return register_benchmark(BenchmarkClass)

# Dynamically discover and create benchmark classes for all splits in libero datasets folder
# that are not already registered in the libero benchmark module
def discover_and_register_benchmarks():
    """Discover split directories in libero datasets folder and create benchmark classes for them."""
    libero_datasets_dir = get_libero_path('datasets')
    
    # Get all existing benchmark names from libero (case-insensitive)
    # Note: We check LIBERO_BENCHMARK_MAPPING which was imported at module level,
    # before any of our benchmarks are registered
    existing_benchmark_names = {name.lower() for name in LIBERO_BENCHMARK_MAPPING.keys()}
    
    # Discover all directories in the libero datasets folder
    if not os.path.exists(libero_datasets_dir):
        return
    
    all_splits = [
        d for d in os.listdir(libero_datasets_dir)
        if os.path.isdir(os.path.join(libero_datasets_dir, d))
    ]
    
    # Filter out splits that are already registered in libero benchmark module
    new_splits = [
        split for split in all_splits
        if split.lower() not in existing_benchmark_names
    ]
    
    # Create and register benchmark classes for new splits
    for split_name in sorted(new_splits):
        class_name = split_name.upper()
        benchmark_class = _create_benchmark_class(split_name)
        globals()[class_name] = benchmark_class

def get_hdf5_files(dataset_path, dataset_splits, include_file_filters=None):
    dataset_paths = []
    for dataset_split in sorted(dataset_splits):
        assert os.path.exists(os.path.join(dataset_path, dataset_split)), f"dataset split {dataset_split} does not exist in {dataset_path}"
        cur_dataset_paths = glob.glob(os.path.join(dataset_path, dataset_split) + "/*.hdf5")

        if include_file_filters:
            cur_dataset_paths = [x for x in cur_dataset_paths if os.path.basename(x).replace('.hdf5', '') in include_file_filters or os.path.basename(x).replace('.hdf5', '').replace("_demo", "") in include_file_filters]

        cur_dataset_paths.sort() # sort the paths for consistent dataset ordering
        dataset_paths.extend(cur_dataset_paths)
    
    if include_file_filters:
        assert len(dataset_paths) == len(include_file_filters), f"number of dataset paths ({len(dataset_paths)}) does not match number of include file filters ({len(include_file_filters)})"

    return dataset_paths

def hdf5_to_split(hdf5_path):
    libero_split = os.path.basename(os.path.dirname(hdf5_path))
    return libero_split

def hdf5_to_bddl(hdf5_path):
    bddl_path = get_libero_path("bddl_files")
    libero_split_name = hdf5_to_split(hdf5_path)
    hdf5_name = os.path.basename(hdf5_path)
    bddl_name = hdf5_name.replace("_demo.hdf5", ".bddl")
    cur_bddl_path = os.path.join(bddl_path, libero_split_name, bddl_name)
    return cur_bddl_path

def bddl_to_hdf5(bddl_path):
    hdf5_path = get_libero_path("datasets")
    libero_split_name = os.path.basename(os.path.dirname(bddl_path))
    hdf5_name = os.path.basename(bddl_path).replace(".bddl", "_demo.hdf5")
    cur_hdf5_path = os.path.join(hdf5_path, libero_split_name, hdf5_name)
    return cur_hdf5_path

def hdf5_to_task(hdf5_path):
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    libero_split_name = hdf5_to_split(hdf5_path)
    bddl_path = hdf5_to_bddl(hdf5_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_split_name]()

    # find the corresponding task id
    task_id = None
    for cur_task_id, task in enumerate(task_suite.tasks):
        if task.bddl_file == os.path.basename(bddl_path):
            task_id = cur_task_id
            break
    assert task_id is not None, f"task id not found for bddl file: {env_meta['bddl_file']}"
    assert task.problem_folder == libero_split_name

    task = task_suite.get_task(task_id)
    return task

def vis_prompt(prompt, output_path):
    """Visualizes the RGB and trajectory from a prompt batch sampled from the LiberoReplayImageDataset with prompting enabled."""

    """Save RGB video streams"""
    tmp_output_dir = os.path.join(os.path.dirname(output_path), "tmp")
    os.makedirs(tmp_output_dir, exist_ok=True)
    agentview_rgb = prompt['obs']['agentview_rgb'] # 0 to 1
    eye_in_hand_rgb = prompt['obs']['eye_in_hand_rgb'] # 0 to 1
    # Convert the numpy videos to uint8 format (0-255)
    agentview_rgb = (agentview_rgb * 255).astype('uint8')
    eye_in_hand_rgb = (eye_in_hand_rgb * 255).astype('uint8')
    agentview_rgb = np.transpose(agentview_rgb, (0, 2, 3, 1)) # (T, H, W, C)
    eye_in_hand_rgb = np.transpose(eye_in_hand_rgb, (0, 2, 3, 1)) # (T, H, W, C)

    # Ensure both videos have the same number of frames
    assert agentview_rgb.shape[0] == eye_in_hand_rgb.shape[0], "Frame count mismatch between agentview and eye_in_hand"

    # Resize eye_in_hand_rgb to match agentview_rgb width if needed
    if eye_in_hand_rgb.shape[2] != agentview_rgb.shape[2]:
        new_width = agentview_rgb.shape[2]
        new_height = eye_in_hand_rgb.shape[1]
        eye_in_hand_rgb = np.array([
            cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            for frame in eye_in_hand_rgb
        ])

    # Stack agentview_rgb on top of eye_in_hand_rgb for each frame
    combined_frames = [np.vstack((agentview_frame, eye_in_hand_frame))
                       for agentview_frame, eye_in_hand_frame in zip(agentview_rgb, eye_in_hand_rgb)]
    combined_frames = np.array(combined_frames)

    # Define the output video file path and parameters
    combined_output_path = os.path.join(tmp_output_dir, "combined_rgb.mp4")
    height, width, _ = combined_frames[0].shape
    if 'action' in prompt:
        chunk_dim = prompt['action'].shape[1] # (T, chunk_dim, num_pred_steps, action_dim)
    else:
        chunk_dim = 1
    fps = 20 / chunk_dim
    video_writer = imageio.get_writer(combined_output_path, fps=fps, codec='libx264')

    # Write each frame to the video file
    for frame in combined_frames:
        video_writer.append_data(frame) # convert to RGB

    # Release the video writer
    video_writer.close()

    """Convert trajectory to 4x4 pose"""

    ee_pos = prompt['obs']['ee_pos'] # absolute position: (T, num_pred_steps [optional], 3)
    ee_ori = prompt['obs']['ee_ori'] # absolute orientation: (T, num_pred_steps [optional], 6) rot6d

    if len(ee_pos.shape) == 3: # remove the num_pred_steps horizon
        ee_pos = ee_pos[:, 0]
        ee_ori = ee_ori[:, 0]

    rotation_transformer_rot6d_to_axisangle = RotationTransformer("rotation_6d", "axis_angle")
    ee_ori = rotation_transformer_rot6d_to_axisangle.forward(ee_ori) # (T, 3) axis angle

    pose6d = np.concatenate((ee_pos, ee_ori), axis=1) # (T, 6)

    """Chunk trajectory into segments according to the different sequences"""
    if 'eos' in prompt['metadata']:
        eos = prompt['metadata']['eos']
    else:
        eos = np.zeros(len(ee_pos), dtype=bool)
        eos[-1] = True # assume the last frame is the end of the sequence

    end_segment_locations = list(np.where(eos)[0])

    if (len(eos) - 1) not in end_segment_locations:
        end_segment_locations.append(len(eos) - 1) # ensure the end of the prompt is considered as a separate segment

    trajectories = []
    for i in range(len(end_segment_locations)):
        if i == 0:
            start = 0
        else:
            start = end_segment_locations[i-1] + 1
        end = end_segment_locations[i]

        # pad the pose6d by repeating the first and last frames
        cur_pose6d = pose6d[start:end] # (T_segment, 6)
        pad_start = np.tile(cur_pose6d[0], (start, 1))
        pad_end = np.tile(cur_pose6d[-1], (len(ee_pos) - end, 1))
        cur_pose6d = np.concatenate((pad_start, cur_pose6d, pad_end), axis=0) # (T, 6)
        cur_pose4x4 = pose_6d_to_4x4(cur_pose6d) # (T, 4, 4)

        trajectories.append(cur_pose4x4)

    """Plot video and trajectory"""
    vis_video_aligned_trajectories(
        video_path=combined_output_path,
        video_out_path=output_path,
        poses=trajectories,
        axis_every_n_steps=1
    )
    print(f'Saved video to {output_path}')

    # Remove the temporary output directory and its contents
    shutil.rmtree(tmp_output_dir)

def pad_action(action, action_rep, desired_length):
    """
    Pads the input action array of shape (B, T, D=10) according to the action_rep parameter, up to desired_length.
    Args:
        action (np.ndarray): Input action array of shape (B, T, 10).
        action_rep (str): Either 'delta' or 'absolute'.
        desired_length (int): Desired length along the time dimension (T).
    Returns:
        np.ndarray: Padded action array of shape (B, desired_length, 10).
    """
    assert action.shape[-1] == 10, "Action dimension D must be 10."
    B, T, D = action.shape
    # Split action into position, rot6d, gripper
    pos = action[..., :3]        # (B, T, 3)
    rot6d = action[..., 3:9]     # (B, T, 6)
    gripper = action[..., 9:]    # (B, T, 1)

    pad_count = max(0, desired_length - T)
    if pad_count == 0:
        return action

    if action_rep == 'delta':
        pad_pos = np.zeros((B, pad_count, 3), dtype=action.dtype)
        pad_rot6d = np.tile(np.array([[1,0,0,0,1,0]], dtype=action.dtype), (B, pad_count, 1)) # identity rotation
    elif action_rep == 'absolute':
        pad_pos = np.tile(pos[:, -1:, :], (1, pad_count, 1))
        pad_rot6d = np.tile(rot6d[:, -1:, :], (1, pad_count, 1))
    else:
        raise ValueError(f"Unknown action_rep: {action_rep}")

    pad_gripper = np.tile(gripper[:, -1:, :], (1, pad_count, 1)) # all rotation representations use the same gripper action and we chose to just repeat the last gripper action

    pad_action = np.concatenate([pad_pos, pad_rot6d, pad_gripper], axis=-1)  # (B, pad_count, 10)
    padded_action = np.concatenate([action, pad_action], axis=1)  # (B, T+pad_count, 10)
    return padded_action

