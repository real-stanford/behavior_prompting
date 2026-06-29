"""
For the new tasks generated with `gen_extra_libero_envs.py`, automatically generate demonstrations for each task via replay of parts from training data + scripted policy.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
import traceback
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import h5py
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import numpy as np
import cv2
from copy import deepcopy

from behavior_prompting.train_network import fix_robosuite_log_permission_issue
fix_robosuite_log_permission_issue()

from libero.libero import get_libero_path
from robosuite import load_controller_config


from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
import yaml

from collect_demonstration import gather_demonstrations_as_hdf5
from create_dataset import create_dataset
from behavior_prompting.train_network.scripts.libero.vis_hdf5 import generate_video_from_hdf5
from behavior_prompting.train_network.utils.libero_util import bddl_to_hdf5

def get_libero_path_with_suffix(path_type, suffix=None):
    """
    Get libero path with optional suffix appended.
    
    Args:
        path_type: Type of path ('bddl_files', 'datasets', 'init_files', etc.)
        suffix: Optional suffix to append to the base path (underscore will be added automatically)
    
    Returns:
        Path string with suffix appended if provided
    """
    base_path = get_libero_path(path_type)
    if suffix is not None:
        # Add underscore before suffix if it doesn't already start with one
        if not suffix.startswith('_'):
            suffix = '_' + suffix
        return base_path + suffix
    return base_path


def load_demonstrations_config(config_path):
    """
    Load demonstration generation configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing all configuration values with numpy arrays converted from lists
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert lists to numpy arrays where needed
    # Convert object_grasp_configs (no conversion needed, it's a dict)
    object_grasp_configs = config.get("object_grasp_configs", {})
    
    # Convert no_position_variation_combinations from lists to tuples
    no_position_variation_combinations = [
        tuple(pair) for pair in config.get("no_position_variation_combinations", [])
    ]
    
    # Convert place_location_local_offset offsets to numpy arrays
    place_location_local_offset = config.get("place_location_local_offset", [])
    for entry in place_location_local_offset:
        if "offset" in entry and isinstance(entry["offset"], list):
            entry["offset"] = np.array(entry["offset"])
    
    # Convert intermediate_quaternion to numpy array
    intermediate_quaternion = np.array(config.get("intermediate_quaternion", [1.0, 0.0, 0.0, 0.0]))
    
    # Convert intermediate_pose_overrides positions to numpy arrays
    # Support both single position (list of 3) and multiple positions (list of lists)
    intermediate_pose_overrides = config.get("intermediate_pose_overrides", [])
    for entry in intermediate_pose_overrides:
        if "pos" in entry and isinstance(entry["pos"], list):
            # Check if it's a list of lists (multiple positions) or single position
            if len(entry["pos"]) > 0 and isinstance(entry["pos"][0], list):
                # Multiple positions: convert each to numpy array
                entry["pos"] = [np.array(pos) for pos in entry["pos"]]
            else:
                # Single position: convert to numpy array
                entry["pos"] = np.array(entry["pos"])
    
    # Calculate derived values
    intermediate_rot_variation_rad = np.deg2rad(config.get("intermediate_rot_variation_degrees", 10.0))
    object_fall_angle_threshold_rad = np.deg2rad(config.get("object_fall_angle_threshold_degrees", 20.0))
    
    return {
        "object_grasp_configs": object_grasp_configs,
        "no_position_variation_combinations": no_position_variation_combinations,
        "default_above_place_location_offset": config.get("default_above_place_location_offset", 0),
        "default_release_depth_offset": config.get("default_release_depth_offset", -0.1),
        "above_place_location_offsets_world": config.get("above_place_location_offsets_world", {}),
        "place_location_local_offset": place_location_local_offset,
        "release_depth_place_location_offsets_world": config.get("release_depth_place_location_offsets_world", {}),
        "release_depth_grasped_object_offsets_world": config.get("release_depth_grasped_object_offsets_world", {}),
        "ignore_predicate_check_during_place_movement_for_location": config.get(
            "ignore_predicate_check_during_place_movement_for_location", []
        ),
        "ignore_predicate_check_during_set_down_for_location": config.get(
            "ignore_predicate_check_during_set_down_for_location", []
        ),
        "intermediate_quaternion": intermediate_quaternion,
        "intermediate_pose_overrides": intermediate_pose_overrides,
        "intermediate_pos_variation_range": config.get("intermediate_pos_variation_range", 0.05),
        "intermediate_rot_variation_degrees": config.get("intermediate_rot_variation_degrees", 10.0),
        "intermediate_rot_variation_rad": intermediate_rot_variation_rad,
        "intermediate_pose_z_offset": config.get("intermediate_pose_z_offset", 0.1),
        "max_intermediate_pose_height": config.get("max_intermediate_pose_height", {}),
        "max_height": config.get("max_height", {}),
        "pregrasp_offset": config.get("pregrasp_offset", 0.05),
        "target_pos_variation_range": config.get("target_pos_variation_range", 0.03),
        "target_pos_variation_range_overrides": config.get("target_pos_variation_range_overrides", {}),
        "rot_gain_position_scaling": config.get("rot_gain_position_scaling", None),
        "object_fall_angle_threshold_degrees": config.get("object_fall_angle_threshold_degrees", 20.0),
        "object_fall_angle_threshold_rad": object_fall_angle_threshold_rad,
        "additional_replay_steps_after_stage": config.get("additional_replay_steps_after_stage", []),
        "check_place_location_fall_instead_of_placed_object": config.get("check_place_location_fall_instead_of_placed_object", []),
        "skip_fall_check_when_placed_on": config.get("skip_fall_check_when_placed_on", {}),
        "grasp_position_precision_override": config.get("grasp_position_precision_override", {}),
        "pregrasp_additional_z_offset_overrides": config.get("pregrasp_additional_z_offset_overrides", []),
    }

def load_split_metadata(split_name, bddl_path=None, suffix=None):
    """
    Load split metadata from task_metadata.yaml.
    
    Args:
        split_name: Name of the split (e.g., 'libero_goal_extra' or 'libero_goal')
        bddl_path: Optional path to bddl_files directory. If provided, uses this path instead of suffix.
        suffix: Optional suffix to append to bddl_files base path. Used only if bddl_path is None.
    """
    if bddl_path is None:
        bddl_path = get_libero_path_with_suffix("bddl_files", suffix)
    split_metadata_path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
    with open(split_metadata_path, "r") as f:
        split_metadata = yaml.safe_load(f)
    return split_metadata

def load_metadata_for_task(bddl_file_path, bddl_path=None, suffix=None):
    # Load the metadata for the task
    split_name = os.path.basename(os.path.dirname(bddl_file_path))
    # Use bddl_path if provided, otherwise fall back to suffix
    split_metadata = load_split_metadata(split_name, bddl_path=bddl_path, suffix=suffix)

    bddl_file_name = os.path.basename(bddl_file_path).replace(".bddl", "")
    base_split_name = split_metadata["base_split_name"]
    tasks_dict = split_metadata["tasks"]
    task_metadata = tasks_dict[bddl_file_name]
    
    # Load metadata from original split for reference tasks (existing tasks)
    # Use bddl_path if provided, otherwise fall back to suffix
    original_split_metadata = load_split_metadata(base_split_name, bddl_path=bddl_path, suffix=suffix)
    original_tasks_dict = original_split_metadata["tasks"]
    
    # Combine metadata: original split (existing tasks) + current split (new tasks)
    all_metadata = {**original_tasks_dict, **tasks_dict}
    
    task_metadata['all_metadata'] = all_metadata
    task_metadata['base_split_name'] = base_split_name

    return task_metadata

def task_name_to_bddl_path(task_metadata, task_name, bddl_path=None, suffix=None):
    """
    Get BDDL path for a task. This is used for reference tasks (existing tasks).
    
    Args:
        task_metadata: Task metadata dictionary
        task_name: Name of the task
        bddl_path: Optional path to bddl_files directory. If provided, uses this path instead of suffix.
        suffix: Optional suffix to append to bddl_files base path. Used only if bddl_path is None.
    """
    base_split_name = task_metadata["base_split_name"]
    if bddl_path is None:
        bddl_path = get_libero_path_with_suffix("bddl_files", suffix)
    return os.path.join(bddl_path, base_split_name, task_name + ".bddl")

def task_name_to_hdf5_path(task_metadata, task_name, suffix=None):
    """
    Get HDF5 path for a task. This is used for reference tasks (existing tasks),
    so we should NOT use the suffix.
    """
    base_split_name = task_metadata["base_split_name"]
    # Reference tasks are in base split, so don't use suffix
    return os.path.join(get_libero_path('datasets'), base_split_name, task_name + "_demo.hdf5")

def get_environment(bddl_file_path, use_camera_obs=False, resolution=128, suppress_print=False):
    # Get controller config
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configuration
    config = {
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }

    assert os.path.exists(bddl_file_path)
    problem_info = BDDLUtils.get_problem_info(bddl_file_path)

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]

    if not suppress_print:
        print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file_path,
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,  # Set to True so environment continues recording even when done=True (e.g., horizon reached)
        use_camera_obs=use_camera_obs,
        reward_shaping=True,
        control_freq=20,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=resolution,
        camera_widths=resolution,
    )

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    return env, env_info, problem_info


def stage_to_predicate(stage):
    predicate = [x.lower() for x in stage.split(" ")]
    return predicate

def get_base_object_name(obj_name: str) -> str:
    """
    Extract the base object name by removing numeric instance suffix (_1, _2, etc.)
    and _resized suffix.
    Handles cases where the suffix appears at the end or in the middle of the name.
    
    Examples:
        akita_black_bowl_1 -> akita_black_bowl
        plate_2 -> plate
        flat_stove_1 -> flat_stove
        flat_stove_1_cook_region -> flat_stove_cook_region
        wine_rack_1_top_region -> wine_rack_top_region
        wooden_cabinet_1_middle_region -> wooden_cabinet_middle_region
        wooden_cabinet_1_top_side -> wooden_cabinet_top_side
        main_table_stove_front_region -> main_table_stove_front_region (no change if no numeric suffix)
        main_table_table_center_resized -> main_table_table_center
        main_table_table_center_1_resized -> main_table_table_center
    """
    # First strip _resized suffix if present
    if obj_name.endswith('_resized'):
        obj_name = obj_name[:-8]  # Remove '_resized' (8 characters)
    
    # Split by underscores and filter out parts that are purely numeric (instance numbers)
    parts = obj_name.split('_')
    filtered_parts = []
    for part in parts:
        # Skip parts that are purely numeric (these are instance numbers like "1", "2")
        if not part.isdigit():
            filtered_parts.append(part)
    
    # Rejoin with underscores
    if filtered_parts:
        return '_'.join(filtered_parts)
    
    # Edge case: if all parts were numeric, return original (shouldn't happen in practice)
    return obj_name




def get_robot_poses_at_current_state(env, obs: Optional[dict]=None):
    # Get observation to get actual current pose
    if obs is None:
        obs = env._get_observations()
    robot = env.robots[0]
    robot_prefix = robot.robot_model.naming_prefix
    eef_pos = obs[f"{robot_prefix}eef_pos"]
    eef_quat = obs[f"{robot_prefix}eef_quat"]
    pose = np.concatenate([eef_pos, eef_quat])
    return pose

def get_object_pose(env, object_name):
    """
    Get the pose (position and quaternion) of an object.
    Handles both regular objects and site objects.
    
    Args:
        env: The environment
        object_name: Name of the object
        
    Returns:
        pose: numpy array of shape (7,) containing [pos (3,), quat (4,)]
    """
    # Check if it's a site object
    if object_name in env.object_sites_dict:
        object_pos = env.sim.data.get_site_xpos(object_name)
        object_xmat = env.sim.data.get_site_xmat(object_name)
        # Convert rotation matrix to quaternion (xyzw format)
        object_quat = Rotation.from_matrix(object_xmat).as_quat(scalar_first=False)
    else:
        # Regular object
        object_pos = env.sim.data.body_xpos[env.obj_body_id[object_name]]
        object_quat = env.sim.data.body_xquat[env.obj_body_id[object_name]]
        # Convert from wxyz to xyzw format (robosuite convention)
        object_quat = np.array([object_quat[1], object_quat[2], object_quat[3], object_quat[0]])
    
    pose = np.concatenate([object_pos, object_quat])
    return pose

def get_object_height(env, object_name, return_bbox=False):
    """
    Get the height of an object and optionally return bounding box information.
    
    Args:
        env: The environment
        object_name: Name of the object
        return_bbox: If True, also return a dictionary with bounding box info (min_z, max_z)
        
    Returns:
        height: float, the height of the object (z-dimension of bounding box)
        bbox_info (optional): dict with keys 'min_z' and 'max_z' if return_bbox=True
    """
    # Handle the case where the environment is wrapped
    if hasattr(env, 'env'):
        env = env.env
    
    # Get the object
    obj = env.get_object(object_name)
    
    # Try to get height from object's geometry
    # Most objects in robosuite have a root body with geometry
    body_id = env.obj_body_id[object_name]
    
    # Get all geoms associated with this body
    geom_ids = []
    for geom_id in range(env.sim.model.ngeom):
        if env.sim.model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
    
    assert len(geom_ids) > 0, f"No geometry found for object {object_name}"
    # Compute bounding box from all geoms
    min_z = float('inf')
    max_z = float('-inf')
    
    for geom_id in geom_ids:
        geom_size = env.sim.model.geom_size[geom_id]
        # Get geom position (relative to body)
        geom_pos = env.sim.model.geom_pos[geom_id]
        
        # For box/capsule/cylinder, size[2] is typically the half-height
        # We need to account for the geom position and size
        if len(geom_size) >= 3:
            geom_min_z = geom_pos[2] - geom_size[2]
            geom_max_z = geom_pos[2] + geom_size[2]
            min_z = min(min_z, geom_min_z)
            max_z = max(max_z, geom_max_z)
    
    assert max_z != float('-inf'), f"No geometry found for object {object_name}"
    
    height = max_z - min_z
    
    if return_bbox:
        return height, {'min_z': min_z, 'max_z': max_z}
    else:
        return height

def get_object_min_z(env, object_name):
    """
    Get the minimum z-coordinate of the object's geometry relative to its origin (body position).
    This represents the bottom of the object's bounding box.
    
    Args:
        env: The environment
        object_name: Name of the object
        
    Returns:
        min_z: float, the minimum z-coordinate of the object's geometry relative to its origin
               (negative means geometry extends below origin, positive means all geometry is above origin)
    """
    _, bbox_info = get_object_height(env, object_name, return_bbox=True)
    return bbox_info['min_z']

def compute_relative_pose(robot_pose, object_pose):
    """
    Compute the robot pose relative to the object pose.
    
    Args:
        robot_pose: numpy array of shape (7,) containing [pos (3,), quat (4,)] in xyzw format
        object_pose: numpy array of shape (7,) containing [pos (3,), quat (4,)] in xyzw format
        
    Returns:
        relative_pose: numpy array of shape (7,) containing relative pose [pos (3,), quat (4,)] in xyzw format
    """
    robot_pos = robot_pose[:3]
    robot_quat = robot_pose[3:7]
    object_pos = object_pose[:3]
    object_quat = object_pose[3:7]
    
    # Convert quaternions to rotation matrices
    robot_rot_mat = Rotation.from_quat(robot_quat, scalar_first=False).as_matrix()
    object_rot_mat = Rotation.from_quat(object_quat, scalar_first=False).as_matrix()
    
    # Compute relative position: transform robot position to object frame
    rel_pos = object_rot_mat.T @ (robot_pos - object_pos)
    
    # Compute relative rotation: object_rot_mat.T @ robot_rot_mat
    rel_rot_mat = object_rot_mat.T @ robot_rot_mat
    
    # Convert back to quaternion
    rel_quat = Rotation.from_matrix(rel_rot_mat).as_quat(scalar_first=False)
    
    relative_pose = np.concatenate([rel_pos, rel_quat])
    return relative_pose

def apply_relative_pose_to_object(relative_pose, object_pose, current_robot_pose=None):
    """
    Apply a relative pose to an object pose to get the absolute robot pose.
    
    Args:
        relative_pose: numpy array of shape (7,) containing relative pose [pos (3,), quat (4,)] in xyzw format.
                       If quat contains None values, only position will be used and current_robot_pose orientation will be used.
        object_pose: numpy array of shape (7,) containing object pose [pos (3,), quat (4,)] in xyzw format
        current_robot_pose: Optional numpy array of shape (7,) containing current robot pose [pos (3,), quat (4,)] in xyzw format.
                          Used when relative_pose orientation is None to preserve current gripper orientation.
        
    Returns:
        robot_pose: numpy array of shape (7,) containing absolute robot pose [pos (3,), quat (4,)] in xyzw format
    """
    rel_pos = relative_pose[:3]
    rel_quat = relative_pose[3:7]
    object_pos = object_pose[:3]
    object_quat = object_pose[3:7]
    
    # Check if orientation is missing (contains NaN)
    orientation_missing = np.isnan(rel_quat[0])
    
    if orientation_missing:
        # Only use position, keep current gripper orientation
        if current_robot_pose is None:
            raise ValueError("current_robot_pose must be provided when relative_pose orientation is missing")
        
        # Convert object quaternion to rotation matrix for position transformation
        object_rot_mat = Rotation.from_quat(object_quat, scalar_first=False).as_matrix()
        
        # Compute absolute position: transform relative position to world frame
        robot_pos = object_pos + object_rot_mat @ rel_pos
        
        # Use current robot orientation
        robot_quat = current_robot_pose[3:7]
    else:
        # Convert quaternions to rotation matrices
        rel_rot_mat = Rotation.from_quat(rel_quat, scalar_first=False).as_matrix()
        object_rot_mat = Rotation.from_quat(object_quat, scalar_first=False).as_matrix()
        
        # Compute absolute position: transform relative position to world frame
        robot_pos = object_pos + object_rot_mat @ rel_pos
        
        # Compute absolute rotation: object_rot_mat @ rel_rot_mat
        robot_rot_mat = object_rot_mat @ rel_rot_mat
        
        # Convert back to quaternion
        robot_quat = Rotation.from_matrix(robot_rot_mat).as_quat(scalar_first=False)
    
    robot_pose = np.concatenate([robot_pos, robot_quat])
    return robot_pose

def check_object_grasped(env, object_name):
    """Standalone function for check_predicate_success - uses standard _check_grasp without special bowl logic."""
    if hasattr(env, 'env'):
        env = env.env
    return env._check_grasp(env.robots[0].gripper, env.get_object(object_name))

def set_sim_state_from_flattened_with_update(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def get_end_location_of_stage(stage):
    predicate = stage_to_predicate(stage)
    action = predicate[0]
    if action == 'open':
        return predicate[1]
    elif action == 'grasp':
        return predicate[1]
    elif action == 'turnon':
        return predicate[1]
    elif action == 'place':
        return predicate[2]
    else:
        raise NotImplementedError("Not implemented")

class TaskDemonstrationGenerator:
    """Class to generate demonstrations for a task, storing task_metadata and env as instance variables."""
    
    def __init__(self, bddl_file_path, vis_grasp_poses=False, keep_failures=False, max_grasp_poses=None, resolution=128, log_position_error_on_fail=False, suppress_print=False, run_dir=None, config_path=None, bddl_path=None, suffix=None, skip_variations=False):
        if run_dir is None:
            raise ValueError("run_dir must be provided")
        self.bddl_file_path = bddl_file_path
        self.vis_grasp_poses = vis_grasp_poses
        self.keep_failures = keep_failures
        self.max_grasp_poses = max_grasp_poses  # None means no limit
        self.resolution = resolution
        self.log_position_error_on_fail = log_position_error_on_fail
        self.suppress_print = suppress_print
        self.run_dir = run_dir
        self.bddl_path = bddl_path
        self.suffix = suffix
        self.skip_variations = skip_variations
        
        # Load configuration from YAML file
        if config_path is None:
            # Default to generate_spec.yaml in generate_demonstrations_configs directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, "generate_demonstrations_configs")
            config_path = os.path.join(config_dir, "generate_spec.yaml")
        
        demo_config = load_demonstrations_config(config_path)
        
        # Store all configuration values as instance variables
        self.object_grasp_configs = demo_config["object_grasp_configs"]
        self.no_position_variation_combinations = demo_config["no_position_variation_combinations"]
        self.default_above_place_location_offset = demo_config["default_above_place_location_offset"]
        self.default_release_depth_offset = demo_config["default_release_depth_offset"]
        self.above_place_location_offsets_world = deepcopy(demo_config["above_place_location_offsets_world"])
        self.place_location_local_offset = deepcopy(demo_config["place_location_local_offset"])
        self.release_depth_place_location_offsets_world = deepcopy(demo_config["release_depth_place_location_offsets_world"])
        self.release_depth_grasped_object_offsets_world = deepcopy(demo_config["release_depth_grasped_object_offsets_world"])
        self.ignore_predicate_check_during_place_movement_for_location = deepcopy(
            demo_config["ignore_predicate_check_during_place_movement_for_location"]
        )
        self.ignore_predicate_check_during_place_movement_for_location_base = {
            get_base_object_name(name)
            for name in self.ignore_predicate_check_during_place_movement_for_location
        }
        self.ignore_predicate_check_during_set_down_for_location = deepcopy(
            demo_config["ignore_predicate_check_during_set_down_for_location"]
        )
        self.ignore_predicate_check_during_set_down_for_location_base = {
            get_base_object_name(name)
            for name in self.ignore_predicate_check_during_set_down_for_location
        }
        self.intermediate_quaternion = demo_config["intermediate_quaternion"].copy()
        self.intermediate_pose_overrides = deepcopy(demo_config["intermediate_pose_overrides"])
        self.intermediate_pos_variation_range = demo_config["intermediate_pos_variation_range"]
        self.intermediate_rot_variation_degrees = demo_config["intermediate_rot_variation_degrees"]
        self.intermediate_rot_variation_rad = demo_config["intermediate_rot_variation_rad"]
        self.intermediate_pose_z_offset = demo_config["intermediate_pose_z_offset"]
        self.max_intermediate_pose_height = demo_config["max_intermediate_pose_height"]
        self.max_height = demo_config["max_height"]
        self.pregrasp_offset = demo_config["pregrasp_offset"]
        self.pregrasp_additional_z_offset_overrides = demo_config.get("pregrasp_additional_z_offset_overrides", [])
        self.target_pos_variation_range = demo_config["target_pos_variation_range"]
        self.target_pos_variation_range_overrides = demo_config["target_pos_variation_range_overrides"]
        self.rot_gain_position_scaling = demo_config["rot_gain_position_scaling"]
        self.object_fall_angle_threshold_degrees = demo_config["object_fall_angle_threshold_degrees"]
        self.object_fall_angle_threshold_rad = demo_config["object_fall_angle_threshold_rad"]
        self.additional_replay_steps_after_stage = deepcopy(demo_config.get("additional_replay_steps_after_stage", []))
        self.check_place_location_fall_instead_of_placed_object = demo_config.get("check_place_location_fall_instead_of_placed_object", [])
        self.skip_fall_check_when_placed_on = demo_config.get("skip_fall_check_when_placed_on", {})
        self.grasp_position_precision_override = demo_config.get("grasp_position_precision_override", {})
        split_name = os.path.basename(os.path.dirname(bddl_file_path))
        bddl_name = os.path.basename(bddl_file_path).replace(".bddl", "")
        self.bddl_name = bddl_name
        
        # Load the metadata for the task
        self.task_metadata = load_metadata_for_task(bddl_file_path, bddl_path=self.bddl_path, suffix=self.suffix)
        self.actions_from = self.task_metadata.get("actions_from", None)  # Optional
        self.base_split_name = self.task_metadata["base_split_name"]
        self.actions_from_steps = self.task_metadata.get("actions_from_steps", [])  # Optional, default to empty list
        self.execution_steps = self.task_metadata["execution_steps"]
        
        # Parse initial states from BDDL file for checking initial object locations
        self.initial_states = self._parse_initial_states_from_bddl(bddl_file_path)
        
        # Load the environment (enable camera obs if visualizing grasp poses)
        env, self.env_info, self.problem_info = get_environment(bddl_file_path, use_camera_obs=self.vis_grasp_poses, resolution=self.resolution, suppress_print=self.suppress_print)
        
        # Create mapping from actual object/location names to base names
        # This allows us to match generic names from YAML config to specific instances in the environment
        self.env_name_to_base_name = {}
        if hasattr(env, 'env'):
            env_unwrapped = env.env
        else:
            env_unwrapped = env
        
        # Collect all object and location names from the environment
        all_env_names = set()
        if hasattr(env_unwrapped, 'object_states_dict'):
            all_env_names.update(env_unwrapped.object_states_dict.keys())
        if hasattr(env_unwrapped, 'object_sites_dict'):
            all_env_names.update(env_unwrapped.object_sites_dict.keys())
        if hasattr(env_unwrapped, 'objects_dict'):
            all_env_names.update(env_unwrapped.objects_dict.keys())
        if hasattr(env_unwrapped, 'fixtures_dict'):
            all_env_names.update(env_unwrapped.fixtures_dict.keys())
        
        # Create mapping: actual_name -> base_name
        for actual_name in all_env_names:
            base_name = get_base_object_name(actual_name)
            self.env_name_to_base_name[actual_name] = base_name
        
        # Create base output directory for grasp pose images if visualizing
        if self.vis_grasp_poses:
            self.grasp_poses_base_dir = os.path.join(self.run_dir, "tmp_grasp_poses", f"{split_name}/{bddl_name}")
            os.makedirs(self.grasp_poses_base_dir, exist_ok=True)
        
        # Setup temporary directories
        extension = f"{split_name}/{bddl_name}/{str(time.time()).replace('.', '_')}"
        self.tmp_directory = os.path.join(self.run_dir, "states", extension)
        self.tmp_out_directory = os.path.join(self.run_dir, "demonstrations", extension)
        os.makedirs(self.tmp_directory, exist_ok=True)
        os.makedirs(self.tmp_out_directory, exist_ok=True)
        self.env = DataCollectionWrapper(env, self.tmp_directory)
        
        # Get path to actions from hdf5 (only if actions_from is specified)
        self.actions_from_hdf5_path = None
        if self.actions_from is not None:
            self.actions_from_hdf5_path = task_name_to_hdf5_path(self.task_metadata, self.actions_from, suffix=self.suffix)        
        
        
        # Track failed demo directories to exclude from final dataset
        self.failed_demo_directories = []
        
        # Track seeds used for each demo (mapping from demo_idx to seed)
        self.demo_seeds = {}
        # Track mapping from episode directory to demo_idx (for matching seeds when gathering)
        self.ep_directory_to_demo_idx = {}

        # Compute relative grasp locations during initialization
        # Only compute for steps that are not covered by actions_from
        relative_grasp_locations = {}
        if self.actions_from is not None:
            search_execution_steps = self.execution_steps[len(self.actions_from_steps):]
        else:
            # If no actions_from, all steps need relative grasp locations
            search_execution_steps = self.execution_steps

        for stage in search_execution_steps:
            relative_grasp_locations[stage] = self.get_rel_robot_pose_from_teleop_data(stage)

        self.relative_grasp_locations = relative_grasp_locations
    
    def _parse_initial_states_from_bddl(self, bddl_file_path):
        """
        Parse initial states from BDDL file to get object initial locations.
        Returns a dictionary mapping object_name -> location.
        """
        initial_states = {}
        try:
            with open(bddl_file_path, "r", encoding="utf-8") as f:
                bddl_content = f.read()
            
            # Find the :init section
            init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", bddl_content, re.S)
            if not init_block_match:
                return initial_states
            
            init_text = init_block_match.group(1)
            
            # Parse On predicates: (On object location)
            for match in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_text):
                obj, location = match.groups()
                initial_states[obj] = location
            
            # Parse In predicates: (In object container)
            for match in re.finditer(r"\(In\s+(\w+)\s+(\w+)\)", init_text):
                obj, container = match.groups()
                initial_states[obj] = container
            
        except Exception as e:
            self._print(f"Warning: Could not parse initial states from BDDL file {bddl_file_path}: {e}")
        
        return initial_states
    
    def _print(self, *args, **kwargs):
        """Conditional print that respects suppress_print flag."""
        if not self.suppress_print:
            print(*args, **kwargs)

    def _print_goal_predicate_status(self):
        """Print the pass/fail status of each goal predicate in the current environment."""
        self._print(f"   Goal predicate status:")
        raw_env = self.env.env if hasattr(self.env, 'env') else self.env
        for predicate in raw_env.parsed_problem.get("goal_state", []):
            passed = raw_env._eval_predicate(predicate)
            self._print(f"     {'PASS' if passed else 'FAIL'}: {' '.join(predicate)}")
    
    def check_object_grasped(self, env, object_name):
        """
        Check if object is grasped, with special handling for objects configured for single-contact grasp.
        
        Returns:
            tuple: (is_grasped: bool, is_single_contact: bool)
                - is_grasped: True if object is grasped
                - is_single_contact: True if grasp is single contact (only one fingerpad in contact)
        """
        if hasattr(env, 'env'):
            env = env.env
        
        gripper = env.robots[0].gripper
        obj = env.get_object(object_name)
        
        # Get fingerpad geoms (same way _check_grasp does it)
        left_fingerpad_geoms = gripper.important_geoms["left_fingerpad"]
        right_fingerpad_geoms = gripper.important_geoms["right_fingerpad"]
        
        # Get object contact geoms (same way _check_grasp does it)
        obj_geoms = obj.contact_geoms
        
        # Check contact for each fingerpad using env.check_contact (same as _check_grasp)
        left_contact = env.check_contact(left_fingerpad_geoms, obj_geoms)
        right_contact = env.check_contact(right_fingerpad_geoms, obj_geoms)
        
        # Extract base object name (e.g., "akita_black_bowl_1" -> "akita_black_bowl")
        # Object names typically have format: base_name_instance_number
        base_object_name = None
        for base_name in self.object_grasp_configs.keys():
            if object_name.startswith(base_name):
                base_object_name = base_name
                break
        
        # Check if this object has single-contact grasp configuration
        is_single_contact = False
        if base_object_name and base_object_name in self.object_grasp_configs:
            grasp_config = self.object_grasp_configs[base_object_name]
            if grasp_config.get("allow_single_contact", False):
                # If both are in contact, return True immediately (not single contact)
                if left_contact and right_contact:
                    grasp_result = True
                    return (grasp_result, False)
                
                # Check if only one fingerpad is in contact (not both, not neither)
                single_contact = (left_contact and not right_contact) or (right_contact and not left_contact)
                
                # If single contact is detected, return True immediately
                if single_contact:
                    grasp_result = True
                    return (grasp_result, True)
        
        # Get result from _check_grasp
        grasp_result = env._check_grasp(gripper, obj)
        
        return (grasp_result, False)
    
    def check_predicate_success(self, env, stage):
        """Check if a predicate stage has been successfully completed."""
        predicate = stage_to_predicate(stage)

        # handle the case where the environment is wrapped in a visualization wrapper
        if hasattr(env, 'env'):
            env = env.env

        # we add support for additional predicate checks here
        if predicate[0] == 'grasp':
            # Use the class method for grasp checking (includes special bowl handling)
            is_grasped, _ = self.check_object_grasped(env, predicate[1])
            return is_grasped
        elif predicate[0] == 'place':
            # place can be in or on the location
            predicate[0] = 'on' # [place, A, B] to [on, A, B]
            option1 = env._eval_predicate(predicate)
            predicate[0] = 'in' # [place, A, B] to [in, A, B]
            try:
                option2 = env._eval_predicate(predicate)
            except:
                # some objects will fail this predicate if they do not support it. for example you can't check "in" with a plate or it will fail
                option2 = False
            return option1 or option2
        
        return env._eval_predicate(predicate)
    
    def get_rel_robot_pose_from_teleop_data(self, stage):
        # Search through existing tasks to find the location where an object is grasped
        all_metadata = self.task_metadata["all_metadata"]
        reference_task_names = sorted(list(all_metadata.keys()))
        
        requested_stage_predicate = stage_to_predicate(stage)
        
        # For place actions, compute relative pose based on site location
        if requested_stage_predicate[0] == 'place':
            # Get the place location (site) and the object being placed
            place_location = requested_stage_predicate[2]  # The site/location where to place
            placed_object_name = requested_stage_predicate[1]  # The object being placed
            
            # Get the minimum z-coordinate of the object's geometry relative to its origin
            # This represents the bottom of the object's bounding box
            object_min_z = get_object_min_z(self.env, placed_object_name)
            
            # Compute relative pose: object should be placed at site location + z offset
            # The relative pose represents where the object origin should be (not the robot)
            # Position: [0, 0, -object_min_z] in world coordinates
            # This places the object origin so that the bottom of the object (min_z) is at the site location
            # If min_z is negative (geometry extends below origin), we need to move origin up by -min_z
            rel_pos = np.array([0.0, 0.0, -object_min_z])
            
            # Apply manual offsets for specific place locations (world coordinates, z-only)
            # Default to default_above_place_location_offset if not specified
            # Use base name for lookup to match generic names in YAML config
            place_location_base_name = get_base_object_name(place_location)
            offset = self.above_place_location_offsets_world.get(place_location_base_name, self.default_above_place_location_offset)
            rel_pos[2] += offset
            
            # Check for custom stage-based offsets (in site's local coordinate frame)
            # These will be transformed to world coordinates in move_from_to using the current site pose
            local_offset = None
            # Check place_location_local_offset for the first satisfied condition
            for offset_entry in self.place_location_local_offset:
                # Check base_split_condition
                if offset_entry.get('base_split_condition') != self.base_split_name:
                    continue
                
                # Check predicate_conditions - all must be satisfied (if specified)
                predicate_conditions = offset_entry.get('predicate_conditions', [])
                all_predicates_satisfied = True
                for predicate_str in predicate_conditions:
                    if not self.check_predicate_success(self.env, predicate_str):
                        all_predicates_satisfied = False
                        break
                if not all_predicates_satisfied:
                    continue
                
                # Check target_conditions
                target_conditions = offset_entry.get('target_conditions', {})
                target_placed_object = target_conditions.get('placed_object_name')
                target_place_location = target_conditions.get('place_location')
                
                # Check place_location match (required)
                # Support both string and list of strings
                # Use base name for comparison to handle _resized suffixes
                place_location_base = get_base_object_name(place_location)
                if isinstance(target_place_location, list):
                    # If it's a list, check if place_location_base matches any element's base name
                    place_location_matches = False
                    for loc in target_place_location:
                        loc_base = get_base_object_name(loc)
                        if loc_base == place_location_base:
                            place_location_matches = True
                            break
                    if not place_location_matches:
                        continue
                else:
                    # If it's a string, do base name comparison
                    target_place_location_base = get_base_object_name(target_place_location)
                    if target_place_location_base != place_location_base:
                        continue
                
                # Check placed_object_name match (optional - only if specified)
                if target_placed_object is not None and target_placed_object != placed_object_name:
                    continue
                
                # All conditions satisfied, use this offset
                local_offset = offset_entry['offset'].copy()
                break
            
            # For place, we only store position (orientation will be determined by current robot orientation)
            # Store as relative pose with NaN for orientation to indicate position-only
            relative_pose = np.concatenate([rel_pos, [np.nan, np.nan, np.nan, np.nan]])
            
            # Return a list with a single entry (consistent with the format from _process_matching_stage_for_rel_robot_pose)
            result_dict = {
                'relative_pose': relative_pose,
                'object_name': place_location,  # The site location
                'use_position_only': True,  # Only position is used for place
                'is_object_target': True,  # Flag to indicate this is object target, not robot target
            }
            
            # Store local offset if present (will be transformed in move_from_to)
            if local_offset is not None:
                result_dict['local_offset'] = local_offset
            
            return [result_dict]
        elif requested_stage_predicate[0] == 'touch':
            # touch is not supported yet except in action replay
            return []
        elif requested_stage_predicate[0] == 'grasp':
            object_name = requested_stage_predicate[1]
            
            # Check if grasps_from is specified in metadata
            # Format: {'object_name': {'task': 'task_name', 'object': 'source_object_name'}}
            # The 'object' field specifies which object in the source task to use
            # (e.g., if current task uses bowl_1 but source task has bowl_2)
            grasps_from = self.task_metadata.get('grasps_from', {})
            if object_name in grasps_from:
                grasp_info = grasps_from[object_name]
                if not isinstance(grasp_info, dict):
                    raise ValueError(
                        f"grasps_from['{object_name}'] must be a dict with 'task' and 'object' keys. "
                        f"Got: {type(grasp_info).__name__} = {grasp_info}"
                    )
                if 'task' not in grasp_info or 'object' not in grasp_info:
                    raise ValueError(
                        f"grasps_from['{object_name}'] must be a dict with 'task' and 'object' keys. "
                        f"Got: {grasp_info}"
                    )
                reference_task_name = grasp_info['task']
                source_object_name = grasp_info['object']
                
                reference_task_data = all_metadata[reference_task_name]
                
                # Construct the stage string using the source object name
                # Stage format is like "Grasp bowl_2"
                reference_stage_str = f"Grasp {source_object_name}"
                
                # Find the matching stage in the reference task
                reference_execution_steps = reference_task_data["execution_steps"]
                reference_stage = None
                for ref_stage in reference_execution_steps:
                    if ref_stage == reference_stage_str:
                        reference_stage = ref_stage
                        break
                
                if reference_stage is None:
                    raise ValueError(f"Could not find stage {reference_stage_str} in reference task {reference_task_name} from grasps_from")
                
                use_position_only = False  # For grasp, always use full pose
                # Use the source object name from grasps_from when querying the reference environment
                # But store the original object name from the current task in the result
                result = self._process_matching_stage_for_rel_robot_pose(
                    reference_task_name, reference_stage, stage,
                    requested_stage_predicate, use_position_only,
                    object_name_override=source_object_name,  # Use source object name from grasps_from for querying
                    original_object_name=object_name  # Use original object name from current task for storage
                )
                if result is not None:
                    return result
                else:
                    raise ValueError(f"Could not find valid states for stage {reference_stage} in reference task {reference_task_name} from grasps_from")
            else:
                # grasps_from must be specified for all grasp actions
                raise ValueError(
                    f"grasps_from must be specified in task metadata for object '{object_name}' in stage '{stage}'. "
                    f"Please ensure grasps_from is properly configured in the task metadata."
                )
        else:
            raise NotImplementedError(f"Not implemented for action: {requested_stage_predicate[0]} for stage {stage} for bddl file {self.bddl_file_path}")        
    
    def _process_matching_stage_for_rel_robot_pose(self, reference_task_name, reference_stage, stage, 
                                requested_stage_predicate, use_position_only, object_name_override=None, 
                                original_object_name=None):
        """Process a matching stage and return relative poses list, or None if no valid states found.
        
        Args:
            reference_task_name: Name of the reference task
            reference_stage: Stage string from the reference task (e.g., "Grasp bowl_2")
            stage: Stage string from the current task (e.g., "Grasp bowl_1")
            requested_stage_predicate: Predicate from the current task's stage
            use_position_only: Whether to use position only or full pose
            object_name_override: Optional object name to use when querying the reference environment.
                                 If provided, this overrides the object name from requested_stage_predicate.
                                 This is used when the object name in the reference task differs from
                                 the current task (e.g., when using grasps_from).
            original_object_name: Optional object name from the current task to store in the result.
                                 If provided, this is used instead of the object name from the reference task.
                                 If None, extracts from requested_stage_predicate.
        """
        # This means that this reference task has the informmation we are searching for

        # Load the reference environment (enable camera obs if visualizing)
        reference_bddl_path = task_name_to_bddl_path(self.task_metadata, reference_task_name, bddl_path=self.bddl_path, suffix=self.suffix)
        reference_env, _, _ = get_environment(reference_bddl_path, use_camera_obs=self.vis_grasp_poses, resolution=self.resolution, suppress_print=self.suppress_print)
        reference_env.seed(7)
        reference_env.reset()
        
        # Load the reference demonstration data
        hdf5_path = task_name_to_hdf5_path(self.task_metadata, reference_task_name, suffix=self.suffix)
        
        # Get the object/location name to use when querying the reference environment
        # If object_name_override is provided (e.g., from grasps_from), use that instead
        # Otherwise, extract from the requested_stage_predicate
        if object_name_override is not None:
            query_object_name = object_name_override  # Use source object name for querying reference env
        else:
            # Get the object/location name from the stage
            # For grasp: object is predicate[1] (the object to grasp)
            # For place: location is predicate[2] (where to place the object)
            if requested_stage_predicate[0] == 'grasp':
                query_object_name = requested_stage_predicate[1]
            elif requested_stage_predicate[0] == 'place':
                query_object_name = requested_stage_predicate[2]  # The location where to place
            else:
                raise NotImplementedError(f"Not implemented for action: {requested_stage_predicate[0]}")
        
        # Get the object name to store in the result (from current task, not reference task)
        if original_object_name is not None:
            stored_object_name = original_object_name  # Use original object name from current task
        else:
            # Extract from requested_stage_predicate (same as query_object_name when no override)
            if requested_stage_predicate[0] == 'grasp':
                stored_object_name = requested_stage_predicate[1]
            elif requested_stage_predicate[0] == 'place':
                stored_object_name = requested_stage_predicate[2]
            else:
                raise NotImplementedError(f"Not implemented for action: {requested_stage_predicate[0]}")
        
        # Create subdirectory for this stage if visualizing
        if self.vis_grasp_poses:
            safe_stage = stage.replace(" ", "_").replace("/", "_")
            stage_dir = os.path.join(self.grasp_poses_base_dir, safe_stage)
            os.makedirs(stage_dir, exist_ok=True)
        
        # Collect relative poses from demos until we have max_grasp_poses (if set)
        # Continue through demos even if some don't have valid grasp poses
        relative_poses_list = []
        
        with h5py.File(hdf5_path, "r") as f:
            # Get all demo keys and sort them
            demo_keys = sorted([key for key in f["data"].keys() if key.startswith("demo_")], 
                             key=lambda x: int(x.split("_")[1]))
            
            # Use tqdm to show progress when processing demos
            desc = f"Collecting grasp poses for {stage}"
            for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc=desc, leave=False)):
                # Check if we've collected enough grasp poses
                if self.max_grasp_poses is not None and len(relative_poses_list) >= self.max_grasp_poses:
                    break
                
                demo_group = f["data"][demo_key]
                reference_states = demo_group["states"][:]
                
                # find a state in the reference states that meets the requested predicate
                found_state = False
                for i in range(len(reference_states)):
                    reference_state = reference_states[i]
                    set_sim_state_from_flattened_with_update(reference_env, reference_state)
                    if self.check_predicate_success(reference_env, reference_stage):
                        # this means that this stage satisfies the requested predicate, so we want to extract the relative robot pose
                        found_state = True
                        
                        # Save image if visualization is enabled
                        if self.vis_grasp_poses:
                            obs = reference_env._get_observations()
                            # Convert BGR to RGB and flip vertically (robosuite convention)
                            img = obs["agentview_image"][::-1, :, ::-1]
                            # Save image with descriptive filename
                            safe_ref_stage = reference_stage.replace(" ", "_").replace("/", "_")
                            filename = f"grasp_pose_from_{reference_task_name}_{safe_ref_stage}_demo_{demo_idx}.png"
                            filepath = os.path.join(stage_dir, filename)
                            cv2.imwrite(filepath, img)
                        
                        # Get robot pose and object pose
                        current_robot_pose = get_robot_poses_at_current_state(reference_env)
                        
                        # Get object/location pose using the query object name (from reference task)
                        object_pose = get_object_pose(reference_env, query_object_name)
                        
                        # Compute relative pose from the robot pose
                        if use_position_only:
                            # Only compute position, set orientation to NaN to indicate it's missing
                            rel_pos = compute_relative_pose(current_robot_pose, object_pose)[:3]
                            relative_pose = np.concatenate([rel_pos, [np.nan, np.nan, np.nan, np.nan]])
                        else:
                            # Compute full relative pose (position + orientation)
                            relative_pose = compute_relative_pose(current_robot_pose, object_pose)
                        
                        # Store relative pose info with the original object name from the current task
                        relative_poses_list.append({
                            'relative_pose': relative_pose,
                            'object_name': stored_object_name,  # Use object name from current task, not reference task
                            'use_position_only': use_position_only,
                        })
                        
                        break  # Found the state for this demo, move to next demo
                
                if not found_state:
                    self._print(f" - Warning: Could not find the location where the stage {reference_stage} is satisfied in demo {demo_key} of reference task {reference_task_name}")
        
        reference_env.close()
        
        # Assert that at least 1 grasp pose was successfully collected
        assert len(relative_poses_list) > 0, (
            f"No valid grasp poses found for stage {reference_stage} in reference task {reference_task_name}. "
            f"Searched through {len(demo_keys)} demos."
        )
        
        # Return list of relative poses from all demos
        return relative_poses_list

    def move_to_pose(self, pose, close_gripper=False, kp_pos=6.0, kp_rot=2.0, max_pos_speed=0.5, max_rot_speed=0.5, 
                     pos_threshold=0.01, rot_threshold=0.05, max_steps=150, 
                     enable_early_termination=False, early_termination_distance=0.04, early_termination_speed=0.25,
                     check_stage=None, rot_gain_position_scaling=None):
        """
        Move end effector to target pose using proportional controller.
        
        Args:
            pose: Target pose as [pos (3,), quat (4,)] = 7 values
            close_gripper: If True, gripper action is 1 (close), if False, gripper action is -1 (open)
            kp_pos: Position proportional gain
            kp_rot: Rotation proportional gain
            max_pos_speed: Maximum position speed (m/s)
            max_rot_speed: Maximum rotation speed (rad/s)
            pos_threshold: Position convergence threshold (m)
            rot_threshold: Rotation convergence threshold (rad)
            max_steps: Maximum number of control steps
            early_termination_distance: Distance threshold for early termination (m)
            early_termination_speed: Speed threshold for early termination (m/s)
            check_stage: Optional stage to check predicate for. If predicate succeeds during movement, returns True early.
            rot_gain_position_scaling: Scaling constant for rotation gain based on position distance.
                                      If None, rotation gain is constant. If provided, rotation gain is scaled
                                      as kp_rot * min(1.0, rot_gain_position_scaling / pos_error_norm).
                                      This makes rotation smoother when far away (lower gain) and more responsive
                                      when close (higher gain, up to original kp_rot).
        """
        # Extract target position and quaternion
        # Note: robosuite returns quaternions in XYZW format (x, y, z, w)
        target_pos = pose[:3].copy()
        target_quat = pose[3:7]  # Already in XYZW format from robosuite observations
        
        # Clip z value to max_height if configured for this base split
        max_height = self.max_height.get(self.base_split_name)
        if max_height is not None and target_pos[2] > max_height:
            original_z = target_pos[2]
            target_pos[2] = max_height
            self._print(f"     - Clipped target pose z from {original_z:.4f} to {max_height:.4f} (max_height for {self.base_split_name})")
        
        # Convert target quaternion to rotation matrix
        target_rot_mat = Rotation.from_quat(target_quat, scalar_first=False).as_matrix()
        
        # Get initial observation
        obs = self.env._get_observations()
        
        # Track errors over time for logging if convergence fails (only if flag is enabled)
        if self.log_position_error_on_fail:
            pos_errors = []
            rot_errors = []
            positions = []
        
        for step in range(max_steps):
            # Get current pose from observation
            current_pose = get_robot_poses_at_current_state(self.env, obs=obs)
            current_pos = current_pose[:3]
            current_quat = current_pose[3:7]
            
            # Convert current quaternion to rotation matrix
            current_rot_mat = Rotation.from_quat(current_quat, scalar_first=False).as_matrix()
            
            # Compute position error
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            # Compute rotation error (axis-angle difference)
            rel_rot_mat = target_rot_mat @ current_rot_mat.T
            rel_rot = Rotation.from_matrix(rel_rot_mat).as_rotvec()
            rot_error_norm = np.linalg.norm(rel_rot)
            
            # Store errors for logging (only if flag is enabled)
            if self.log_position_error_on_fail:
                pos_errors.append(pos_error_norm)
                rot_errors.append(rot_error_norm)
                positions.append(current_pos.copy())
            
            # Check convergence
            if pos_error_norm < pos_threshold and rot_error_norm < rot_threshold:
                break
            
            # Compute proportional control action
            # Position action
            pos_action = kp_pos * pos_error
            pos_norm = np.linalg.norm(pos_action)
            if pos_norm > max_pos_speed:
                pos_action = pos_action / pos_norm * max_pos_speed
            
            # Rotation action with distance-based gain scaling
            # Scale rotation gain based on position distance: far away = lower gain, close = higher gain
            # Use instance variable if parameter is None
            scaling_constant = rot_gain_position_scaling if rot_gain_position_scaling is not None else self.rot_gain_position_scaling
            if scaling_constant is not None:
                # Avoid division by zero by using a small epsilon
                epsilon = 1e-6
                # Scale factor: when far away (large pos_error_norm), scaling is small
                # When close (small pos_error_norm), scaling approaches 1.0 (but capped at 1.0)
                rot_gain_scale = min(1.0, scaling_constant / max(pos_error_norm, epsilon))
                kp_rot_scaled = kp_rot * rot_gain_scale
            else:
                kp_rot_scaled = kp_rot
            
            rot_action = kp_rot_scaled * rel_rot
            rot_norm = np.linalg.norm(rot_action)
            if rot_norm > max_rot_speed:
                rot_action = rot_action / rot_norm * max_rot_speed
            
            # Check for early termination: within distance threshold and speed is low (only if enabled)
            if (enable_early_termination and 
                pos_error_norm < early_termination_distance and pos_norm < early_termination_speed):
                self._print(f"     - Early termination: within {early_termination_distance:.3f}m and speed {pos_norm:.3f} < {early_termination_speed:.3f}")
                break
            
            # Check stage predicate if provided (check during movement)
            if check_stage is not None:
                if self.check_predicate_success(self.env, check_stage):
                    self._print(f"     - Stage predicate {check_stage} succeeded during movement, exiting successfully")
                    return True
            
            # Gripper action: 1 to close, -1 to open
            gripper_action = 1.0 if close_gripper else -1.0
            
            # Combine into delta action
            action = np.concatenate([pos_action, rot_action, [gripper_action]])
            
            # Execute action and get observation
            obs, reward, done, info = self.env.step(action)
        
        # Check if we converged successfully
        if step == max_steps - 1:
            self._print(f"     - Warning: move_to_pose did not converge after {max_steps} steps. "
                  f"Final pos error: {pos_error_norm:.4f}, rot error: {rot_error_norm:.4f}")
            if self.log_position_error_on_fail:
                self._print(f"\n     Position and orientation errors over time:")
                self._print(f"     Step | Position Error (m) | Orientation Error (rad)  |         x         |         y         |         z")
                self._print(f"     -----|---------------------|-------------------------|-------------------|-------------------|------------------")
                for i, (pos_err, rot_err, pos) in enumerate(zip(pos_errors, rot_errors, positions)):
                    self._print(f"     {i:4d} | {pos_err:19.6f} | {rot_err:23.6f} | {pos[0]:17.6f} | {pos[1]:17.6f} | {pos[2]:17.6f}")
                self._print(f"\n     Target pose: pos={target_pos}, quat={target_quat}")
                self._print(f"     Final pose: pos={current_pos}, quat={current_quat}")
            return False
        
        self._print(f"     - Moved to pose after {step} steps")
        return True

    def _add_rotation_variation_to_quat(self, quat):
        """
        Add random rotation variation to a quaternion.
        
        Args:
            quat: numpy array of shape (4,) containing quaternion in xyzw format
            
        Returns:
            varied_quat: numpy array of shape (4,) with added rotation variation
        """
        # Add uniform rotation variation: ±intermediate_rot_variation_rad
        # Generate a random rotation axis and angle
        random_axis = np.random.randn(3)
        random_axis = random_axis / np.linalg.norm(random_axis)  # Normalize to unit vector
        random_angle = np.random.uniform(
            -self.intermediate_rot_variation_rad,
            self.intermediate_rot_variation_rad
        )
        
        # Create rotation from axis-angle
        variation_rotation = Rotation.from_rotvec(random_angle * random_axis)
        
        # Apply variation to the original quaternion
        original_rotation = Rotation.from_quat(quat, scalar_first=False)
        varied_rotation = variation_rotation * original_rotation
        varied_quat = varied_rotation.as_quat(scalar_first=False)
        
        return varied_quat

    def _add_variation_to_pose(self, pose, enable_position_variation=True, enable_rotation_variation=True):
        """
        Add random variation to a pose.
        
        Args:
            pose: numpy array of shape (7,) containing [pos (3,), quat (4,)] in xyzw format
            enable_position_variation: If True, apply variation to position components
            enable_rotation_variation: If True, apply variation to rotation (quaternion)
            
        Returns:
            varied_pose: numpy array of shape (7,) with added variation
        """
        # If skip_variations is enabled, return original pose without any variation
        if self.skip_variations:
            return pose.copy()
        
        pos = pose[:3]
        quat = pose[3:7]
        
        # Add uniform position variation: ±intermediate_pos_variation_range for each component
        if enable_position_variation:
            pos_variation = np.random.uniform(
                -self.intermediate_pos_variation_range,
                self.intermediate_pos_variation_range,
                size=3
            )
            varied_pos = pos + pos_variation
        else:
            varied_pos = pos
        
        # Add rotation variation
        if enable_rotation_variation:
            varied_quat = self._add_rotation_variation_to_quat(quat)
        else:
            varied_quat = quat
        
        varied_pose = np.concatenate([varied_pos, varied_quat])
        return varied_pose

    def move_from_to(self, from_location_name, to_stage, grasped_object_name=None, demo_idx=0, ensure_upright_object=False, enable_target_position_variation=False, skip_predicate_check_for_place_location=False):
        # If grasped_object_name is provided, we're holding an object and should keep gripper closed
        close_gripper = grasped_object_name is not None
        self._print(f"Moving from location: {from_location_name} to stage: {to_stage} with grasped_object: {grasped_object_name}")
        
        # first move to intermediate location
        self._print(" - Moving to intermediate location")
        
        # Check intermediate_pose_overrides for the first satisfied condition
        intermediate_pose_overrides_list = None  # Can be None, single pose, or list of poses
        auto_compute_after = False  # Whether to append a dynamic pose after the custom override poses
        override_skip_variation = False  # Whether to skip variation for override poses (auto_compute_after pose still gets variation)
        to_stage_predicate = stage_to_predicate(to_stage)
        
        for override_entry in self.intermediate_pose_overrides:
            # Check base_split_condition
            if override_entry.get('base_split_condition') != self.base_split_name:
                continue

            # Check predicate_conditions - all must be satisfied
            predicate_conditions = override_entry.get('predicate_conditions', [])
            all_predicates_satisfied = True
            for predicate_str in predicate_conditions:
                # Parse predicate string: "PredicateName arg1 [arg2]"
                predicate_parts = predicate_str.split()
                if len(predicate_parts) == 3:
                    # Two-argument predicate (e.g. "On wine_bottle wooden_cabinet_top_side")
                    # Resolve each argument independently to their env names, then try all combinations
                    predicate_name = predicate_parts[0]
                    arg1_generic = predicate_parts[1]
                    arg2_generic = predicate_parts[2]

                    matching_arg1_names = [
                        actual_name for actual_name, base_name in self.env_name_to_base_name.items()
                        if base_name == arg1_generic
                    ] or [arg1_generic]  # fall back to literal if no mapping found

                    matching_arg2_names = [
                        actual_name for actual_name, base_name in self.env_name_to_base_name.items()
                        if base_name == arg2_generic
                    ] or [arg2_generic]

                    predicate_satisfied = False
                    for env_name1 in matching_arg1_names:
                        for env_name2 in matching_arg2_names:
                            predicate_str_with_env_names = f"{predicate_name} {env_name1} {env_name2}"
                            if self.check_predicate_success(self.env, predicate_str_with_env_names):
                                predicate_satisfied = True
                                break
                        if predicate_satisfied:
                            break

                    if not predicate_satisfied:
                        all_predicates_satisfied = False
                        break
                elif len(predicate_parts) == 2:
                    # Single-argument predicate (e.g. "Open wooden_cabinet_top_region", "TurnOn flat_stove")
                    predicate_name = predicate_parts[0]
                    location_or_object_generic = predicate_parts[1]

                    # Find all environment names that map to this generic base name
                    matching_env_names = [
                        actual_name for actual_name, base_name in self.env_name_to_base_name.items()
                        if base_name == location_or_object_generic
                    ]

                    # Try each matching environment name until one satisfies the predicate
                    predicate_satisfied = False
                    for env_name in matching_env_names:
                        predicate_str_with_env_name = f"{predicate_name} {env_name}"
                        if self.check_predicate_success(self.env, predicate_str_with_env_name):
                            predicate_satisfied = True
                            break

                    if not predicate_satisfied:
                        all_predicates_satisfied = False
                        break
                else:
                    # No arguments to resolve, check as-is
                    if not self.check_predicate_success(self.env, predicate_str):
                        all_predicates_satisfied = False
                        break
            if not all_predicates_satisfied:
                continue

            # Check from_location condition if specified
            from_location_condition = override_entry.get('from_location')
            if from_location_condition is not None:
                from_location_base = get_base_object_name(from_location_name)
                from_location_condition_base = get_base_object_name(from_location_condition)
                if from_location_base != from_location_condition_base:
                    continue

            # Check target_conditions
            target_conditions = override_entry.get('target_conditions', {})
            target_object_name = target_conditions.get('object_name')
            target_place_location = target_conditions.get('place_location')
            target_placed_object = target_conditions.get('placed_object_name')
            
            target_conditions_satisfied = len(target_conditions) == 0  # True if no target_conditions specified
            if to_stage_predicate[0] == 'grasp':
                # For grasp: check if object_name matches and grasp_from matches (if specified)
                # Convert specific instance name to generic name for comparison
                grasp_object = to_stage_predicate[1]
                if target_object_name is not None:
                    grasp_object_base = get_base_object_name(grasp_object)
                    target_object_name_base = get_base_object_name(target_object_name)
                    if grasp_object_base == target_object_name_base:
                        grasp_from = target_conditions.get('grasp_from')
                        if grasp_from is None:
                            # No grasp_from specified, so object_name match is sufficient
                            target_conditions_satisfied = True
                        else:
                            # Check if grasps_from metadata matches
                            # Format: {'object_name': {'task': 'task_name', 'object': 'source_object_name'}}
                            grasps_from = self.task_metadata.get('grasps_from', {})
                            if grasp_object in grasps_from:
                                grasp_info = grasps_from[grasp_object]
                                if not isinstance(grasp_info, dict):
                                    raise ValueError(
                                        f"grasps_from['{grasp_object}'] must be a dict with 'task' and 'object' keys. "
                                        f"Got: {type(grasp_info).__name__} = {grasp_info}"
                                    )
                                if 'task' not in grasp_info:
                                    raise ValueError(
                                        f"grasps_from['{grasp_object}'] dict must have 'task' key. "
                                        f"Got: {grasp_info}"
                                    )
                                if grasp_info['task'] == grasp_from:
                                    target_conditions_satisfied = True
            elif to_stage_predicate[0] == 'place':
                # For place: support both legacy 'object_name' (as place_location)
                # and the newer, more explicit 'place_location' / 'placed_object_name'
                # Convert specific instance names to generic names for comparison
                placed_object = to_stage_predicate[1]
                placed_object_base = get_base_object_name(placed_object)
                place_location = to_stage_predicate[2]
                place_location_base = get_base_object_name(place_location)
                
                # Start assuming match then rule out by constraints
                candidate_match = True
                
                # If explicit place_location is provided, it must match (by base name for locations/regions)
                # Support both string and list of strings
                if target_place_location is not None:
                    if isinstance(target_place_location, list):
                        # If it's a list, check if place_location_base matches any element's base name
                        place_location_matches = False
                        for loc in target_place_location:
                            loc_base = get_base_object_name(loc)
                            if loc_base == place_location_base:
                                place_location_matches = True
                                break
                        if not place_location_matches:
                            candidate_match = False
                    else:
                        # If it's a string, do base name comparison
                        target_place_location_base = get_base_object_name(target_place_location)
                        if target_place_location_base != place_location_base:
                            candidate_match = False
                
                # If explicit placed_object_name is provided, it must match (by base name)
                if target_placed_object is not None:
                    target_placed_object_base = get_base_object_name(target_placed_object)
                    if placed_object_base != target_placed_object_base:
                        candidate_match = False
                
                # If neither explicit key is provided, fall back to legacy behavior:
                # interpret object_name as the place_location (by base name).
                if target_place_location is None and target_placed_object is None:
                    if target_object_name is not None:
                        target_object_name_base = get_base_object_name(target_object_name)
                        if target_object_name_base != place_location_base:
                            candidate_match = False
                    else:
                        candidate_match = False
                
                # Check initial_location condition if specified
                # Convert both to base names for comparison (initial_location may be a region with instance number)
                target_initial_location = target_conditions.get('initial_location')
                if target_initial_location is not None:
                    # Check if the placed object was initially at the specified location
                    object_initial_location = self.initial_states.get(placed_object)
                    if object_initial_location is not None:
                        # Convert both to base names for comparison
                        object_initial_location_base = get_base_object_name(object_initial_location)
                        target_initial_location_base = get_base_object_name(target_initial_location)
                        if object_initial_location_base != target_initial_location_base:
                            candidate_match = False
                    else:
                        # Object doesn't have an initial location, can't match
                        candidate_match = False
                
                # Check place_location_initial_location condition if specified
                # This allows matching based on where the place_location object is located
                target_place_location_initial_location = target_conditions.get('place_location_initial_location')
                if target_place_location_initial_location is not None:
                    # Check if the place_location object was initially at the specified location
                    place_location_object_initial_location = self.initial_states.get(place_location)
                    if place_location_object_initial_location is not None:
                        # Convert both to base names for comparison
                        place_location_object_initial_location_base = get_base_object_name(place_location_object_initial_location)
                        target_place_location_initial_location_base = get_base_object_name(target_place_location_initial_location)
                        if place_location_object_initial_location_base != target_place_location_initial_location_base:
                            candidate_match = False
                    else:
                        # Place location object doesn't have an initial location, can't match
                        candidate_match = False
                
                if candidate_match:
                    target_conditions_satisfied = True
            
            if target_conditions_satisfied:
                # All conditions satisfied, use this override
                # Validate that only one of 'pos', 'pose', 'z_delta' is specified
                has_pos = 'pos' in override_entry
                has_pose = 'pose' in override_entry
                has_pos_delta = 'pos_delta' in override_entry
                if sum([has_pos, has_pose, has_pos_delta]) > 1:
                    raise ValueError(f"Override entry must specify exactly one of 'pos', 'pose', or 'pos_delta': {override_entry}")

                if has_pose:
                    # Use full 7D pose if specified (single pose only)
                    intermediate_pose_overrides_list = [override_entry['pose'].copy()]
                elif has_pos:
                    # Check if pos is a list of positions (multiple intermediate poses) or single position
                    pos_value = override_entry['pos']
                    if isinstance(pos_value, list) and len(pos_value) > 0:
                        # Check if first element is a list/array (multiple positions)
                        if isinstance(pos_value[0], (list, np.ndarray)):
                            # Multiple positions: create list of poses
                            intermediate_pose_overrides_list = [
                                np.concatenate([np.array(pos), self.intermediate_quaternion.copy()])
                                for pos in pos_value
                            ]
                        else:
                            # Single position: create single pose
                            intermediate_pose_overrides_list = [
                                np.concatenate([np.array(pos_value), self.intermediate_quaternion.copy()])
                            ]
                    else:
                        # Single position (already numpy array)
                        intermediate_pose_overrides_list = [
                            np.concatenate([pos_value.copy(), self.intermediate_quaternion.copy()])
                        ]
                elif has_pos_delta:
                    # Offset current robot position by pos_delta (start_pos computed below)
                    # We store a sentinel and resolve it after start_pos is available
                    intermediate_pose_overrides_list = [('pos_delta', np.array(override_entry['pos_delta'], dtype=float))]
                else:
                    raise ValueError(f"Override entry must have one of 'pose', 'pos', or 'pos_delta': {override_entry}")
                self._print(f"   - Using intermediate pose override: base_split={override_entry.get('base_split_condition')}, "
                          f"predicates={predicate_conditions}, target={target_conditions}, "
                          f"num_poses={len(intermediate_pose_overrides_list)}")
                auto_compute_after = override_entry.get('auto_compute_after', False)
                override_skip_variation = override_entry.get('skip_variation', False)
                break

        # Get current robot pose (needed for computing dynamic intermediate pose and for orientation)
        obs = self.env._get_observations()
        current_robot_pose = get_robot_poses_at_current_state(self.env, obs=obs)
        start_pos = current_robot_pose[:3]

        def _compute_dynamic_intermediate_pose(start_pos):
            """Compute the auto-computed intermediate pose based on current and target positions."""
            relative_grasp_list = self.relative_grasp_locations[to_stage]
            selected_idx = demo_idx % len(relative_grasp_list)
            relative_grasp_info = relative_grasp_list[selected_idx]
            object_name = relative_grasp_info['object_name']
            target_object_pose = get_object_pose(self.env, object_name)
            target_pos = target_object_pose[:3]
            use_average_z = (
                self.actions_from is None and
                len(self.execution_steps) > 0 and to_stage == self.execution_steps[0] and
                to_stage_predicate[0] == 'grasp' and
                start_pos[2] > target_pos[2]
            )
            if use_average_z:
                intermediate_z = (start_pos[2] + target_pos[2]) / 2
            else:
                intermediate_z = max(start_pos[2], target_pos[2]) + self.intermediate_pose_z_offset
            intermediate_pos = np.array([
                (start_pos[0] + target_pos[0]) / 2,
                (start_pos[1] + target_pos[1]) / 2,
                intermediate_z
            ])
            return np.concatenate([intermediate_pos, self.intermediate_quaternion.copy()])

        # Resolve pos_delta sentinel now that start_pos is known
        if intermediate_pose_overrides_list is not None and len(intermediate_pose_overrides_list) == 1 and isinstance(intermediate_pose_overrides_list[0], tuple) and intermediate_pose_overrides_list[0][0] == 'pos_delta':
            pos_delta = intermediate_pose_overrides_list[0][1]
            intermediate_pose_overrides_list = [np.concatenate([start_pos + pos_delta, current_robot_pose[3:7]])]

        # Determine list of intermediate poses to visit
        # skip_position_variation_list is parallel to intermediate_poses_list
        if intermediate_pose_overrides_list is not None:
            # Use override poses (can be single or multiple)
            intermediate_poses_list = intermediate_pose_overrides_list
            skip_position_variation_list = [override_skip_variation] * len(intermediate_pose_overrides_list)
            # If auto_compute_after is set, append a dynamically computed pose after the custom poses
            if auto_compute_after:
                dynamic_pose = _compute_dynamic_intermediate_pose(start_pos)
                intermediate_poses_list = intermediate_poses_list + [dynamic_pose]
                skip_position_variation_list = skip_position_variation_list + [False]  # auto-computed pose always gets variation
                self._print(f"   - Appending auto-computed intermediate pose after custom override poses")
        else:
            # Compute intermediate pose dynamically: midpoint in x,y, and z offset above max(start_z, target_z)
            intermediate_poses_list = [_compute_dynamic_intermediate_pose(start_pos)]
            skip_position_variation_list = [False]

        # Move through each intermediate pose in sequence
        for pose_idx, (intermediate_pose, skip_variation) in enumerate(zip(intermediate_poses_list, skip_position_variation_list)):
            # If holding an object, substitute current orientation and adjust height based on object
            if grasped_object_name is not None:
                # Get object height and adjust intermediate position height
                object_height = get_object_height(self.env, grasped_object_name)
                adjusted_intermediate_pos = intermediate_pose[:3].copy()
                adjusted_intermediate_pos[2] += object_height / 2  # Add object height to z-coordinate # TODO: not the best way to do this because it assumes the robot grabs the object always at the exact top of the object. The divide by 2 is also hacky, but it seems to work well
                
                # Use adjusted intermediate position and current orientation
                intermediate_pose = np.concatenate([
                    adjusted_intermediate_pos,  # Intermediate position with height adjustment
                    current_robot_pose[3:7]     # Current orientation
                ])
                
                # Apply variation only to position (not rotation) when holding an object
                varied_intermediate_pose = self._add_variation_to_pose(
                    intermediate_pose,
                    enable_position_variation=not skip_variation,
                    enable_rotation_variation=False
                )
            else:
                # Apply variation to the full pose (position + rotation) when not holding an object
                varied_intermediate_pose = self._add_variation_to_pose(
                    intermediate_pose,
                    enable_position_variation=not skip_variation,
                    enable_rotation_variation=not skip_variation
                )
            
            # Limit height to max_intermediate_pose_height if configured for this base split
            max_height = self.max_intermediate_pose_height.get(self.base_split_name)
            if max_height is not None:
                if varied_intermediate_pose[2] > max_height:
                    self._print(f"     - Clipped intermediate pose z from {varied_intermediate_pose[2]:.4f} to {max_height:.4f} (max_intermediate_pose_height for {self.base_split_name})")
                    varied_intermediate_pose[2] = max_height
            
            # Move to the varied intermediate pose with early termination enabled
            if len(intermediate_poses_list) > 1:
                self._print(f" - Moving to intermediate pose {pose_idx + 1}/{len(intermediate_poses_list)}")
            result = self.move_to_pose(varied_intermediate_pose, close_gripper=close_gripper,
                                       enable_early_termination=True)
            
            if not result:
                return False
            
            # Validate object is still grasped if we're holding one (after each intermediate pose)
            if grasped_object_name:
                is_grasped, _ = self.check_object_grasped(self.env, grasped_object_name)
                if not is_grasped:
                    self._print(f" - Object {grasped_object_name} was dropped during move to intermediate location")
                    return False
            
            # Update current robot pose for next iteration (if there are more poses)
            if pose_idx < len(intermediate_poses_list) - 1:
                obs = self.env._get_observations()
                current_robot_pose = get_robot_poses_at_current_state(self.env, obs=obs)
        
        # then move to the target location
        self._print(" - Moving to target location")
        relative_grasp_list = self.relative_grasp_locations[to_stage]
        
        # Use modulo to ensure we always have a valid index
        selected_idx = demo_idx % len(relative_grasp_list)
        relative_grasp_info = relative_grasp_list[selected_idx]
        relative_pose = relative_grasp_info['relative_pose']
        object_name = relative_grasp_info['object_name']
        use_position_only = relative_grasp_info.get('use_position_only', False)
        is_object_target = relative_grasp_info.get('is_object_target', False)
        
        # Get current robot pose (needed for orientation if position-only, and for object-to-robot transform)
        obs = self.env._get_observations()
        current_robot_pose = get_robot_poses_at_current_state(self.env, obs=obs)
        
        # Check if this is a place action with a grasped object
        if is_object_target and grasped_object_name is not None:
            # The relative pose represents where the object should be (relative to the site)
            # We need to compute where the robot should be to place the object there
            
            # Get the site pose (the place location)
            site_pose = get_object_pose(self.env, object_name)
            
            # Apply relative pose to site to get object target position
            # Since relative_pose is position-only (orientation is NaN), we only use position
            rel_pos = relative_pose[:3]
            site_pos = site_pose[:3]
            site_rot_mat = Rotation.from_quat(site_pose[3:7], scalar_first=False).as_matrix()
            
            # Check if there's a local offset that needs to be transformed from site's local frame to world frame
            local_offset = relative_grasp_info.get('local_offset', None)
            if local_offset is not None:
                # Transform offset from site's local frame to world frame using current site pose
                world_offset = site_rot_mat @ local_offset
                # Add transformed offset to rel_pos (which is in world coordinates)
                rel_pos = rel_pos + world_offset
            
            # Apply relative position offset directly in world frame
            object_target_pos = site_pos + rel_pos
            
            # Get current grasped object pose
            current_object_pose = get_object_pose(self.env, grasped_object_name)
            current_object_pos = current_object_pose[:3]
            
            # Compute object-to-robot offset (in world frame)
            # This is the vector from object to robot: robot_pos - object_pos
            object_to_robot_offset = current_robot_pose[:3] - current_object_pos
            
            # Robot target position = object target position + object-to-robot offset
            # (to maintain the same relative position between robot and object)
            robot_target_pos = object_target_pos + object_to_robot_offset
            
            # Compute target robot orientation
            if ensure_upright_object:
                # Ensure object is held upright (z-up) using minimal rotation
                # Get current object rotation
                current_object_quat = current_object_pose[3:7]
                current_robot_quat = current_robot_pose[3:7]
                
                # Convert to rotation matrices
                current_object_rot = Rotation.from_quat(current_object_quat, scalar_first=False)
                current_robot_rot = Rotation.from_quat(current_robot_quat, scalar_first=False)
                
                # Compute grasp relative rotation: object_rotation = robot_rotation * grasp_relative_rotation
                # So: grasp_relative_rotation = inverse(robot_rotation) * object_rotation
                grasp_relative_rot = current_robot_rot.inv() * current_object_rot
                
                # Get object's current z-axis in world coordinates
                object_rot_mat = current_object_rot.as_matrix()
                object_z_axis = object_rot_mat[:, 2]  # Third column is z-axis
                
                # Find minimal rotation to align object's z-axis with world z-axis [0, 0, 1]
                world_z_axis = np.array([0.0, 0.0, 1.0])
                
                # Check if already aligned (within tolerance)
                if np.abs(np.dot(object_z_axis, world_z_axis) - 1.0) < 1e-6:
                    # Already aligned, use current orientation
                    desired_object_rot = current_object_rot
                else:
                    # Compute rotation axis (cross product) and angle
                    rotation_axis = np.cross(object_z_axis, world_z_axis)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)
                    
                    if rotation_axis_norm < 1e-6:
                        # Vectors are parallel (either same or opposite direction)
                        if np.dot(object_z_axis, world_z_axis) < 0:
                            # Opposite direction, rotate 180 degrees around any perpendicular axis
                            # Use x-axis as rotation axis
                            rotation_axis = np.array([1.0, 0.0, 0.0])
                            rotation_to_upright = Rotation.from_rotvec(np.pi * rotation_axis)
                        else:
                            # Already aligned
                            desired_object_rot = current_object_rot
                            rotation_to_upright = None
                    else:
                        rotation_axis = rotation_axis / rotation_axis_norm
                        # Compute angle between vectors
                        cos_theta = np.clip(np.dot(object_z_axis, world_z_axis), -1.0, 1.0)
                        theta = np.arccos(cos_theta)
                        # Create rotation from axis-angle
                        rotation_to_upright = Rotation.from_rotvec(theta * rotation_axis)
                    
                    if rotation_to_upright is not None:
                        # Apply this rotation to the current object orientation to get desired object orientation
                        desired_object_rot = rotation_to_upright * current_object_rot
                
                # Compute desired robot rotation: desired_object_rotation = desired_robot_rotation * grasp_relative_rotation
                # So: desired_robot_rotation = desired_object_rotation * inverse(grasp_relative_rotation)
                desired_robot_rot = desired_object_rot * grasp_relative_rot.inv()
                robot_target_quat = desired_robot_rot.as_quat(scalar_first=False)
            else:
                # Use current robot orientation (keep the same orientation when placing)
                robot_target_quat = current_robot_pose[3:7]
            
            robot_target_pose = np.concatenate([robot_target_pos, robot_target_quat])
        else:
            # Standard case: relative pose represents robot pose relative to object/site
            # Get current object/site pose in the environment
            current_object_pose = get_object_pose(self.env, object_name)
            
            # Get current robot pose if orientation is missing
            current_robot_pose_for_apply = current_robot_pose if use_position_only else None
            
            # Apply relative pose to current object position to get target robot pose
            robot_target_pose = apply_relative_pose_to_object(relative_pose, current_object_pose, current_robot_pose=current_robot_pose_for_apply)
        
        # Apply target position variation if enabled (x and y only, in global coordinates)
        # Skip variation for specific grasped/placed object combinations
        should_apply_variation = enable_target_position_variation and not self.skip_variations
        to_stage_predicate = stage_to_predicate(to_stage)
        if should_apply_variation and grasped_object_name is not None:
            # Check if this is a place action
            if to_stage_predicate[0] == 'place':
                # Get place location from predicate (predicate[2] for place actions)
                place_location = to_stage_predicate[2]
                
                # Extract base names for grasped object and place location
                grasped_base_name = get_base_object_name(grasped_object_name)
                placed_base_name = get_base_object_name(place_location)
                
                # Check if this combination matches any in the exclusion list
                if (grasped_base_name, placed_base_name) in self.no_position_variation_combinations:
                    should_apply_variation = False
        
        if should_apply_variation:
            # Get the variation range, checking for overrides based on place location
            variation_range = self.target_pos_variation_range
            if to_stage_predicate[0] == 'place':
                # Get place location from predicate (predicate[2] for place actions)
                place_location = to_stage_predicate[2]
                # Get base name for place location to match against overrides
                place_location_base_name = get_base_object_name(place_location)
                # Check if there's an override for this place location
                if place_location_base_name in self.target_pos_variation_range_overrides:
                    variation_range = self.target_pos_variation_range_overrides[place_location_base_name]
            
            # Add uniform variation to x and y coordinates only
            xy_variation = np.random.uniform(
                -variation_range,
                variation_range,
                size=2
            )
            robot_target_pose[0] += xy_variation[0]  # x
            robot_target_pose[1] += xy_variation[1]  # y
            # z remains unchanged
        
        # Check if this is a grasp action and add pregrasp pose
        # Pregrasp pose approaches the grasp at the correct orientation
        is_grasp_action = (grasped_object_name is None and 
                          to_stage_predicate[0] == 'grasp')
        
        if is_grasp_action:
            # Compute pregrasp pose: move backwards from grasp pose along gripper approach direction
            # The gripper approach direction is the negative z-axis of the gripper frame
            grasp_pos = robot_target_pose[:3]
            grasp_quat = robot_target_pose[3:7]
            
            # Convert quaternion to rotation matrix
            grasp_rot = Rotation.from_quat(grasp_quat, scalar_first=False)
            grasp_rot_mat = grasp_rot.as_matrix()
            
            # Get gripper z-axis (third column of rotation matrix)
            # This is the approach direction, so we move backwards (negative direction)
            gripper_z_axis = grasp_rot_mat[:, 2]
            
            # Compute pregrasp position: move backwards by pregrasp_offset
            pregrasp_pos = grasp_pos - self.pregrasp_offset * gripper_z_axis

            # Apply any additional world-z offset overrides for this grasp
            grasp_object_base_for_pregrasp = get_base_object_name(to_stage_predicate[1])
            from_location_base_for_pregrasp = get_base_object_name(from_location_name)
            for pregrasp_override in self.pregrasp_additional_z_offset_overrides:
                split_cond = pregrasp_override.get('base_split_condition')
                if split_cond is not None and split_cond != self.base_split_name:
                    continue
                override_object = pregrasp_override.get('object_name')
                if override_object is not None and get_base_object_name(override_object) != grasp_object_base_for_pregrasp:
                    continue
                override_from = pregrasp_override.get('from_location')
                if override_from is not None and get_base_object_name(override_from) != from_location_base_for_pregrasp:
                    continue
                pregrasp_pos[2] += pregrasp_override['z_offset']
                self._print(f"   - Applied pregrasp additional z_offset={pregrasp_override['z_offset']} for object={grasp_object_base_for_pregrasp} from_location={from_location_base_for_pregrasp}")
                break

            # Pregrasp pose has same orientation as grasp pose
            pregrasp_pose = np.concatenate([pregrasp_pos, grasp_quat])
            
            # Move to pregrasp pose first
            self._print(" - Moving to pregrasp pose")
            result = self.move_to_pose(pregrasp_pose, close_gripper=close_gripper, enable_early_termination=True)
            if not result:
                return False
        
        # Move to final target pose (grasp pose for grasp actions, place pose for place actions)
        check_stage = to_stage
        if skip_predicate_check_for_place_location and to_stage_predicate[0] == 'place':
            place_location = to_stage_predicate[2]
            normalized_place_location = (
                place_location[:-8] if place_location.endswith("_resized") else place_location
            )
            place_location_base_name = get_base_object_name(normalized_place_location)
            if (
                normalized_place_location in self.ignore_predicate_check_during_place_movement_for_location
                or place_location in self.ignore_predicate_check_during_place_movement_for_location
                or place_location_base_name in self.ignore_predicate_check_during_place_movement_for_location_base
            ):
                self._print(f"   - Skipping predicate check for place location: {place_location}")
                check_stage = None
        grasp_pos_threshold = 0.01
        if is_grasp_action:
            grasp_object_base = get_base_object_name(to_stage_predicate[1])
            if grasp_object_base in self.grasp_position_precision_override:
                grasp_pos_threshold = self.grasp_position_precision_override[grasp_object_base]
        result = self.move_to_pose(robot_target_pose, close_gripper=close_gripper, check_stage=check_stage, pos_threshold=grasp_pos_threshold)
        if not result:
            return False

        # Validate object is still grasped if we're holding one
        if grasped_object_name:
            is_grasped, _ = self.check_object_grasped(self.env, grasped_object_name)
            if not is_grasped:
                self._print(f"   - Object {grasped_object_name} was dropped during move to target location")
                return False
        
        return True

    def grasp_object(self, object_name, stage, max_grasp_steps=15):
        self._print(f" - Grasping object: {object_name} for stage: {stage}")
        
        for i in range(max_grasp_steps):
            ret = self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
            is_grasped, is_single_contact = self.check_object_grasped(self.env, object_name)
            if is_grasped:
                # If single contact grasp and not replaying actions, do additional closing steps
                if is_single_contact and self.actions_from is None:
                    self._print(f"   - Single contact grasp detected, performing additional closing steps")
                    additional_steps = 3
                    for j in range(additional_steps):
                        # Only close gripper (no position/rotation movement)
                        self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
                    
                    # Validate that grasp is still achieved after additional steps
                    is_grasped_after, _ = self.check_object_grasped(self.env, object_name)
                    if not is_grasped_after:
                        self._print(f"   - Grasp lost after additional closing steps")
                        return False
                    self._print(f"   - Grasp maintained after additional closing steps")
                
                return True

        self._print(f"   - Failed to grasp object: {object_name} for stage: {stage} after {max_grasp_steps} steps")

        return False

    def release_object(self, object_name, stage, max_release_steps=10):
        self._print(f" - Releasing object: {object_name} for stage: {stage}")
        
        for i in range(max_release_steps):
            ret = self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))  # -1.0 opens the gripper
            is_grasped, _ = self.check_object_grasped(self.env, object_name)
            if not is_grasped:
                return True

        self._print(f" - Failed to release object: {object_name} for stage: {stage} after {max_release_steps} steps")
        return False

    def move_and_grasp_object(self, from_location_name, object_name, stage, demo_idx=0):
        # Move to object location
        move_success = self.move_from_to(from_location_name, stage, demo_idx=demo_idx)
        if not move_success:
            return False
        
        # Grasp the object
        grasp_success = self.grasp_object(object_name, stage)
        if not grasp_success:
            return False
        
        # Validate that the object is actually grasped
        is_grasped, _ = self.check_object_grasped(self.env, object_name)
        if not is_grasped:
            self._print(f"   - Object {object_name} was not grasped after grasp_object call")
            return False
        
        return True

    def set_down_object(self, object_name, place_location_name, stage=None):
        """
        Move the robot down slightly before releasing to set down the object.
        Uses release_depth_place_location_offsets_world to determine the downward movement.
        
        Args:
            place_location_name: Name of the place location
            stage: Optional stage to check predicate for. If predicate succeeds during movement, returns True early.
        """
        self._print(f" - Setting down object: {object_name} at location: {place_location_name} for stage: {stage}")
        # Get the current robot pose
        obs = self.env._get_observations()
        current_robot_pose = get_robot_poses_at_current_state(self.env, obs=obs)
        
        # Get the release depth offset (negative means move down)
        # Default to default_release_depth_offset if not specified
        # Use base name for lookup to match generic names in YAML config
        place_location_base_name = get_base_object_name(place_location_name)
        release_depth_offset = self.release_depth_place_location_offsets_world.get(place_location_base_name, self.default_release_depth_offset)
        
        # Get the above place location offset that was used when placing
        # We need to move down by this amount to compensate for the higher placement
        above_offset = self.above_place_location_offsets_world.get(place_location_base_name, self.default_above_place_location_offset)
        
        # Get the grasped object offset (additive modifier based on what is being placed)
        grasped_object_base_name = get_base_object_name(object_name)
        grasped_object_offset = self.release_depth_grasped_object_offsets_world.get(grasped_object_base_name, 0)

        # Total z offset: release depth offset minus the above offset, plus any grasped object offset
        # (above_offset is positive upward, so we subtract it to move down)
        z_offset = release_depth_offset - above_offset + grasped_object_offset
        
        # Create target pose: same position and orientation, but moved down by the offset
        target_pose = current_robot_pose.copy()
        target_pose[2] += z_offset  # Move down in z-direction
        
        # Move to the set-down position
        normalized_place_location_name = (
            place_location_name[:-8]
            if place_location_name.endswith("_resized")
            else place_location_name
        )
        normalized_place_location_base_name = get_base_object_name(normalized_place_location_name)
        disable_predicate_check = (
            normalized_place_location_name in self.ignore_predicate_check_during_set_down_for_location
            or place_location_name in self.ignore_predicate_check_during_set_down_for_location
            or normalized_place_location_base_name in self.ignore_predicate_check_during_set_down_for_location_base
        )
        if disable_predicate_check:
            self._print(f"   - Skipping predicate check during set_down: {place_location_name}")
        set_down_success = self.move_to_pose(
            target_pose,
            close_gripper=True,
            enable_early_termination=True,
            check_stage=None if disable_predicate_check else stage,
        )
        if not set_down_success:
            self._print(f"   - Failed to set down at location: {place_location_name}")
            return False
        
        return True

    def move_and_place_grasped_object(self, object_name, to_location_name, stage, demo_idx=0, max_fall_wait_steps=10):
        # Move to placement location with upright object orientation and target position variation
        move_success = self.move_from_to(
            object_name,
            stage,
            grasped_object_name=object_name,
            demo_idx=demo_idx,
            ensure_upright_object=True,
            enable_target_position_variation=True,
            skip_predicate_check_for_place_location=True,
        )
        if not move_success:
            self._print(f"   - Failed to move to place location: {to_location_name}")
            return False
        
        # Set down the object (move down slightly before releasing)
        set_down_success = self.set_down_object(object_name, to_location_name, stage=stage)
        if not set_down_success:
            self._print(f"   - Failed to set down object: {object_name} at location: {to_location_name}")
            return False
        
        # Release the object
        release_success = self.release_object(object_name, stage)
        if not release_success:
            self._print(f"   - Failed to release object: {object_name} for stage: {stage}")
            return False

        # Wait some number of steps for the object to fall
        for i in range(max_fall_wait_steps):
            ret = self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
            if self.check_predicate_success(self.env, stage):
                return True

        self._print(f"   - After release of object: {object_name} onto target location: {to_location_name} for stage: {stage} after {max_fall_wait_steps} steps did not result in a successful predicate check")
        return False

    def complete_stage(self, from_stage, to_stage, demo_idx=0):
        predicate = stage_to_predicate(to_stage)
        action = predicate[0]

        # Handle case where there's no previous stage (first stage when actions_from is not specified)
        if from_stage is None:
            # For the first stage, determine the starting location based on the action type
            # For grasp, open, and turnon actions, the object/location name is the starting point
            if action in ['grasp', 'open', 'turnon']:
                end_of_previous_stage = predicate[1]  # Use the object/location name as starting location
            else:
                raise ValueError(f"Cannot determine starting location for first stage {to_stage} (action: {action}) when actions_from is not specified")
        else:
            end_of_previous_stage = get_end_location_of_stage(from_stage)

        if action == 'grasp':
            success = self.move_and_grasp_object(end_of_previous_stage, predicate[1], to_stage, demo_idx=demo_idx)
            if not success:
                self._print(f"   - Failed to complete stage: {to_stage}")
            return success
        elif action == 'place':
            assert end_of_previous_stage == predicate[1], "The end of the previous stage must be the same as the object to place"
            success = self.move_and_place_grasped_object(predicate[1], predicate[2], to_stage, demo_idx=demo_idx)
            if not success:
                self._print(f"   - Failed to complete stage: {to_stage}")
            # Verify placement using predicate check
            if success:
                success = self.check_predicate_success(self.env, to_stage)
                if not success:
                    self._print(f"   - Placement predicate check failed for stage: {to_stage}")
            return success
        else:
            raise NotImplementedError("Not implemented")
    
    def copy_actions_from_hdf5(self, demo_idx):
        """Copy actions from existing hdf5 demonstration data."""
        self._print(f"Copying actions from hdf5 file: {os.path.basename(self.actions_from_hdf5_path)}")
        # Load hdf5 file
        with h5py.File(self.actions_from_hdf5_path, "r") as f:
            num_demos = len(f["data"].keys())
            demo_key = f"demo_{demo_idx % num_demos}"
            demo_group = f["data"][demo_key]
            actions = demo_group["actions"][:]
            dataset_init_state = demo_group["states"][0]
        
        # Reset environment to initial state
        set_sim_state_from_flattened_with_update(self.env, dataset_init_state)
        self.env._start_new_episode() # marks the data collection wrapper to start the episode using the state we just set as the initial state
        
        # Track the episode directory for this demo (now that episode has started)
        if hasattr(self.env, 'ep_directory') and self.env.ep_directory is not None:
            ep_dir_name = os.path.basename(self.env.ep_directory)
            # Get the demo_idx from task_metadata (set in generate_single_demo)
            demo_idx = self.task_metadata.get('demo_idx')
            if demo_idx is not None:
                self.ep_directory_to_demo_idx[ep_dir_name] = demo_idx
        
        # Let environment settle
        settle_steps = 5
        zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        for i in range(settle_steps):
            self.env.step(zero_action)
        
        # Replay actions until all steps are executed
        self._print(f" - Replaying up to {len(actions)} actions from hdf5 until stage completed")
        current_stage_i = 0
        step = 0
        while step < len(actions):
            action = actions[step]
            ret = self.env.step(action)
            
            # Check if current stage is a "Touch" stage - skip predicate check for touch stages
            current_stage = self.actions_from_steps[current_stage_i]
            predicate = stage_to_predicate(current_stage)
            is_touch_stage = predicate[0] == 'touch'
            
            if is_touch_stage:
                # Skip predicate check for touch stages and move to next stage
                # Assert that there is a next stage
                assert current_stage_i + 1 < len(self.actions_from_steps), \
                    f"Touch stage '{current_stage}' is the last stage in actions_from_steps, but touch stages must be followed by another stage"
                current_stage_i += 1
                self._print(f" - Skipped predicate check for touch stage '{current_stage}', moved to next stage")
                step += 1
            elif self.check_predicate_success(self.env, current_stage):
                # Check if we need to replay additional steps after this stage
                # Match against the actions_from task name, not the current task name
                additional_steps = None
                for config in self.additional_replay_steps_after_stage:
                    if config["stage"] != current_stage:
                        continue
                    base_split_filter = config.get("base_split_condition")
                    if base_split_filter is not None and base_split_filter != self.base_split_name:
                        continue
                    task_name_filter = config.get("task_name")
                    if task_name_filter is not None and task_name_filter != self.actions_from:
                        continue
                    additional_steps = config["additional_steps"]
                    break
                
                # If additional steps are configured, replay them without checking predicates
                if additional_steps is not None:
                    assert current_stage_i == len(self.actions_from_steps) - 1, \
                        f"additional_replay_steps_after_stage is only supported for the last stage in actions_from_steps, " \
                        f"but stage '{current_stage}' is at index {current_stage_i} of {len(self.actions_from_steps) - 1}"
                    self._print(f" - Stage '{current_stage}' predicate satisfied, replaying {additional_steps} additional steps")
                    # Replay the specified number of additional steps
                    for _ in range(additional_steps):
                        step += 1
                        if step < len(actions):
                            ret = self.env.step(actions[step])
                        else:
                            # If we've run out of actions, break
                            self._print(f" - Warning: Ran out of actions while replaying additional steps")
                            break
                
                current_stage_i += 1
                if current_stage_i == len(self.actions_from_steps):
                    self._print(f" - Completed replaying actions from hdf5 after {step+1} steps of {len(actions)} from the task")
                    return True
                step += 1
            else:
                step += 1
        self._print(f" - Warning: Did not complete all stages from hdf5 replay")
        return False
    
    def generate_single_demo(self, demo_idx):
        """Generate a single demonstration.
        
        Returns:
            bool: True if demonstration was successful, False otherwise
        """
        self.task_metadata['demo_idx'] = demo_idx
        self._print(f"\n{'='*80}")
        self._print(f"Generating demo {demo_idx}")
        self._print(f"{'='*80}")
        
        # sometimes the reset fails so keep trying until it succeeds
        reset_success = False
        max_tries_left = 10
        seed_used = demo_idx  # Use demo_idx as the seed
        while not reset_success:
            try:
                self.env.seed(seed_used)  # Seed the environment for consistent object positions
                self.env.reset()
                reset_success = True
            except:
                max_tries_left -= 1
                if max_tries_left == 0:
                    self._print(f" - Failed to reset environment after 10 retries for demo {demo_idx}")
                    self._mark_current_demo_as_failed()
                    return False
                continue
        
        # Store the seed used for this demo
        self.demo_seeds[demo_idx] = seed_used
        
        # Track the episode directory for this demo (will be set after first step)
        # We'll update this in copy_actions_from_hdf5 or after the first step

        # Check if the final success predicate is already met - if so, fail the demonstration
        # We don't want data marked as success if it's already done at the start
        final_step = self.execution_steps[-1]
        if self.check_predicate_success(self.env, final_step):
            self._print(f" - Final success predicate '{final_step}' is already met at the start of demo {demo_idx}. Failing demonstration.")
            self._mark_current_demo_as_failed()
            return False

        # first copy steps from existing demonstration data (if actions_from is specified)
        if self.actions_from is not None:
            assert self.execution_steps[:len(self.actions_from_steps)] == self.actions_from_steps, "The steps from the existing demonstration data do not match the steps from the task"
            success = self.copy_actions_from_hdf5(demo_idx)
            if not success:
                self._print(f" - Failed to copy actions from hdf5 for demo {demo_idx}")
                self._mark_current_demo_as_failed()
                return False
            # complete the remaining steps
            remaining_steps = self.execution_steps[len(self.actions_from_steps):]
            last_stage_after_replay = self.execution_steps[len(self.actions_from_steps) - 1]
        else:
            # No actions_from, so all steps need to be completed
            # Start the episode for data collection
            self.env._start_new_episode()
            
            # Track the episode directory for this demo (now that episode has started)
            if hasattr(self.env, 'ep_directory') and self.env.ep_directory is not None:
                ep_dir_name = os.path.basename(self.env.ep_directory)
                self.ep_directory_to_demo_idx[ep_dir_name] = demo_idx
            
            remaining_steps = self.execution_steps
            last_stage_after_replay = None

        # complete the remaining steps
        for i, cur_stage in enumerate(remaining_steps):
            if i == 0:
                if last_stage_after_replay is not None:
                    last_stage = last_stage_after_replay
                else:
                    # No previous stage, start from beginning
                    # For the first step, we need to determine the starting location
                    # This depends on the action type - for now, use a placeholder
                    last_stage = None  # Will be handled by complete_stage
            else:
                last_stage = remaining_steps[i - 1]
            
            # Add newline between stages
            if i > 0:
                self._print()
            self._print(f"Stage {i+1}/{len(remaining_steps)}: {cur_stage}")
            
            stage_success = self.complete_stage(last_stage, cur_stage, demo_idx=demo_idx)
            if not stage_success:
                self._print(f" - Failed at stage {cur_stage} for demo {demo_idx}")
                self._mark_current_demo_as_failed()
                return False
        
        # take 10 additional steps to settle the environment
        self._print("\nSettling environment...")
        found_done = False
        for i in range(10):
            _, _, done, _ = self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))

            if done:
                found_done = True
                self._print(f" - Settled environment after {i + 1} steps for demo {demo_idx}")
                break
        
        if not found_done:
            self._print(f" - Failed to settle environment after {i + 1} steps for demo {demo_idx}")
            self._print_goal_predicate_status()
            self._mark_current_demo_as_failed()
            return False

        # take 20 additional steps (@20Hz so 1 additional second) to add some padding to the end of the demonstration since it's nice to see the object settle in the demonstrations
        self._print("\nAdding padding to the end of the demonstration...")
        for i in range(20):
            _, _, done, _ = self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        if not done:
            self._print(f" - Failed to add padding to the end of the demonstration after 20 steps for demo {demo_idx}")
            self._print_goal_predicate_status()
            self._mark_current_demo_as_failed()
            return False
        
        # Check if any placed objects (or their place locations) have fallen over
        # Iterate through all place operations, not just the last one
        for step in self.execution_steps:
            step_predicate = stage_to_predicate(step)
            if step_predicate[0] == 'place':
                placed_object_name = step_predicate[1]
                place_location_name = step_predicate[2]
                
                # Determine which object to check: placed object or place location
                placed_object_base_name = get_base_object_name(placed_object_name)
                place_location_base_name = get_base_object_name(place_location_name)
                check_object_name = None
                check_object_type = None

                # Skip fall check for specific (object, location) pairs where angled placement is expected
                skip_locations = self.skip_fall_check_when_placed_on.get(placed_object_base_name, [])
                if place_location_base_name in skip_locations:
                    self._print(f" - Skipping fall check for {placed_object_name} on {place_location_name} (configured in skip_fall_check_when_placed_on)")
                    continue

                if place_location_base_name in self.check_place_location_fall_instead_of_placed_object:
                    # Check the place location instead of the placed object
                    check_object_name = place_location_name
                    check_object_type = "place location"
                    self._print(f" - Checking place location {place_location_name} for fall instead of placed object {placed_object_name} (configured in check_place_location_fall_instead_of_placed_object)")
                else:
                    # Check the placed object (default behavior)
                    check_object_name = placed_object_name
                    check_object_type = "placed object"
                
                # Get the object's current pose
                try:
                    object_pose = get_object_pose(self.env, check_object_name)
                except (KeyError, AttributeError):
                    # Object might not exist or might be a site (which doesn't have rotation)
                    # Skip this check if we can't get the pose
                    self._print(f" - Warning: Could not get pose for {check_object_type} {check_object_name}, skipping fall check")
                    continue
                
                object_quat = object_pose[3:7]
                
                # Convert to rotation matrix and get z-axis
                object_rot = Rotation.from_quat(object_quat, scalar_first=False)
                object_rot_mat = object_rot.as_matrix()
                object_z_axis = object_rot_mat[:, 2]  # Third column is z-axis
                
                # Compute angle between object's z-axis and world z-axis [0, 0, 1]
                world_z_axis = np.array([0.0, 0.0, 1.0])
                cos_theta = np.clip(np.dot(object_z_axis, world_z_axis), -1.0, 1.0)
                theta = np.arccos(cos_theta)  # Angle in radians
                
                # Check if the angle exceeds the threshold
                if theta > self.object_fall_angle_threshold_rad:
                    angle_degrees = np.rad2deg(theta)
                    self._print(f" - {check_object_type.capitalize()} {check_object_name} has fallen over: {angle_degrees:.1f} degrees from upright (threshold: {self.object_fall_angle_threshold_degrees} degrees) for demo {demo_idx}")
                    if check_object_type == "place location":
                        self._print(f"   (Checked place location because {place_location_name} is in check_place_location_fall_instead_of_placed_object)")
                    self._mark_current_demo_as_failed()
                    return False
        
        self._print(f"Demo {demo_idx} completed successfully!")
        return True
    
    def _mark_current_demo_as_failed(self):
        """Mark the current episode directory as failed so it will be excluded from the final dataset."""
        current_ep_directory = self._get_current_episode_directory()
        if current_ep_directory:
            self.failed_demo_directories.append(current_ep_directory)
    
    def _get_current_episode_directory(self):
        """Get the current episode directory from the DataCollectionWrapper."""
        assert self.env.ep_directory is not None
        # Extract just the directory name (not full path)
        return os.path.basename(self.env.ep_directory)
    
    def finalize(self, generate_video=False, run_dir=None, overall_success=True, demo_results=None):
        """Finalize the demonstration generation by gathering data and creating the final dataset.
        
        Args:
            generate_video: Whether to generate video
            run_dir: Run directory for output
            overall_success: Whether the overall BDDL generation succeeded (all required demos generated)
            demo_results: List of (demo_idx, success) tuples to save to text file
        """
        self.env.close() # flush out the data collection wrapper

        # Determine output directories based on overall BDDL success
        if overall_success:
            # Overall BDDL succeeded: use normal directories
            # keep_failures only applies when overall BDDL succeeds
            remove_directory = [] if self.keep_failures else self.failed_demo_directories
            datasets_out_dir = os.path.join(run_dir, "datasets")
            video_base_dir = os.path.join(run_dir, "videos")
            demo_results_dir = os.path.join(run_dir, "demo_results")
        else:
            # Overall BDDL failed: save to failed directories
            # Always include all demos (including failures) when overall BDDL fails
            remove_directory = []
            datasets_out_dir = os.path.join(run_dir, "datasets_failed")
            video_base_dir = os.path.join(run_dir, "videos_failed")
            demo_results_dir = os.path.join(run_dir, "demo_results_failed")
        
        # Create mapping from directory to seed for gathering
        directory_to_seed = {}
        for ep_dir_name, demo_idx in self.ep_directory_to_demo_idx.items():
            if demo_idx in self.demo_seeds:
                directory_to_seed[ep_dir_name] = self.demo_seeds[demo_idx]
        gather_demonstrations_as_hdf5(
            self.tmp_directory, self.tmp_out_directory, self.env_info, self.problem_info, self.bddl_file_path,
            remove_directory=remove_directory, directory_to_seed=directory_to_seed
        )
        
        # create the final hdf5 file that contains the states, actions, image observations, and the rest of the info
        os.makedirs(datasets_out_dir, exist_ok=True)
        intermediate_hdf5_path = os.path.join(self.tmp_out_directory, "demo.hdf5")
        # skip_done_check should only be True when overall BDDL fails OR keep_failures is True
        skip_done_check = not overall_success or self.keep_failures
        final_hdf5_path = create_dataset(intermediate_hdf5_path, True, False, False, self.resolution, skip_done_check=skip_done_check, output_dir=datasets_out_dir)
        
        # If overall_success is True, delete any corresponding failed dataset and video from previous runs
        if overall_success:
            # Get the paths for the failed dataset and video
            _, failed_dataset_path = get_expected_dataset_paths(self.bddl_file_path, run_dir)
            
            # Delete failed dataset if it exists
            if os.path.exists(failed_dataset_path):
                try:
                    os.remove(failed_dataset_path)
                    self._print(f"Deleted previous failed dataset: {failed_dataset_path}")
                except Exception as e:
                    self._print(f"Warning: Could not delete failed dataset {failed_dataset_path}: {e}")
        
        # Generate video if requested
        video_path = None
        if generate_video:
            # Construct video path consistently for both success and failure cases
            # Extract split name and task name from hdf5 path
            split_name = os.path.basename(os.path.dirname(final_hdf5_path))
            dataset_name = os.path.basename(final_hdf5_path).replace('.hdf5', '')
            video_path = os.path.join(video_base_dir, split_name, dataset_name + '.mp4')
            # Generate video by passing full path as output_dir (function detects .mp4 extension)
            video_path = generate_video_from_hdf5(final_hdf5_path, output_dir=video_path, enable_print=not self.suppress_print)
            
            # If overall_success is True, delete any corresponding failed video from previous runs
            if overall_success:
                # Construct the failed video path
                failed_video_path = os.path.join(run_dir, "videos_failed", split_name, dataset_name + '.mp4')
                
                # Delete failed video if it exists
                if os.path.exists(failed_video_path):
                    try:
                        os.remove(failed_video_path)
                        self._print(f"Deleted previous failed video: {failed_video_path}")
                    except Exception as e:
                        self._print(f"Warning: Could not delete failed video {failed_video_path}: {e}")
        
        # Save demo results to text file
        if demo_results is not None:
            # Extract split name and task name from bddl_file_path
            split_name = os.path.basename(os.path.dirname(self.bddl_file_path))
            task_name = os.path.basename(self.bddl_file_path).replace(".bddl", "")
            
            # Create demo_results directory
            os.makedirs(demo_results_dir, exist_ok=True)
            
            # Create subdirectory for split
            split_demo_results_dir = os.path.join(demo_results_dir, split_name)
            os.makedirs(split_demo_results_dir, exist_ok=True)
            
            # Create text file with checkmarks/x marks on a single line
            results_file_path = os.path.join(split_demo_results_dir, f"{task_name}.txt")
            
            # Sort demo_results by demo_idx to ensure consistent ordering
            sorted_demo_results = sorted(demo_results, key=lambda x: x[0])
            
            # Create line with checkmarks (✓) or x marks (✗)
            marks = []
            for demo_idx, success in sorted_demo_results:
                marks.append("✓" if success else "✗")
            marks_line = " ".join(marks)
            
            # Write to file
            with open(results_file_path, "w") as f:
                f.write(marks_line + "\n")
        
        return final_hdf5_path, video_path


def get_expected_dataset_path(bddl_file_path, run_dir):
    """
    Compute the expected dataset path for a BDDL file based on how create_dataset names files.
    
    Args:
        bddl_file_path: Full path to the BDDL file
        run_dir: Run directory where datasets are stored
        
    Returns:
        Expected dataset path
    """
    # Extract split name from bddl_file_path
    # Path structure: .../bddl_files{suffix}/{split_name}/{task_name}.bddl
    # Use basename of directory to get split name (works regardless of suffix)
    bddl_file_dir = os.path.dirname(bddl_file_path)
    split_name = os.path.basename(bddl_file_dir)
    
    # Get task name (basename without .bddl extension)
    task_name = os.path.basename(bddl_file_path).replace(".bddl", "")
    
    # Construct dataset path: {run_dir}/datasets/{split_name}/{task_name}_demo.hdf5
    datasets_dir = os.path.join(run_dir, "datasets")
    dataset_path = os.path.join(datasets_dir, split_name, f"{task_name}_demo.hdf5")
    
    return dataset_path

def get_expected_dataset_paths(bddl_file_path, run_dir):
    """
    Get both expected dataset paths (success and failed) for a BDDL file.
    
    Args:
        bddl_file_path: Full path to the BDDL file
        run_dir: Run directory where datasets are stored
        
    Returns:
        Tuple of (success_path, failed_path)
    """
    # Extract split name from bddl_file_path
    # Use basename of directory to get split name (works regardless of suffix)
    bddl_file_dir = os.path.dirname(bddl_file_path)
    split_name = os.path.basename(bddl_file_dir)
    
    # Get task name (basename without .bddl extension)
    task_name = os.path.basename(bddl_file_path).replace(".bddl", "")
    
    # Construct both paths
    success_path = os.path.join(run_dir, "datasets", split_name, f"{task_name}_demo.hdf5")
    failed_path = os.path.join(run_dir, "datasets_failed", split_name, f"{task_name}_demo.hdf5")
    
    return success_path, failed_path

def print_dataset_status(run_dir, include_splits=None, bddl_path=None, suffix=None):
    """
    Print the status of which datasets are present and which are not.
    Shows ALL BDDL files in the specified splits, regardless of filtering options.
    
    Args:
        run_dir: Run directory where datasets are stored
        include_splits: List of split names to check, or None to check all splits ending with _extra or _view
        bddl_path: Optional path to bddl_files directory. If provided, uses this path instead of suffix.
        suffix: Optional suffix to append to bddl_files base path. Used only if bddl_path is None.
    """
    if bddl_path is None:
        bddl_path = get_libero_path_with_suffix("bddl_files", suffix)
    
    if include_splits is not None:
        split_names = include_splits
    else:
        split_names = [x for x in os.listdir(bddl_path) if x.endswith("_extra") or x.endswith("_view")]
    
    # Build view mappings so we can collapse view tasks and their source tasks
    # into a single "logical task" for summary statistics.
    try:
        view_mappings, _source_splits, _base_source_splits, _view_splits = process_view_splits(split_names, bddl_path)
    except Exception:
        view_mappings = {}
    
    print("\n" + "=" * 80)
    print(f"Dataset Status for run directory: {run_dir}")
    print("=" * 80)
    
    # Track per-split summaries (for the detailed breakdown) and also
    # aggregate over "logical tasks" for the top-level summary.
    split_summaries = {}  # Store summaries for each split
    logical_status = {}   # (logical_split, logical_task_name) -> "present" | "failed" | "missing"
    status_priority = {"missing": 0, "failed": 1, "present": 2}
    
    for split_name in sorted(split_names):
        split_path = os.path.join(bddl_path, split_name)
        if not os.path.exists(split_path):
            continue
        
        bddl_files_list = sorted([f for f in os.listdir(split_path) if f.endswith(".bddl")])
        
        split_present = []
        split_missing = []
        split_failed = []
        
        # Check all BDDL files in the split (no filtering)
        for bddl_file in bddl_files_list:
            bddl_file_name = os.path.splitext(bddl_file)[0]
            bddl_file_path = os.path.join(split_path, bddl_file)
            success_path, failed_path = get_expected_dataset_paths(bddl_file_path, run_dir)
            
            # Check for demo_results file
            demo_results_success_path = os.path.join(run_dir, "demo_results", split_name, f"{bddl_file_name}.txt")
            demo_results_failed_path = os.path.join(run_dir, "demo_results_failed", split_name, f"{bddl_file_name}.txt")
            demo_results_path = None
            demo_results_info = None
            
            if os.path.exists(demo_results_success_path):
                demo_results_path = demo_results_success_path
            elif os.path.exists(demo_results_failed_path):
                demo_results_path = demo_results_failed_path
            
            # Parse demo_results file if it exists
            if demo_results_path:
                try:
                    with open(demo_results_path, "r") as f:
                        content = f.read().strip()
                    marks = content.split()
                    total_demos = len(marks)
                    successful_demos = sum(1 for mark in marks if mark == "✓")
                    failed_demos = total_demos - successful_demos
                    success_rate = (successful_demos / total_demos * 100) if total_demos > 0 else 0
                    demo_results_info = f"({successful_demos}/{total_demos} success/total, {success_rate:.1f}%)"
                except Exception as e:
                    demo_results_info = f"(error reading: {e})"
            
            # Determine the "logical task" key: for view splits that map to a
            # source split / task, we use the source; otherwise we use the
            # current split / task directly.
            logical_key = (split_name, bddl_file_name)
            if view_mappings and split_name in view_mappings:
                task_mapping = view_mappings[split_name].get(bddl_file_name)
                if task_mapping is not None:
                    source_split, source_task_name, _is_base_split = task_mapping
                    logical_key = (source_split, source_task_name)
            
            if os.path.exists(success_path):
                split_present.append((bddl_file_name, demo_results_info))
                status = "present"
            elif os.path.exists(failed_path):
                split_failed.append((bddl_file_name, demo_results_info))
                status = "failed"
            else:
                split_missing.append((bddl_file_name, demo_results_info))
                status = "missing"
            
            # Update logical status with precedence: present > failed > missing.
            prev_status = logical_status.get(logical_key)
            if prev_status is None or status_priority[status] > status_priority[prev_status]:
                logical_status[logical_key] = status
        
        # Print status for this split
        if split_present or split_missing or split_failed:
            print(f"\n{split_name}:")
            print("-" * 80)
            if split_present:
                print(f"  ✓ Present ({len(split_present)}):")
                for bddl_name, demo_info in sorted(split_present):
                    if demo_info:
                        print(f"    - {bddl_name} {demo_info}")
                    else:
                        print(f"    - {bddl_name}")
            if split_failed:
                print(f"  ✗ Failed ({len(split_failed)}):")
                for bddl_name, demo_info in sorted(split_failed):
                    if demo_info:
                        print(f"    - {bddl_name} {demo_info}")
                    else:
                        print(f"    - {bddl_name}")
            if split_missing:
                print(f"  ○ Missing ({len(split_missing)}):")
                for bddl_name, demo_info in sorted(split_missing):
                    if demo_info:
                        print(f"    - {bddl_name} {demo_info}")
                    else:
                        print(f"    - {bddl_name}")
            
            # Store summary for this split
            split_summaries[split_name] = {
                'present': len(split_present),
                'failed': len(split_failed),
                'missing': len(split_missing),
                'total': len(split_present) + len(split_failed) + len(split_missing)
            }
    
    # Compute and print overall summary over UNIQUE logical tasks.
    total_present = sum(1 for s in logical_status.values() if s == "present")
    total_failed = sum(1 for s in logical_status.values() if s == "failed")
    total_missing = sum(1 for s in logical_status.values() if s == "missing")
    
    print("\n" + "=" * 80)
    print("Summary (unique logical tasks):")
    print(f"  ✓ Present: {total_present}")
    print(f"  ✗ Failed: {total_failed}")
    print(f"  ○ Missing: {total_missing}")
    print(f"  Total: {total_present + total_failed + total_missing}")
    print("=" * 80)
    
    # Print summary by split
    if split_summaries:
        print("\nSummary by Split:")
        print("-" * 80)
        for split_name in sorted(split_summaries.keys()):
            summary = split_summaries[split_name]
            print(f"\n{split_name}:")
            print(f"  ✓ Present: {summary['present']}")
            print(f"  ✗ Failed: {summary['failed']}")
            print(f"  ○ Missing: {summary['missing']}")
            print(f"  Total: {summary['total']}")
        print("=" * 80)
    
    print()

def collect_base_splits_from_metadata(bddl_path, split_names):
    """
    Collect base split names from metadata files of extra/view splits.
    
    Args:
        bddl_path: Path to the source BDDL files directory
        split_names: List of split names (extra/view splits)
    
    Returns:
        Set of base split names that need to be backed up
    """
    base_splits = set()
    
    for split_name in split_names:
        split_metadata_path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
        if os.path.exists(split_metadata_path):
            try:
                with open(split_metadata_path, "r") as f:
                    split_metadata = yaml.safe_load(f)
                base_split_name = split_metadata.get("base_split_name")
                if base_split_name:
                    base_splits.add(base_split_name)
            except Exception as e:
                print(f"Warning: Could not load metadata from {split_metadata_path}: {e}")
    
    return base_splits

def process_view_splits(split_names, bddl_path):
    """
    Process view splits to extract source splits and create mappings.
    
    Args:
        split_names: List of split names (may include view splits ending with _view)
        bddl_path: Path to the BDDL files directory
    
    Returns:
        tuple: (view_mappings, source_splits, base_source_splits, view_splits)
            - view_mappings: Dict mapping {view_split: {view_task_name: (source_split, source_task_name, is_base_split)}}
            - source_splits: Set of all non-base source splits that need to be processed
            - base_source_splits: Set of all base source splits (for symlinking only, not generation)
            - view_splits: List of view splits that were found
    """
    view_mappings = {}
    source_splits = set()
    base_source_splits = set()
    view_splits = []
    
    for split_name in split_names:
        if not split_name.endswith("_view"):
            continue
        
        view_splits.append(split_name)
        split_metadata_path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
        
        if not os.path.exists(split_metadata_path):
            print(f"Warning: View split {split_name} has no task_metadata.yaml, skipping")
            continue
        
        try:
            with open(split_metadata_path, "r") as f:
                split_metadata = yaml.safe_load(f)
            
            tasks_dict = split_metadata.get("tasks", {})
            view_mappings[split_name] = {}
            
            for view_task_name, task_metadata in tasks_dict.items():
                source_split = task_metadata.get("source_split")
                if source_split is None:
                    print(f"Warning: Task {view_task_name} in view {split_name} has no source_split, skipping")
                    continue
                
                # View task name should match source task name (they're the same task)
                source_task_name = view_task_name
                
                # Check if source split is a base split (doesn't end with _extra or _view)
                is_base_split = not (source_split.endswith("_extra") or source_split.endswith("_view"))
                
                view_mappings[split_name][view_task_name] = (source_split, source_task_name, is_base_split)
                
                if is_base_split:
                    base_source_splits.add(source_split)
                else:
                    source_splits.add(source_split)
        
        except Exception as e:
            print(f"Warning: Could not process view split {split_name}: {e}")
            continue
    
    return view_mappings, source_splits, base_source_splits, view_splits

def create_view_symlinks(run_dir, view_mappings):
    """
    Create absolute symlinks from source datasets/videos/results to view locations.
    
    Args:
        run_dir: Run directory where datasets are stored
        view_mappings: Dict mapping {view_split: {view_task_name: (source_split, source_task_name, is_base_split)}}
    """
    run_dir = os.path.abspath(run_dir)
    libero_datasets_dir = get_libero_path("datasets")
    
    for view_split, task_mappings in view_mappings.items():
        for view_task_name, (source_split, source_task_name, is_base_split) in task_mappings.items():
            # Define all file types to symlink
            file_types = [
                ("datasets", f"{source_task_name}_demo.hdf5", f"{view_task_name}_demo.hdf5"),
                ("datasets_failed", f"{source_task_name}_demo.hdf5", f"{view_task_name}_demo.hdf5"),
                ("videos", f"{source_task_name}_demo.mp4", f"{view_task_name}_demo.mp4"),
                ("videos_failed", f"{source_task_name}_demo.mp4", f"{view_task_name}_demo.mp4"),
                ("demo_results", f"{source_task_name}.txt", f"{view_task_name}.txt"),
                ("demo_results_failed", f"{source_task_name}.txt", f"{view_task_name}.txt"),
            ]
            
            for dir_type, source_filename, view_filename in file_types:
                # For base splits, look in libero datasets folder; for others, look in run_dir
                if is_base_split:
                    # Base splits are in the libero datasets folder
                    # Note: videos and demo_results for base splits may not exist in libero folder
                    # Only datasets are typically in the libero folder for base splits
                    if dir_type == "datasets":
                        source_path = os.path.join(libero_datasets_dir, source_split, source_filename)
                    else:
                        # Skip videos and demo_results for base splits (they may not exist)
                        continue
                else:
                    # Non-base splits are in the run directory
                    source_path = os.path.join(run_dir, dir_type, source_split, source_filename)
                
                dest_path = os.path.join(run_dir, dir_type, view_split, view_filename)
                
                # Skip if source doesn't exist
                if not os.path.exists(source_path):
                    continue
                
                # Create parent directory if needed
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Remove existing symlink or file if it exists
                if os.path.exists(dest_path) or os.path.islink(dest_path):
                    if os.path.islink(dest_path):
                        os.remove(dest_path)
                    else:
                        # If it's a regular file, remove it (shouldn't happen, but be safe)
                        os.remove(dest_path)
                
                # Create absolute symlink
                try:
                    source_abs = os.path.abspath(source_path)
                    os.symlink(source_abs, dest_path)
                except Exception as e:
                    print(f"Warning: Could not create symlink {dest_path} -> {source_path}: {e}")

def backup_bddl_files(bddl_path, run_dir, split_names):
    """
    Backup BDDL files to the run directory.
    
    Args:
        bddl_path: Path to the source BDDL files directory
        run_dir: Run directory where backups should be stored
        split_names: List of split names to backup
    """
    import shutil
    
    # Create backup directory directly in run directory
    bddl_backup_dir = os.path.join(run_dir, "bddl_files")
    os.makedirs(bddl_backup_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Backing up BDDL files")
    print(f"{'='*60}")
    print(f"BDDL source: {bddl_path}")
    print(f"Backup destination: {run_dir}/")
    print()
    
    # Backup BDDL files for each split
    for split_name in split_names:
        source_bddl_split_dir = os.path.join(bddl_path, split_name)
        target_bddl_split_dir = os.path.join(bddl_backup_dir, split_name)
        
        if os.path.exists(source_bddl_split_dir):
            # Remove existing backup if it exists (overwrite)
            if os.path.exists(target_bddl_split_dir):
                if os.path.islink(target_bddl_split_dir):
                    os.remove(target_bddl_split_dir)
                else:
                    shutil.rmtree(target_bddl_split_dir)
            
            # Copy the entire split directory
            shutil.copytree(source_bddl_split_dir, target_bddl_split_dir)
            print(f"Backed up BDDL split: {split_name}")
        else:
            print(f"Warning: BDDL split directory does not exist: {source_bddl_split_dir}")
    
    print(f"\nBDDL files backup completed successfully!")
    print(f"{'='*60}\n")


def backup_init_files(init_files_path, run_dir, split_names):
    """
    Backup init_files to the run directory.
    
    Args:
        init_files_path: Path to the source init_files directory
        run_dir: Run directory where backups should be stored
        split_names: List of split names to backup
    """
    import shutil
    
    # Create backup directory directly in run directory
    init_files_backup_dir = os.path.join(run_dir, "init_files")
    os.makedirs(init_files_backup_dir, exist_ok=True)
    
    # Assert that for every BDDL file in the run directory, there's a corresponding init file
    bddl_backup_dir = os.path.join(run_dir, "bddl_files")
    if os.path.exists(bddl_backup_dir):
        print(f"\n{'='*60}")
        print(f"Verifying BDDL and init_files correspondence")
        print(f"{'='*60}")
        
        while True:
            missing_init_files = []
            for split_name in split_names:
                bddl_split_dir = os.path.join(bddl_backup_dir, split_name)
                init_files_split_dir = os.path.join(init_files_path, split_name)
                
                if not os.path.exists(bddl_split_dir):
                    continue  # Skip if BDDL split doesn't exist
                
                if not os.path.exists(init_files_split_dir):
                    # If init_files split doesn't exist, all BDDL files in this split are missing init files
                    bddl_files = [f for f in os.listdir(bddl_split_dir) if f.endswith('.bddl')]
                    for bddl_file in bddl_files:
                        missing_init_files.append((split_name, bddl_file))
                    continue
                
                # Check each BDDL file has a corresponding init file
                bddl_files = [f for f in os.listdir(bddl_split_dir) if f.endswith('.bddl')]
                init_files = set(os.listdir(init_files_split_dir))
                
                for bddl_file in bddl_files:
                    # Init file should have the same name as BDDL file with .pruned_init extension
                    bddl_name = os.path.splitext(bddl_file)[0]
                    expected_init_file = f"{bddl_name}.pruned_init"
                    if expected_init_file not in init_files:
                        missing_init_files.append((split_name, bddl_file))
            
            if missing_init_files:
                print(f"\nError: Found {len(missing_init_files)} BDDL file(s) without corresponding init files:")
                for split_name, bddl_file in missing_init_files:
                    print(f"  - {split_name}/{bddl_file}")
                print(f"\nEach BDDL file must have a corresponding init file. ")
                print(f"Found {len(missing_init_files)} BDDL file(s) missing init files. ")
                print(f"Please ensure all BDDL files have corresponding init files. Perhaps what happened is that you were generating bddls with gen_extra_libero_envs.py concurrently with this script. The init states are only available once get_extra_libero_envs.py is finished running and maybe it's not done yet. To solve this you can just copy over the init states into this run directory once it finishes running and you will be all set to then link in the demonstrations properly.")
                
                # Ask user if they want to try again
                while True:
                    user_input = input("\nWould you like to try the check again? (yes/no): ").strip().lower()
                    if user_input in ['yes', 'y']:
                        print("\nRetrying init file check...")
                        break
                    elif user_input in ['no', 'n']:
                        raise AssertionError(
                            f"Each BDDL file must have a corresponding init file. "
                            f"Found {len(missing_init_files)} BDDL file(s) missing init files. "
                            f"Please ensure all BDDL files have corresponding init files. Perhaps what happened is that you were generating bddls with gen_extra_libero_envs.py concurrently with this script. The init states are only available once get_extra_libero_envs.py is finished running and maybe it's not done yet. To solve this you can just copy over the init states into this run directory once it finishes running and you will be all set to then link in the demonstrations properly."
                        )
                    else:
                        print("Please enter 'yes' or 'no'.")
            else:
                print(f"✓ All BDDL files have corresponding init files")
                break
        print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"Backing up init_files")
    print(f"{'='*60}")
    print(f"Init_files source: {init_files_path}")
    print(f"Backup destination: {run_dir}/")
    print()
    
    # Backup init_files for each split
    for split_name in split_names:
        source_init_files_split_dir = os.path.join(init_files_path, split_name)
        target_init_files_split_dir = os.path.join(init_files_backup_dir, split_name)
        
        if os.path.exists(source_init_files_split_dir):
            # Remove existing backup if it exists (overwrite)
            if os.path.exists(target_init_files_split_dir):
                if os.path.islink(target_init_files_split_dir):
                    os.remove(target_init_files_split_dir)
                else:
                    shutil.rmtree(target_init_files_split_dir)
            
            # Copy the entire split directory
            shutil.copytree(source_init_files_split_dir, target_init_files_split_dir)
            print(f"Backed up init_files split: {split_name}")
        else:
            print(f"Warning: Init_files split directory does not exist: {source_init_files_split_dir}")
    
    print(f"\nInit_files backup completed successfully!")
    print(f"{'='*60}\n")

def _process_bddl_file(args_tuple):
    """Worker function for processing a single BDDL file."""
    (bddl_file_path, bddl_file_name, args_dict) = args_tuple
    video_path, successful_demos, failed_demos, success_rate, demo_results = generate_demonstrations(bddl_file_path, **args_dict)
    return bddl_file_name, video_path, successful_demos, failed_demos, success_rate, demo_results

def generate_demonstrations(bddl_file_path, n_demos=None, generate_video=False, vis_grasp_poses=False, keep_failures=False, max_grasp_poses=None, resolution=128, log_position_error_on_fail=False, start_demo_idx=0, run_dir=None, suppress_print=False, config_path=None, bddl_path=None, suffix=None, skip_variations=False, min_attempts_for_rate_check=20, min_success_rate=0.10, max_attempts=None):
    if n_demos is None:
        raise ValueError("n_demos must be provided")
    if max_attempts is not None and max_attempts < n_demos:
        raise ValueError(f"max_attempts ({max_attempts}) must be >= n_demos ({n_demos})")
    
    # Create the task demonstration generator instance (handles all setup)
    generator = TaskDemonstrationGenerator(bddl_file_path, vis_grasp_poses=vis_grasp_poses, keep_failures=keep_failures, max_grasp_poses=max_grasp_poses, resolution=resolution, log_position_error_on_fail=log_position_error_on_fail, suppress_print=suppress_print, run_dir=run_dir, config_path=config_path, bddl_path=bddl_path, suffix=suffix, skip_variations=skip_variations)
    log = generator._print

    # Generate demonstrations, tracking successful ones
    # n_demos represents the number of successful demos we want
    successful_demos = 0
    failed_demos = 0
    demo_idx = start_demo_idx
    demo_results = []  # List of (demo_idx, success) tuples to track individual demo outcomes

    with tqdm(total=n_demos, desc=f"Generating demonstrations for {os.path.basename(bddl_file_path)}", leave=False) as pbar:
        while successful_demos < n_demos:
            total_attempts = successful_demos + failed_demos
            # Print header for demo start
            log("\n" + "="*80)
            log(f"Starting Demo {demo_idx} (Success: {successful_demos}/{n_demos}, Failed: {failed_demos})")
            log("="*80)

            success = generator.generate_single_demo(demo_idx)
            demo_results.append((demo_idx, success))
            demo_idx += 1

            # Print result with clear spacing
            log("\n" + "-"*80)
            if success:
                successful_demos += 1
                pbar.update(1)
                log(f"✓ Demo {demo_idx-1} SUCCEEDED")
            else:
                failed_demos += 1
                log(f"✗ Demo {demo_idx-1} FAILED")

            # Stop if max_attempts has been reached
            total_attempts = successful_demos + failed_demos
            if max_attempts is not None and total_attempts >= max_attempts:
                log(f"  Reached max_attempts ({max_attempts}), stopping generation with {successful_demos}/{n_demos} successful demos")
                break

            # After minimum attempts, stop if success rate is too low
            if total_attempts >= min_attempts_for_rate_check:
                current_rate = successful_demos / total_attempts
                if current_rate < min_success_rate:
                    log(f"  Success rate {current_rate:.1%} ({successful_demos}/{total_attempts}) below {min_success_rate:.0%} after {total_attempts} attempts, exiting generation")
                    break
            log("-"*80 + "\n")
    
    success = successful_demos == n_demos
    log("\n" + "-"*80)
    log(f"Overall success: {success}. Generated {successful_demos} successful demonstrations. Note that {failed_demos} failures occurred during that generation process.")
    log("-"*80 + "\n")

    # Finalize (gather data, create dataset, generate video if requested)
    # Results are always saved, but to different directories based on overall success
    final_hdf5_path, video_path = generator.finalize(generate_video=generate_video, run_dir=run_dir, overall_success=success, demo_results=demo_results)
    
    # Calculate total attempts (successful + failed)
    total_attempts = successful_demos + failed_demos
    success_rate = (successful_demos / total_attempts * 100) if total_attempts > 0 else 0.0
    
    return video_path, successful_demos, failed_demos, success_rate, demo_results


def parse_chain_steps(execution_steps):
    """Split execution_steps into (first_step, second_step) using the last Grasp as boundary.

    The second step is the final Grasp+Place pair. The first step is everything before it.
    Returns ([], execution_steps) when there are no preceding steps before the last Grasp,
    i.e. the task is not chained (plain pick-and-place).
    """
    last_grasp_idx = None
    for i, step in enumerate(execution_steps):
        if step.startswith("Grasp "):
            last_grasp_idx = i
    if last_grasp_idx is None or last_grasp_idx == 0:
        return [], list(execution_steps)
    return list(execution_steps[:last_grasp_idx]), list(execution_steps[last_grasp_idx:])


def tasks_are_similar(target_first, target_second, candidate_steps, cand_meta=None, all_task_steps=None):
    """Return True if candidate_steps is similar to a chained task with target_first/target_second.

    Case 1 – candidate is also chained: first steps must match exactly AND both second steps
             pick the same object (destination may differ).
    Case 2 – candidate is not chained (plain pick-and-place): its steps must exactly match
             the target's second step (same object AND same destination) AND its init state
             must reflect the post-first-step environment, i.e. it has derived_from_chain=True
             and its derived_from_full_chain_task has first steps matching target_first and
             second steps matching target_second.
    """
    cand_first, cand_second = parse_chain_steps(candidate_steps)
    if not cand_second or not cand_second[0].startswith("Grasp "):
        return False
    target_obj = target_second[0].split(" ")[1]
    cand_obj = cand_second[0].split(" ")[1]
    if len(cand_first) == 0:
        # Case 2: non-chained, must exactly match second step AND have matching init state
        if cand_second != target_second:
            return False
        if cand_meta is None or all_task_steps is None:
            return True  # fallback: no metadata available, use old behavior
        # Check that init state reflects post-first-step environment
        if not cand_meta.get("derived_from_chain", False):
            return False
        full_chain_task = cand_meta.get("derived_from_full_chain_task")
        if full_chain_task is None:
            return False
        chain_steps = all_task_steps.get(full_chain_task)
        if chain_steps is None:
            return False
        chain_first, chain_second = parse_chain_steps(chain_steps)
        return chain_first == target_first and chain_second == target_second
    else:
        # Case 1: chained, first steps match AND second step picks same object
        return cand_first == target_first and cand_obj == target_obj


def print_chain_similarity_status(run_dir, view_name, bddl_path):
    """For each chained task in view_name, print similar tasks and their demo generation status."""
    view_metadata_path = os.path.join(bddl_path, view_name, "task_metadata.yaml")
    with open(view_metadata_path, "r") as f:
        view_metadata = yaml.safe_load(f)
    view_tasks = view_metadata.get("tasks", {})

    all_split_names = sorted(
        x for x in os.listdir(bddl_path)
        if os.path.isdir(os.path.join(bddl_path, x))
    )
    source_splits = [s for s in all_split_names if not s.endswith("_view")]
    view_splits   = [s for s in all_split_names if s.endswith("_view") and s != view_name]

    # Collect all task steps from ALL splits for looking up derived_from_full_chain_task
    all_task_steps = {}  # task_name -> execution_steps (first occurrence wins)
    for split_name in all_split_names:
        metadata_path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        for task_name, task_meta in meta.get("tasks", {}).items():
            if task_name not in all_task_steps:
                all_task_steps[task_name] = task_meta.get("execution_steps", [])

    # Collect full task metadata from source (non-view) splits only
    source_tasks = {}  # (split_name, task_name) -> task_meta dict
    for split_name in source_splits:
        metadata_path = os.path.join(bddl_path, split_name, "task_metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        for task_name, task_meta in meta.get("tasks", {}).items():
            source_tasks[(split_name, task_name)] = task_meta

    # Build reverse mapping: task_name -> sorted list of view split names that contain it
    task_in_views = {}  # task_name -> [view_split, ...]
    for vs in view_splits:
        metadata_path = os.path.join(bddl_path, vs, "task_metadata.yaml")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        for task_name in meta.get("tasks", {}).keys():
            task_in_views.setdefault(task_name, []).append(vs)

    print("\n" + "=" * 80)
    print(f"Chain Similarity Status for view: {view_name}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)
    print()
    print("Legend:")
    print("  Similar tasks: single-step or chained tasks whose demos can serve as a reference")
    print("    for this chained task. For chained candidates, the first step must match exactly")
    print("    and the second step must pick the same object. For single-step candidates, the")
    print("    action must exactly match the chain's second step AND the task must be initialized")
    print("    with the environment already reflecting the outcome of the chain's first step")
    print("    (e.g. the drawer is already open, or the object is already placed).")
    print("  Similar [only second step matches]: single-step tasks whose action matches the")
    print("    chain's second step exactly, but whose init state is a fresh/default environment")
    print("    rather than the post-first-step state. These are weaker references — the robot")
    print("    has not yet performed the first step when these demos were recorded.")
    print()

    for target_task in sorted(view_tasks.keys()):
        target_steps = view_tasks[target_task].get("execution_steps", [])
        target_first, target_second = parse_chain_steps(target_steps)

        if len(target_first) == 0:
            print(f"\n[SKIP - not chained] {target_task}")
            continue

        target_obj = target_second[0].split(" ")[1]
        target_success_path = os.path.join(run_dir, "datasets", view_name, f"{target_task}_demo.hdf5")
        target_failed_path = os.path.join(run_dir, "datasets_failed", view_name, f"{target_task}_demo.hdf5")
        if os.path.exists(target_success_path):
            target_status = "SUCCEEDED"
        elif os.path.exists(target_failed_path):
            target_status = "FAILED"
        else:
            target_status = "MISSING"
        print(f"\n{target_task} [{target_status}]")
        print(f"  First step : {target_first}")
        print(f"  Second step: {target_second}  (object: {target_obj})")
        print(f"  Similar tasks:")

        similar = []  # list of (status, kind, split_name, task_name, views_list)
        similar_second_step_only = []  # list of (status, split_name, task_name, views_list)
        for (split_name, task_name), cand_meta in sorted(source_tasks.items()):
            cand_steps = cand_meta.get("execution_steps", [])
            if list(cand_steps) == list(target_steps):
                continue

            is_full_match = tasks_are_similar(target_first, target_second, cand_steps, cand_meta=cand_meta, all_task_steps=all_task_steps)

            # Check for "second step only" match: non-chained with matching second step but
            # not a full match (init state does not reflect post-first-step environment)
            cand_first_tmp, cand_second_tmp = parse_chain_steps(cand_steps)
            is_second_step_only = (
                not is_full_match
                and len(cand_first_tmp) == 0
                and bool(cand_second_tmp)
                and cand_second_tmp[0].startswith("Grasp ")
                and cand_second_tmp == target_second
            )

            if not is_full_match and not is_second_step_only:
                continue

            # Base splits (not ending with _extra or _view) are existing datasets — always succeeded
            is_base_split = not split_name.endswith("_extra") and not split_name.endswith("_view")
            if is_base_split:
                status = "SUCCEEDED"
            else:
                success_path = os.path.join(run_dir, "datasets", split_name, f"{task_name}_demo.hdf5")
                failed_path = os.path.join(run_dir, "datasets_failed", split_name, f"{task_name}_demo.hdf5")
                if os.path.exists(success_path):
                    status = "SUCCEEDED"
                elif os.path.exists(failed_path):
                    status = "FAILED"
                else:
                    status = "MISSING"

            views_list = sorted(task_in_views.get(task_name, []))
            if is_full_match:
                kind = "chained" if len(cand_first_tmp) > 0 else "non-chained"
                similar.append((status, kind, split_name, task_name, views_list))
            else:
                similar_second_step_only.append((status, split_name, task_name, views_list))

        if not similar:
            print("    (none found)")
        else:
            categories = [
                ("SUCCEEDED", "chained"),
                ("SUCCEEDED", "non-chained"),
                ("FAILED",    "chained"),
                ("FAILED",    "non-chained"),
                ("MISSING",   "chained"),
                ("MISSING",   "non-chained"),
            ]
            groups = {cat: [] for cat in categories}
            for entry in similar:
                status, kind, split_name, task_name, views_list = entry
                groups[(status, kind)].append((split_name, task_name, views_list))

            for (status, kind) in categories:
                entries = sorted(groups[(status, kind)])  # alphabetical by (split, task)
                if not entries:
                    continue
                print(f"  --- {status} ({kind}) [{len(entries)}] ---")
                for split_name, task_name, views_list in entries:
                    views_str = f"  [views: {', '.join(views_list)}]" if views_list else ""
                    print(f"    {split_name} / {task_name}{views_str}")

        print(f"  Similar [only second step matches]:")
        if not similar_second_step_only:
            print("    (none found)")
        else:
            sso_groups = {"SUCCEEDED": [], "FAILED": [], "MISSING": []}
            for status, split_name, task_name, views_list in similar_second_step_only:
                sso_groups[status].append((split_name, task_name, views_list))
            for status in ("SUCCEEDED", "FAILED", "MISSING"):
                entries = sorted(sso_groups[status])
                if not entries:
                    continue
                print(f"  --- {status} [{len(entries)}] ---")
                for split_name, task_name, views_list in entries:
                    views_str = f"  [views: {', '.join(views_list)}]" if views_list else ""
                    print(f"    {split_name} / {task_name}{views_str}")

        counts = {}
        for (status, kind), entries in (groups if similar else {}).items():
            counts.setdefault(status, {})
            counts[status][kind] = len(entries)
        sso_counts = {s: len(v) for s, v in (sso_groups if similar_second_step_only else {}).items() if v}
        parts = []
        for status in ("SUCCEEDED", "FAILED", "MISSING"):
            c = counts.get(status, {})
            chained_n = c.get("chained", 0)
            nonchained_n = c.get("non-chained", 0)
            total = chained_n + nonchained_n
            sso_n = sso_counts.get(status, 0)
            if total > 0 or sso_n > 0:
                main_str = f"{total} {status.lower()} ({chained_n} chained, {nonchained_n} non-chained)" if total > 0 else ""
                sso_str = f"{sso_n} second-step-only {status.lower()}" if sso_n > 0 else ""
                parts.append(", ".join(filter(None, [main_str, sso_str])))
        print(f"  Counts: {', '.join(parts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-splits", nargs="+", type=str, default=None, help="If None, then include all splits ending with `_extra` or `_view`")
    parser.add_argument("--n-demos", type=int, default=50, help="Number of demonstrations to generate for each task in extra/view splits")
    parser.add_argument("--max-tasks-per-split", type=int, default=None, help="Maximum number of tasks to process per split")
    parser.add_argument("--generate-video", type=bool, default=True, help="Generate a video for the generated demonstration")
    parser.add_argument("--vis-grasp-poses", action="store_true", help="Save images of grasp poses from reference demonstrations")
    parser.add_argument("--keep-failures", action="store_true", help="Include failed demonstrations in the final dataset")
    parser.add_argument("--min-attempts-for-rate-check", type=int, default=20, help="Minimum number of attempts (success+fail) before checking success rate (default: 20)")
    parser.add_argument("--min-success-rate", type=float, default=0.20, help="Minimum success rate below which generation stops, checked after --min-attempts-for-rate-check attempts (default: 0.20)")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum number of attempts (successful + failed) per task. Must be >= --n-demos. If not set, unlimited attempts.")
    parser.add_argument("--skip-bddl-files", type=int, default=0, help="Skip the first n bddl files in each split")
    parser.add_argument("--bddl-files", nargs="+", type=str, default=None, help="Only process the specified BDDL file names (without .bddl extension)")
    parser.add_argument("--resolution", type=int, default=128, help="Image resolution for camera observations (default: 128)")
    parser.add_argument("--log-position-error-on-fail", action="store_true", help="Log detailed position and orientation errors over time when move_to_pose fails to converge")
    parser.add_argument("--start-demo-idx", type=int, default=0, help="Start demonstration generation at the given demo index (default: 0)")
    parser.add_argument("--output-dir", type=str, default="tmp_generate_demonstrations_runs", help="Base directory for output. Used to construct run_dir if --run-dir is not provided (default: tmp_generate_demonstrations_runs)")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to store results. If not specified, creates a new run directory with timestamp")
    parser.add_argument("--num-workers", type=int, default=40, help="Number of worker processes for BDDL file processing. Default: 40. Set to 1 to disable.")
    parser.add_argument("--disable-multiprocessing", action="store_true", help="Disable multiprocessing (equivalent to setting num-workers to 1)")
    parser.add_argument("--status-only", action="store_true", help="Only print the status of which datasets are present/not present, without running generation")
    parser.add_argument("--skip-status", action="store_true", help="Skip printing dataset status before and after generation")
    parser.add_argument("--chain-similarity-status", type=str, default=None, metavar="VIEW_NAME", help="For each chained task in the given view (e.g. libero_goal_chain_selected_view), print similar tasks across all splits and their demo generation status. Requires --run-dir.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to YAML configuration file. If not specified, uses generate_spec.yaml in generate_demonstrations_configs directory")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix to append to bddl_files and datasets base paths (e.g., 'test' will make paths like 'bddl_files_test' and 'datasets_test'). Underscore is added automatically.")
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up BDDL files and init_files to the run directory")
    parser.add_argument("--skip-variations", action="store_true", help="Skip rotation and positional variations for both intermediate and target positions")
    parser.add_argument("--overwrite-source-from", type=str, default=None, help="Overwrite BDDL files and init_files in run_dir by copying from source with given suffix (e.g., 'staging'). Requires --run-dir to be specified. Exits after copying.")
    args = parser.parse_args()

    if args.max_attempts is not None and args.max_attempts < args.n_demos:
        parser.error(f"--max-attempts ({args.max_attempts}) must be >= --n-demos ({args.n_demos})")

    start_time = time.time()

    # Handle disable multiprocessing flag
    if args.disable_multiprocessing:
        args.num_workers = 1
    
    # Store whether run_dir was originally provided on command line
    run_dir_provided = args.run_dir is not None
    
    # If --status-only is set, handle it early (before backup and error checks)
    # When status-only is provided, run_dir must also be provided
    if args.status_only:
        if args.run_dir is None:
            raise ValueError(
                f"Error: --run-dir must be provided when using --status-only.\n"
                f"Status check requires an existing run directory with backed up BDDL files."
            )
        
        args.run_dir = os.path.abspath(args.run_dir)
        
        # Use backed up BDDL files from run directory
        bddl_path = os.path.join(args.run_dir, "bddl_files")
        if not os.path.exists(bddl_path):
            raise ValueError(f"Run directory {args.run_dir} was provided but bddl_files backup not found at {bddl_path}. "
                           f"Please ensure the run directory contains backed up BDDL files.")
        
        # Print status and exit
        print_dataset_status(
            run_dir=args.run_dir,
            include_splits=args.include_splits,
            bddl_path=bddl_path,
            suffix=args.suffix
        )
        sys.exit(0)

    # If --chain-similarity-status is set, handle it early (before backup and error checks)
    if args.chain_similarity_status is not None:
        if args.run_dir is None:
            raise ValueError(
                f"Error: --run-dir must be provided when using --chain-similarity-status.\n"
                f"Chain similarity status requires an existing run directory with backed up BDDL files."
            )

        args.run_dir = os.path.abspath(args.run_dir)

        bddl_path = os.path.join(args.run_dir, "bddl_files")
        if not os.path.exists(bddl_path):
            raise ValueError(f"Run directory {args.run_dir} was provided but bddl_files backup not found at {bddl_path}. "
                           f"Please ensure the run directory contains backed up BDDL files.")

        print_chain_similarity_status(
            run_dir=args.run_dir,
            view_name=args.chain_similarity_status,
            bddl_path=bddl_path,
        )
        sys.exit(0)

    # Error if both run_dir and suffix are provided (suffix is not needed when using existing run directory)
    # Skip this check for status-only (already handled above)
    if run_dir_provided and args.suffix is not None:
        raise ValueError(
            f"Error: Cannot specify both --run-dir and --suffix.\n"
            f"When --run-dir is provided, the script uses the BDDL files already backed up in {args.run_dir}/bddl_files.\n"
            f"The --suffix option is only used when creating a new run directory to determine which BDDL files to use.\n"
            f"Please remove --suffix when using an existing run directory."
        )
    
    # Create or use run directory
    if args.run_dir is None:
        # Create a new run directory organized by date
        current_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = str(time.time()).replace('.', '_')
        args.run_dir = os.path.join(args.output_dir, current_date, f"run_{timestamp}")
    
    args.run_dir = os.path.abspath(args.run_dir)

    # Create run directory if it doesn't exist
    os.makedirs(args.run_dir, exist_ok=True)
    print(f"Using run directory: {args.run_dir}")
    
    # Handle --overwrite-source-from flag (must be used with --run-dir)
    # This flag will override bddl_path and init_files_path to use source paths,
    # then reuse the existing backup logic after split determination
    if args.overwrite_source_from is not None:
        if not run_dir_provided:
            raise ValueError(
                f"Error: --overwrite-source-from requires --run-dir to be specified.\n"
                f"Please provide --run-dir when using --overwrite-source-from."
            )
        
        print(f"\n{'='*60}")
        print(f"Overwriting BDDL files and init_files from source suffix: {args.overwrite_source_from}")
        print(f"{'='*60}")
        
        # Get source paths using the suffix
        source_bddl_path = get_libero_path_with_suffix("bddl_files", args.overwrite_source_from)
        source_init_files_path = get_libero_path_with_suffix("init_states", args.overwrite_source_from)
        
        # Check that source paths exist
        if not os.path.exists(source_bddl_path):
            raise ValueError(f"Source BDDL path does not exist: {source_bddl_path}")
        if not os.path.exists(source_init_files_path):
            raise ValueError(f"Source init_files path does not exist: {source_init_files_path}")
        
        # Temporarily override bddl_path and init_files_path to use source paths
        # This allows us to reuse all the existing split determination and backup logic
        bddl_path = source_bddl_path
        init_files_path = source_init_files_path

    # Determine effective BDDL path: if run_dir was provided, use backed up BDDL files from run directory
    # Otherwise, use suffix-based path
    # Skip this if --overwrite-source-from is set (paths already set above)
    if args.overwrite_source_from is None:
        if run_dir_provided:
            # Use backed up BDDL files from run directory
            bddl_path = os.path.join(args.run_dir, "bddl_files")
            if not os.path.exists(bddl_path):
                raise ValueError(f"Run directory {args.run_dir} was provided but bddl_files backup not found at {bddl_path}. "
                               f"Please ensure the run directory contains backed up BDDL files.")
            init_files_path = os.path.join(args.run_dir, "init_files")
        else:
            # Use suffix-based path (for new run directories)
            bddl_path = get_libero_path_with_suffix("bddl_files", args.suffix)
            init_files_path = get_libero_path_with_suffix("init_states", args.suffix)
    
    # Determine which splits to process
    if args.include_splits is not None:
        split_names = args.include_splits
    else:
        split_names = [x for x in os.listdir(bddl_path) if x.endswith("_extra") or x.endswith("_view")]
    
    # Process view splits to extract source splits
    view_mappings, source_splits, base_source_splits, view_splits = process_view_splits(split_names, bddl_path)
    
    # Filter out view splits from split_names (they'll be handled via source splits)
    # Add non-base source splits to the processing list (if not already present)
    # Base source splits are not added - they will be symlinked from libero datasets folder
    split_names = [s for s in split_names if not s.endswith("_view")]
    for source_split in source_splits:
        if source_split not in split_names:
            split_names.append(source_split)
    
    # Print info about base splits that will be symlinked
    if base_source_splits:
        print(f"\nNote: View splits reference base splits {sorted(base_source_splits)}. "
              f"These will be symlinked from libero datasets folder (no generation needed).")
    
    # If --overwrite-source-from is set, backup and exit (reuse existing backup functions)
    if args.overwrite_source_from is not None:
        # Collect base splits needed for metadata loading
        base_splits = collect_base_splits_from_metadata(bddl_path, split_names)
        
        # Combine extra/view splits with their base splits for backup
        all_splits_to_backup = list(split_names) + list(base_splits) + view_splits
        
        if base_splits:
            print(f"\nIncluding base splits in backup: {sorted(base_splits)}")
        if view_splits:
            print(f"Including view splits in backup: {sorted(view_splits)}")
        
        # Use existing backup functions with source paths
        # For BDDL files, include base splits (needed for metadata)
        backup_bddl_files(bddl_path, args.run_dir, all_splits_to_backup)
        # For init files, exclude base splits (same as normal backup flow)
        all_splits_for_init_backup = list(split_names) + view_splits
        backup_init_files(init_files_path, args.run_dir, all_splits_for_init_backup)
        
        print(f"\n{'='*60}")
        print(f"Overwrite completed successfully!")
        print(f"BDDL files copied from: {bddl_path}")
        print(f"Init files copied from: {init_files_path}")
        print(f"Destination: {args.run_dir}")
        print(f"{'='*60}\n")
        
        # Exit after copying
        sys.exit(0)
    
    # Store original init_files_path for backup at the end
    original_init_files_path = init_files_path
    
    # Backup BDDL files to run directory (unless --no-backup is specified or run_dir was provided)
    # If run_dir was provided, we assume backup already exists, so skip backup
    # Note: init_files backup happens at the end after all demonstrations are generated
    if not args.no_backup and not run_dir_provided:
        # Collect base splits needed for metadata loading
        base_splits = collect_base_splits_from_metadata(bddl_path, split_names)
        
        # Combine extra/view splits with their base splits for backup
        all_splits_to_backup = list(split_names) + list(base_splits) + view_splits
        
        if base_splits:
            print(f"\nIncluding base splits in backup: {sorted(base_splits)}")
        if view_splits:
            print(f"Including view splits in backup: {sorted(view_splits)}")
        
        backup_bddl_files(bddl_path, args.run_dir, all_splits_to_backup)
        # After backup, update bddl_path to point to the run folder
        bddl_path = os.path.join(args.run_dir, "bddl_files")
    elif args.no_backup:
        print(f"\nSkipping backup (--no-backup flag specified)")
        print(f"Note: Linking will not be available for this run directory\n")

    # Validate that no base splits are included (base splits don't end with _extra or _view)
    # Note: view splits are already filtered out above, so we only check for base splits here
    base_splits = [s for s in split_names if not (s.endswith("_extra") or s.endswith("_view"))]
    if base_splits:
        raise ValueError(
            f"Error: Cannot generate data for base splits. Base splits detected: {base_splits}\n"
            f"Only splits ending with '_extra' or '_view' are allowed. "
            f"Base splits (like 'libero_goal', 'libero_spatial', 'libero_10', 'libero_90', 'libero_object') "
            f"should not be regenerated as they contain the original training data."
        )

    # If --run-dir was provided on command line, print status before generation
    if run_dir_provided and not args.skip_status:
        print("\n" + "=" * 80)
        print("Status BEFORE generation:")
        print("=" * 80)
        print_dataset_status(
            run_dir=args.run_dir,
            include_splits=args.include_splits,
            bddl_path=bddl_path,
            suffix=args.suffix
        )

    # Track video paths and BDDL names
    video_info = []
    success_stats = {}  # Dictionary to track success stats per BDDL file: {bddl_name: (successful_demos, failed_demos, success_rate, demo_results, video_path)}
    
    # Build mapping of source splits to view tasks (for filtering when view splits are provided)
    # This maps: source_split -> set of task names that are in views
    # Only apply filtering if:
    # 1. User explicitly provided splits (not auto-discovered), AND
    # 2. The source split was NOT explicitly provided
    # (i.e., it was added automatically from a view)
    # If user didn't provide any splits, generate all tasks in all splits (no filtering)
    explicitly_provided_splits = set(args.include_splits) if args.include_splits else set()
    source_split_to_view_tasks = {}
    if view_mappings:
        # Only filter if user explicitly provided splits (not auto-discovered)
        user_provided_splits = args.include_splits is not None
        for view_split, task_mappings in view_mappings.items():
            for view_task_name, (source_split, source_task_name, is_base_split) in task_mappings.items():
                if not is_base_split:  # Only filter non-base splits (base splits are symlinked, not generated)
                    # Only filter if:
                    # 1. User explicitly provided splits (not auto-discovered), AND
                    # 2. Source split wasn't explicitly provided
                    # If user didn't provide any splits, generate all tasks in all splits
                    if user_provided_splits and source_split not in explicitly_provided_splits:
                        if source_split not in source_split_to_view_tasks:
                            source_split_to_view_tasks[source_split] = set()
                        source_split_to_view_tasks[source_split].add(source_task_name)
    
    # First pass: Collect all tasks from all splits
    all_bddl_tasks = []  # List of (task_tuple, split_name) pairs
    for split_name in tqdm(split_names, desc=f"Collecting tasks from splits"):
        num_tasks_processed = 0
        split_path = os.path.join(bddl_path, split_name)
        bddl_files = sorted(os.listdir(split_path))
        skipped_count = 0
        
        # Use the configured number of demos for all splits
        n_demos_for_split = args.n_demos
        
        # Get view tasks for this split (if any views reference this split)
        view_tasks_for_split = source_split_to_view_tasks.get(split_name, None)
        
        # Collect all BDDL files to process for this split
        for bddl_file in bddl_files:
            if not bddl_file.endswith(".bddl"):
                continue
            
            # Get BDDL file name without extension
            bddl_file_name = os.path.splitext(bddl_file)[0]
            
            # If this split has view tasks, only process tasks that are in the view(s)
            if view_tasks_for_split is not None and bddl_file_name not in view_tasks_for_split:
                continue
            
            # Filter by specified BDDL file names if provided
            if args.bddl_files is not None:
                if bddl_file_name not in args.bddl_files:
                    continue
            
            # Skip the first n bddl files if requested
            if skipped_count < args.skip_bddl_files:
                skipped_count += 1
                continue
            if args.max_tasks_per_split is not None and num_tasks_processed >= args.max_tasks_per_split:
                break
            
            bddl_file_path = os.path.join(split_path, bddl_file)
            
            # Check if dataset already exists for this BDDL file
            expected_dataset_path = get_expected_dataset_path(bddl_file_path, args.run_dir)
            if os.path.exists(expected_dataset_path):
                print(f"Skipping {bddl_file_name}: dataset already exists at {expected_dataset_path}")
                continue
            
            num_tasks_processed += 1
            
            # Prepare arguments dictionary for worker
            # max_grasp_poses is automatically set to match n_demos for each BDDL file
            # Note: suppress_print will be set after we know if multiprocessing will be used
            args_dict = {
                'n_demos': n_demos_for_split,
                'generate_video': args.generate_video,
                'vis_grasp_poses': args.vis_grasp_poses,
                'keep_failures': args.keep_failures,
                'max_grasp_poses': n_demos_for_split,  # Automatically match the number of demos requested
                'resolution': args.resolution,
                'log_position_error_on_fail': args.log_position_error_on_fail,
                'start_demo_idx': args.start_demo_idx,
                'run_dir': args.run_dir,
                'suppress_print': False,  # Will be updated below if multiprocessing is used
                'config_path': args.config_path,
                'bddl_path': bddl_path,
                'suffix': args.suffix,
                'skip_variations': args.skip_variations,
                'min_attempts_for_rate_check': args.min_attempts_for_rate_check,
                'min_success_rate': args.min_success_rate,
                'max_attempts': args.max_attempts
            }
            
            task_tuple = (bddl_file_path, bddl_file_name, args_dict)
            all_bddl_tasks.append((task_tuple, split_name))
    
    # Second pass: Process all tasks together (all splits combined)
    use_multiprocessing_bddl = args.num_workers > 1 and len(all_bddl_tasks) > 1
    
    # Automatically suppress print statements when using multiprocessing to avoid interleaved output
    # Only suppress if we actually have multiple tasks and will use multiprocessing
    suppress_print = use_multiprocessing_bddl
    
    # Update suppress_print in all args_dicts
    for task_tuple, _ in all_bddl_tasks:
        task_tuple[2]['suppress_print'] = suppress_print
    
    if use_multiprocessing_bddl:
        # Use multiprocessing for BDDL file processing - submit all tasks at once
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks to executor, tracking split_name and bddl_file_name for each
            futures = {executor.submit(_process_bddl_file, task_tuple): (split_name, task_tuple[1]) 
                      for task_tuple, split_name in all_bddl_tasks}
            
            # Process results as they complete (from any split)
            total_tasks = len(futures)
            completed_tasks = 0
            print(f"Processing {total_tasks} BDDL files with {args.num_workers} workers...")
            for future in as_completed(futures):
                try:
                    bddl_file_name, video_path, successful_demos, failed_demos, success_rate, demo_results = future.result()
                    # Get the split_name for this task
                    split_name, _ = futures[future]
                    if video_path is not None:
                        video_info.append((bddl_file_name, video_path))
                    success_stats[bddl_file_name] = (successful_demos, failed_demos, success_rate, demo_results, video_path, split_name)
                except Exception as e:
                    # Get split_name and bddl_file_name even for failed tasks
                    split_name, bddl_file_name = futures[future]
                    print(f"Error processing BDDL file {bddl_file_name} from {split_name}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                finally:
                    completed_tasks += 1
                    print(f"Progress: {completed_tasks}/{total_tasks} BDDL files completed ({100.0 * completed_tasks / total_tasks:.1f}%)")
    else:
        # Sequential processing
        for task_tuple, split_name in tqdm(all_bddl_tasks, desc=f"Processing all BDDL files"):
            bddl_file_path, bddl_file_name, args_dict = task_tuple
            video_path, successful_demos, failed_demos, success_rate, demo_results = generate_demonstrations(bddl_file_path, **args_dict)
            if video_path is not None:
                video_info.append((bddl_file_name, video_path))
            success_stats[bddl_file_name] = (successful_demos, failed_demos, success_rate, demo_results, video_path, split_name)
    
    print(f"\nAll demonstrations generated. Results stored in:\n{args.run_dir}")
    
    # Print summary of succeeded and failed BDDL files with success rates, video paths, and demo checkmarks
    # Organized by split
    if success_stats:
        # Group stats by split
        stats_by_split = {}
        for bddl_name, stats in success_stats.items():
            successful_demos, failed_demos, success_rate, demo_results, video_path, split_name_for_task = stats
            if split_name_for_task not in stats_by_split:
                stats_by_split[split_name_for_task] = []
            stats_by_split[split_name_for_task].append((bddl_name, stats))
        
        print("\nBDDL File Summary (by split):")
        print("=" * 80)
        
        # Process each split
        for split_name in sorted(stats_by_split.keys()):
            split_stats = stats_by_split[split_name]
            succeeded_bddls = []
            failed_bddls = []
            
            # Use the configured number of demos for all splits
            n_demos_target = args.n_demos
            
            # Categorize tasks for this split
            for bddl_name, stats in split_stats:
                successful_demos, failed_demos, success_rate, demo_results, video_path, _ = stats
                if successful_demos >= n_demos_target:
                    succeeded_bddls.append((bddl_name, stats))
                else:
                    failed_bddls.append((bddl_name, stats))
            
            # Print split header
            print(f"\n{split_name}:")
            print("-" * 80)
            
            # Print succeeded tasks for this split
            if succeeded_bddls:
                print(f"\n  ✓ SUCCEEDED ({len(succeeded_bddls)}):")
                for bddl_name, stats in sorted(succeeded_bddls, key=lambda x: x[0]):
                    successful_demos, failed_demos, success_rate, demo_results, video_path, _ = stats
                    total_attempts = successful_demos + failed_demos
                    # Create checkmark/x mark row for demos
                    demo_marks = []
                    for demo_idx, success in demo_results:
                        demo_marks.append("✓" if success else "✗")
                    marks_str = " ".join(demo_marks)
                    print(f"    {bddl_name}:\n      {success_rate:.1f}% ({successful_demos}/{total_attempts} successful, {failed_demos} failed)")
                    if video_path is not None:
                        print(f"      Video: {video_path}")
                    print(f"      Demos: {marks_str}")
            else:
                print(f"\n  ✓ SUCCEEDED (0): None")
            
            # Print failed tasks for this split
            if failed_bddls:
                print(f"\n  ✗ FAILED ({len(failed_bddls)}):")
                for bddl_name, stats in sorted(failed_bddls, key=lambda x: x[0]):
                    successful_demos, failed_demos, success_rate, demo_results, video_path, _ = stats
                    total_attempts = successful_demos + failed_demos
                    # Create checkmark/x mark row for demos
                    demo_marks = []
                    for demo_idx, success in demo_results:
                        demo_marks.append("✓" if success else "✗")
                    marks_str = " ".join(demo_marks)
                    print(f"    {bddl_name}:\n      {success_rate:.1f}% (got {successful_demos}/{n_demos_target} demos, {failed_demos} failed)")
                    if video_path is not None:
                        print(f"      Video: {video_path}")
                    print(f"      Demos: {marks_str}")
            else:
                print(f"\n  ✗ FAILED (0): None")
        
        print("\n" + "=" * 80)

    # Create symlinks for view splits after all generation is complete
    if view_mappings:
        print("\n" + "=" * 80)
        print("Creating symlinks for view splits...")
        print("=" * 80)
        
        # Filter view_mappings by --bddl-files if specified
        filtered_view_mappings = {}
        if args.bddl_files is not None:
            for view_split, task_mappings in view_mappings.items():
                filtered_tasks = {
                    task_name: mapping 
                    for task_name, mapping in task_mappings.items()
                    if task_name in args.bddl_files
                }
                # Only include view split if it has tasks after filtering
                if filtered_tasks:
                    filtered_view_mappings[view_split] = filtered_tasks
        else:
            filtered_view_mappings = view_mappings
        
        create_view_symlinks(args.run_dir, filtered_view_mappings)
        print("View split symlinks created successfully!")

    # Print status after generation and symlinking
    if not args.skip_status:
        print("\n" + "=" * 80)
        print("Status AFTER generation:")
        print("=" * 80)
        print_dataset_status(
            run_dir=args.run_dir,
            include_splits=args.include_splits,
            bddl_path=bddl_path,
            suffix=args.suffix
        )

    # Backup init_files to run directory at the end (unless --no-backup is specified or run_dir was provided)
    # If run_dir was provided, we assume backup already exists, so skip backup
    if not args.no_backup and not run_dir_provided:
        # Include view splits in backup (they have their own init_files)
        all_splits_for_init_backup = list(split_names) + view_splits
        backup_init_files(original_init_files_path, args.run_dir, all_splits_for_init_backup)

    print(f"\nAll demonstrations generated. Results stored in:\n{args.run_dir}")
    print(f"Time taken: {(time.time() - start_time) / 3600:.2f} hours")
