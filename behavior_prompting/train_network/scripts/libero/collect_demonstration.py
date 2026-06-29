# modified from create_dataset.py in original LIBERO repo

import argparse
import datetime
import cv2
import h5py
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


def save_rgb_observations(env, demo_dir):
    """
    Save RGB observations from all available cameras to the demonstration folder.
    
    Args:
        env: The robosuite environment (may be wrapped)
        demo_dir: Directory path to save the images
    """
    # Get the underlying environment if wrapped
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    # Try to get observations with camera images
    obs = None
    try:
        obs = base_env._get_observations()
    except:
        pass
    
    # Find all RGB observation keys (typically end with '_image')
    rgb_keys = []
    if obs is not None:
        rgb_keys = [k for k in obs.keys() if k.endswith('_image')]
    
    if len(rgb_keys) > 0:
        # Save all RGB observations from obs dict
        timestamp = time.time()
        for key in rgb_keys:
            img = obs[key]
            # Convert from RGB to BGR if needed (robosuite images are typically RGB)
            if len(img.shape) == 3 and img.shape[-1] == 3:
                # Check if image is in [0, 1] range or [0, 255] range
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_bgr = img_bgr[::-1]
                filename = os.path.join(demo_dir, f"{key}_{timestamp:.6f}.png")
                cv2.imwrite(filename, img_bgr)
                print(f"Saved RGB observation: {filename}")


def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[], show_wrist_camera=True, print_pose=False, log_object_name=None, save_image_every=-1
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()

    # open the gripper all the way (this is included in the demonstrations data)
    for _ in range(10):
        ret = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
        if show_wrist_camera:
            obs = ret[0]
            cv2.imshow("robot0_eye_in_hand", obs["robot0_eye_in_hand_image"][::-1, :, ::-1])
        env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Initialize image saving timer if enabled
    last_image_save_time = None
    if save_image_every > 0:    
        # Ensure episode directory exists
        if hasattr(env, 'ep_directory') and env.ep_directory:
            os.makedirs(env.ep_directory, exist_ok=True)

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    while True:
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # don't start recording until the first action occurs
        if action is not None and count == 0 and np.linalg.norm(action - np.array([0, 0, 0, 0, 0, 0, -1])) < 0.0001: # don't start until first action occurs
            continue
        if count == 0:
            print("Starting to collect data")
        count += 1

        # If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # Run environment step

        ret = env.step(action)
        obs = ret[0]
        if show_wrist_camera:
            cv2.imshow("robot0_eye_in_hand", obs["robot0_eye_in_hand_image"][::-1, :, ::-1])
        
        # Print robot pose if flag is enabled
        if print_pose:
            pf = active_robot.robot_model.naming_prefix
            eef_pos = obs[f"{pf}eef_pos"]
            eef_quat = obs[f"{pf}eef_quat"]
            print(f"Robot pose - Position: {eef_pos}, Quaternion: {eef_quat}")
        
        # Log object position if object name is specified
        if log_object_name is not None:
            obj_pos_key = f"{log_object_name}_pos"
            obj_quat_key = f"{log_object_name}_quat"
            if obj_pos_key in obs and obj_quat_key in obs:
                obj_pos = obs[obj_pos_key]
                obj_quat = obs[obj_quat_key]
                print(f"Object '{log_object_name}' pose - Position: {obj_pos}, Quaternion: {obj_quat}")
            else:
                print(f"Warning: Object '{log_object_name}' not found in observations. Available keys: {[k for k in obs.keys() if '_pos' in k or '_quat' in k]}")
        
        # Save RGB images periodically if enabled
        if save_image_every > 0 and hasattr(env, 'ep_directory') and env.ep_directory:
            current_time = time.time()
            if last_image_save_time is None or (current_time - last_image_save_time) >= save_image_every:
                save_rgb_observations(env, env.ep_directory)
                last_image_save_time = current_time
        
        env.render()
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print(count)
    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, problem_info, bddl_file_name, remove_directory=[], directory_to_seed=None
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    ep_directories = sorted(os.listdir(directory))

    for ep_directory in ep_directories:
        if ep_directory in remove_directory:
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # Store seed if available (for consistent environment replay)
        if directory_to_seed is not None and ep_directory in directory_to_seed:
            seed = directory_to_seed[ep_directory]
            ep_data_grp.attrs["seed"] = int(seed)
        else:
            # If no seed mapping provided, use None (backward compatibility)
            ep_data_grp.attrs["seed"] = -1  # Use -1 to indicate seed not available

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = str(open(bddl_file_name, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="tmp_demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Panda"],
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument("--show-wrist-camera", type=bool, default=True)
    parser.add_argument(
        "--print-pose",
        action="store_true",
        help="Print the current pose of the robot during demonstration",
    )
    parser.add_argument(
        "--log-object",
        type=str,
        default=None,
        help="Name of the object to log position for during demonstration",
    )
    parser.add_argument(
        "--save-image-every",
        type=float,
        default=-1,
        help="Save RGB observations every N seconds. Set to -1 to disable (default: -1)",
    )
    parser.add_argument(
        "--save-image-resolution",
        type=int,
        default=384,
        help="Resolution of the images to save. (default: 384)",
    )

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=args.show_wrist_camera,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=args.show_wrist_camera,
        reward_shaping=True,
        control_freq=20,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ] if args.show_wrist_camera else ["agentview"],
        camera_heights=args.save_image_resolution if args.save_image_every > 0 else 128,
        camera_widths=args.save_image_resolution if args.save_image_every > 0 else 128,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    if args.save_image_every > 0:
        print("disabling guide lines since we are saving images")
        env._vis_settings['robots'] = False
        env._vis_settings['env'] = False
        env._vis_settings['grippers'] = False

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "tmp_intermediate_demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        env.viewer.add_keypress_callback(device.on_press)
        # env.viewer.add_keyup_callback(device.on_release)
        # env.viewer.add_keyrepeat_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, remove_directory, args.show_wrist_camera, args.print_pose, args.log_object, args.save_image_every
        )
        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, problem_info, args.bddl_file, remove_directory
            )
            i += 1
