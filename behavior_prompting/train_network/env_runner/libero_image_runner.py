from behavior_prompting.train_network import fix_robosuite_log_permission_issue
fix_robosuite_log_permission_issue()

import os
from typing import Optional
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from multiprocessing import Process
import wandb.sdk.data_types.video as wv
from behavior_prompting.train_network.gym_util.async_vector_env import AsyncVectorEnv
from behavior_prompting.train_network.gym_util.multistep_wrapper import MultiStepWrapper
from behavior_prompting.train_network.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
)
from behavior_prompting.common.trajectory_util import vis_trajectories
from behavior_prompting.train_network.utils.dataset_util import pad_dataset_to_length
from behavior_prompting.train_network.utils.plot_util import PromptAttentionLogger, vis_actions_rollout, vis_proprio_rollout
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder

from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.env_runner.base_runner import BaseRunner
from libero.libero import get_libero_path

import robomimic.utils.file_utils as FileUtils

from behavior_prompting.train_network.utils.video_util import cut_first_n_frames

from transformers import CLIPTokenizer
from behavior_prompting.train_network.model.common.rotation_transformer import (
    RotationTransformer,
)
from behavior_prompting.train_network.utils.libero_util import hdf5_to_bddl, hdf5_to_task, pad_action
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.dataset.libero_replay_image_dataset import LiberoReplayImageDataset
from behavior_prompting.common.pytorch_util import move_batch_to_device, move_batch_to_numpy, remove_batch_dim_from_prompt
from behavior_prompting.train_network.utils.libero_util import vis_prompt
from behavior_prompting.common.transform_util import pose_6d_to_4x4
from behavior_prompting.train_network.utils.prompt_util import collate_prompts
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner

def create_env(bddl_file_name, camera_heights, camera_widths, control_delta, enable_render=True, render_size=None, render_gpu_device_id=-1):
    env = ControlEnvForRunner(
        bddl_file_name=bddl_file_name,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        has_offscreen_renderer=enable_render,
        has_renderer=False,
        use_camera_obs=enable_render,
        control_delta=control_delta,
        render_size=render_size,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.metadata = None # needed for AsyncVectorEnv

    return env


class LiberoImageRunner(BaseRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        dataset_path,
        output_dir,
        shape_meta: dict,
        cache_dir: str,
        text_encoder_model_name: Optional[str]=None,
        n_envs=10, # this is the number of simulation environments to create. If n_train+n_test > n_envs, then we will run multiple chunks of size n_envs until we cover all the environments
        n_train=10,
        n_train_vis=1,
        train_start_idx=0,
        n_test=50,
        n_test_vis=1,
        max_steps=400,
        render_obs_key=["agentview_image", "robot0_eye_in_hand_image"],
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        include_file_filters:list[str]=[], # used in load_env.py which is why it's unused here
        dataset_splits:list[str]=[], # used in load_env.py which is why it's unused here
        unseen_dataset_splits:list[str]=[], # used in load_env.py which is why it's unused here
        unseen_include_file_filters:list[str]=[], # used in load_env.py which is why it's unused here
        exec_action_horizon:int=12,
        vis_prompt:bool=False,
        vis_actions_rollout:bool=False,
        vis_proprio_rollout:bool=False,
        vis_trajectory_video:bool=False,
        replay_dataset_action: bool=False,
        vis_attention_map: bool=False,
        vis_env_index: int=0,
        is_seen: bool=True,
        eval_seen: bool=True, # used in load_env.py which is why it's unused here
        eval_unseen: bool=False, # used in load_env.py which is why it's unused here,
        unseen_max_n: Optional[int]=None,
        render_size: Optional[int]=None,
        attention_map_cmap: str="GnBu",
        save_attention_weights: bool=False,
        scale_attention_weights: bool=False,
        use_proportion_of_colormap: Optional[float]=None,
        render_gpu_device_id: int = -1
    ):
        super().__init__(output_dir)

        self.num_open_gripper_steps = math.ceil(10 / exec_action_horizon) * exec_action_horizon
        max_steps += self.num_open_gripper_steps # at the start of the rollout after reset we first execute the open gripper action so we need to add the action horizon to the max steps to ensure we account for these additional open gripper actions

        if not is_seen:
            n_test += n_train
            n_test_vis += n_train_vis
            n_train = 0
            n_train_vis = 0
            if unseen_max_n is not None:
                n_test = min(n_test, unseen_max_n)

        if n_envs is None or n_envs > n_train + n_test:
            n_envs = n_train + n_test
        assert vis_env_index < n_envs, f'vis_env_index ({vis_env_index}) must be less than n_envs ({n_envs})'

        robosuite_fps = 20
        self.steps_per_render = max(robosuite_fps // fps, 1)
        self.use_prompting = shape_meta['use_prompting']
        self.use_goal_image = 'goal_image' in shape_meta['obs'] and not shape_meta['obs']['goal_image'].get('ignore_by_policy', False)
        self.prompt_sample_mode = shape_meta['prompt_sample_mode'] if self.use_prompting else None
        self.action_rep = shape_meta['action']['rep']
        self.demo_name = os.path.basename(dataset_path).replace('.hdf5', '') # get the name of the hdf5 file without the extension
        self.dataset_name = os.path.basename(os.path.dirname(dataset_path))
        self.is_seen = is_seen

        # observation/action horizon
        self.key_horizon = dict()
        for key, attr in shape_meta['obs'].items():
            self.key_horizon[key] = shape_meta['obs'][key]['horizon']

        if self.prompt_sample_mode == 'sequence':
            # sequence prompting policies need to get all observations that occur during the rollout so we need to have the env runner recording observations for the entire batch of the past actions executed for each step
            max_obs_horizon = exec_action_horizon
        else:
            # for other policies (non-prompting or pair prompting) we need to record history of observations with length of the longest observation horizon
            max_obs_horizon = max(self.key_horizon.values())
        if vis_proprio_rollout or vis_trajectory_video:
            # if we want to visualize the proprio or trajectory video we need to record the entire batch of the past proprio for the entire duration of executed actions
            max_obs_horizon = max(max_obs_horizon, exec_action_horizon)
        assert exec_action_horizon <= shape_meta['action']['horizon'], 'number of steps to execute must not be greater than the action horizon predicted by the policy'

        # get task info
        bddl_path = hdf5_to_bddl(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)    
        env_kwargs = {
            "bddl_file_name": bddl_path,
            "control_delta": self.action_rep == "delta",
            "camera_heights": env_meta["env_kwargs"]["camera_heights"],
            "camera_widths": env_meta["env_kwargs"]["camera_widths"],
            "render_size": render_size,
            "render_gpu_device_id": render_gpu_device_id,
        }
        self.task = hdf5_to_task(dataset_path)

        # determine rgb and lowdim keys
        self.rgb_keys = []
        self.lowdim_keys = []
        for key in shape_meta['obs']:
            if shape_meta['obs'][key]['type'] == 'rgb':
                self.rgb_keys.append(key)
            elif shape_meta['obs'][key]['type'] == 'low_dim':
                self.lowdim_keys.append(key)

        self.sim_rgb_keys = [k for k in self.rgb_keys if k != "goal_image"]

        # proprio keys (for visualization)
        proprio_key_names = []
        proprio_key_sizes = []
        for lowdim_key in sorted(self.lowdim_keys):
            if shape_meta["obs"][lowdim_key]['prompt_type'] == 'proprioception':
                proprio_key_names.append(lowdim_key)
                shape = shape_meta["obs"][lowdim_key]['shape']
                assert len(shape) == 1, f'proprio value must be 1D'
                proprio_key_sizes.append(shape[0])

        # the names of the observations in simulation have different names (and potentially different formats) than the ones in the training dataset (which is the naming convention that the policy uses)
        self.obs_key_to_runner_key = {
            'agentview_rgb': 'agentview_image', # just rename
            'eye_in_hand_rgb': 'robot0_eye_in_hand_image', # just rename
            'ee_pos': 'robot0_eef_pos', # just rename
            'ee_ori': 'robot0_eef_quat', # sim gives xyzw quat (note real part last!), need to convert to xyz euler axis-angle
            'gripper_states': 'robot0_gripper_qpos' # just rename
        }
        self.runner_key_to_obs_key = {v: k for k,v in self.obs_key_to_runner_key.items()}
        
        # get init states
        # init_states = task_suite.get_task_init_states(task_id) # this doesn't work on newer versions of torch so we do it manually
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.task.problem_folder,
            self.task.init_states_file,
        )
        init_states = torch.load(init_states_path, weights_only=False)

        assert n_test <= len(init_states), f"number of test episodes ({n_test}) must be less than or equal to the number of init states ({len(init_states)})"

        def env_fn():
            libero_env = create_env(**env_kwargs)
            libero_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    libero_env,
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=self.steps_per_render,
                ),
                n_obs_steps=max_obs_horizon,
                n_action_steps=exec_action_horizon,
                max_episode_steps=max_steps,
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            libero_env = create_env(**env_kwargs, enable_render=False)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    libero_env,
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=self.steps_per_render,
                ),
                n_obs_steps=max_obs_horizon,
                n_action_steps=exec_action_horizon,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        pre_collected_actions = None
        pre_collected_obs = None

        # train
        with h5py.File(dataset_path, "r") as f:
            if self.is_seen:
                demos = f["data"]
                demos_indices = sorted([int(x.replace('demo_', '')) for x in demos])

            for i in range(n_train):
                train_idx = demos_indices[(train_start_idx + i) % len(demos_indices)]
                env_seed = train_idx
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0] # pull an init state from the first frame of a demonstration from the training dataset. Technically this might be from a validation demonstration... but this doesn't matter to much since we are going to only report numbers for the test set initializations

                if i == 0:
                    action_key = 'actions' if self.action_rep == 'delta' else 'abs_actions'
                    pre_collected_actions = f[f"data/demo_{train_idx}/{action_key}"][:]
                    pre_collected_obs = {k: v[:] for k, v in f[f"data/demo_{train_idx}/obs"].items()}
                
                def init_fn(env, init_state=init_state, enable_render=enable_render, demo_name=self.demo_name, env_seed=env_seed):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", f"train_{demo_name}_{env_seed}_{wv.util.generate_id()}.mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, ControlEnvForRunner)

                    env.env.env.register_init_state(None)
                    env.seed(7) # apparently it's important to seed even if we provide a random state https://github.com/Physical-Intelligence/openpi/blob/95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31/examples/libero/main.py#L195. Using 7 since that is what openvla and openpi do.
                    env.reset()
                    env.env.env.register_init_state(init_state)

                env_seeds.append(env_seed)
                env_prefixs.append(
                    f"train/{self.dataset_name}/{bddl_path.split('/')[-1][:-5]}_"
                )
                env_init_fn_dills.append(dill.dumps(init_fn))
        # test
        for i in range(n_test):
            enable_render = i < n_test_vis
            env_seed = i
            init_state = init_states[i] # pull an init state from the provided init states provided from the benchmark

            def init_fn(env, init_state=init_state, enable_render=enable_render, demo_name=self.demo_name, env_seed=env_seed):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", f"test_{demo_name}_{env_seed}_{wv.util.generate_id()}.mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # the correct ordering seems to be seed, reset, then set_init_state (see LIBERO README). Our setup ensures this occurs in that order. In particular we add a register_init_state call to just save this init_state within the environment and then when we call env.reset() later, since we updated it, it will run env.set_init_state() behind the scenes (see updated env_wrapper.py).
                assert isinstance(env.env.env, ControlEnvForRunner)
                env.env.env.register_init_state(None)
                env.seed(7) # apparently it's important to seed even if we provide a random state https://github.com/Physical-Intelligence/openpi/blob/95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31/examples/libero/main.py#L195
                env.reset()
                env.env.env.register_init_state(init_state)

            env_seeds.append(env_seed)
            env_prefixs.append(f"test/{self.dataset_name}/{bddl_path.split('/')[-1][:-5]}_")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)

        self.shape_meta = shape_meta
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.max_steps = len(pre_collected_actions) if replay_dataset_action else max_steps
        self.rotation_transformer_axisangle_to_rot6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
        self.rotation_transformer_quatwxyz_to_rot6d = RotationTransformer(from_rep='quaternion', to_rep='rotation_6d')
        self.rotation_transformer_quatwxyz_to_axisangle = RotationTransformer(from_rep='quaternion', to_rep='axis_angle')
        self.rotation_transformer_axis_angle_to_matrix = RotationTransformer(from_rep='axis_angle', to_rep='matrix')
        self.tqdm_interval_sec = tqdm_interval_sec
        self.dataset_path = dataset_path
        self.prompt_sequence_length = shape_meta.prompt_sequence_length
        self.exec_action_horizon = exec_action_horizon
        self.vis_prompt = vis_prompt
        self.vis_actions_rollout = vis_actions_rollout
        self.vis_proprio_rollout = vis_proprio_rollout
        self.vis_trajectory_video = vis_trajectory_video
        self.replay_dataset_action = replay_dataset_action
        self.proprio_key_names = proprio_key_names
        self.proprio_key_sizes = proprio_key_sizes
        self.pre_collected_actions = pre_collected_actions
        self.pre_collected_obs = pre_collected_obs
        self.vis_attention_map = vis_attention_map
        self.vis_env_index = vis_env_index
        self.attention_map_cmap = attention_map_cmap
        self.save_attention_weights = save_attention_weights
        self.scale_attention_weights = scale_attention_weights
        self.use_proportion_of_colormap = use_proportion_of_colormap
        self.replay_buffer = None
        self.cache_dir = cache_dir

        # compute language goal embedding
        self.using_language = 'task_language' in self.shape_meta['obs'] and not self.shape_meta['obs']['task_language'].get('ignore_by_policy', False)
        if self.using_language:
            assert shape_meta['obs']['task_language']['horizon'] == 1, 'task_language horizon must be 1'
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
            tokens = self.clip_tokenizer(
                self.task.language,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='np'
            )
            self.task_language_token_ids = tokens['input_ids'].astype(np.int64)

    def run(self, policy: Optional[BasePolicy], enable_expensive_vis: bool=True):
        if self.replay_dataset_action:
            device: torch.device = torch.device('cpu')
            self.vis_attention_map = False
        else:
            device = policy.device
            assert not policy.training, "policy must be in eval mode"

            if not policy.supports_prompting():
                self.vis_attention_map = False

        vis_attention_map = self.vis_attention_map and enable_expensive_vis

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # load all the prompts
        # if policy is prompt based model, then prompt
        if self.use_prompting:
            # load a demonstration from the dataset
            # the way we accomplish this is by creating an instance of the libero dataset with only this dataset file and then sampling a demonstration from it
            prompt_dataset = LiberoReplayImageDataset(
                shape_meta=self.shape_meta,
                dataset_splits=[self.dataset_name],
                dataset_name=self.dataset_name,
                replay_buffer=self.replay_buffer,
                action_padding=False, # this parameter is not used in prompting, so it doesn't matter what we set it to
                use_cache=True,
                overwrite_cache=False,
                cache_dir=self.cache_dir,
                seed=0, # this set sets the random ordering of tasks in the dataset such that we get unique prompts (when using sequence prompting). Here we just set it to 0 for consistency across evals
                val_ratio=0,
                sample_type='task',
                only_prompt=True,
                max_segments=-1, # TODO: instead of loading the whole dataset theoretically we could set the max segments to the max number of segments we want in our prompts to speed things up
                name_suffix=self.demo_name,
                include_file_filters=[self.demo_name], # we only pull demonstration data for this particular task
                training_split_info=policy.get_training_split_info() if self.is_seen else None, # the policy contains the training split info which is used to determine which tasks are in the training split since we only want to sample prompts from the training split
            )
            self.replay_buffer = prompt_dataset.replay_buffer # we store the replay buffer so that we can reuse it for the next evaluation instead of reloading the dataset every single time

            # note that it is theoretically possible that when doing evaluations the prompt we sample corresponds to the exact same initial configuration that is used to create the train sim environment (since the train sim environments are instantiated by looking at state 0 from the demonstration data). Thus it may be the case that the policy could just exactly copy the prompt actions to complete the task. This is such a minor issue and would only impact training eval metrics that we don't worry about it

            if self.prompt_sample_mode == 'sequence':
                # right now we just sample one prompt which is used across all rollouts. This is because ICRT currently only supports prompts of the same length (though this is a software limitation that could be removed and is not a core limitation of ICRT in principle)
                # we create a dataloader that will sample the same prompt n_envs times and collate them together
                prompt_sample_index = 0 # note that by sampling index 0, we are ensuring that the start of the sampled prompt is the start of some episode. This means the prompt will always start with a full episode. If we wanted to change this we could pick a random index in which case the start of the prompt could be anywhere in the episode (but will still include at least one full demonstration of the task)
                prompt_dataset_to_sample = Subset(prompt_dataset, [prompt_sample_index] * n_envs) # repeat the prompt dataset n_envs times
            else:
                assert self.prompt_sample_mode == 'pair'
                # for pair prompting we sample a different prompt for each rollout since we don't have the limitation that the prompts must be the same length across all rollouts
                prompt_dataset_to_sample = pad_dataset_to_length(prompt_dataset, n_envs)
            
            print('Loading prompts...')
            prompt_dataloader = DataLoader(
                prompt_dataset_to_sample,
                batch_size=n_envs,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_prompts,
            )
            prompts_batched = next(iter(prompt_dataloader))
            prompts_batched = move_batch_to_device(prompts_batched, device=device) # move the prompts to the correct device
            prompts_batched = prompts_batched['obs']['prompt']
            print('Finished loading prompts!')
            
            del prompt_dataloader
            del prompt_dataset
            del prompt_dataset_to_sample

            if self.vis_prompt:
                prompt_for_vis = move_batch_to_numpy(remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index))
                vis_prompt(prompt_for_vis, os.path.join(self.output_dir, "vis", f"prompt_{self.demo_name}.mp4"))

            total_available_action_steps_after_prompting = policy.num_available_actions()
            if total_available_action_steps_after_prompting is not None and total_available_action_steps_after_prompting < self.max_steps:
                print(f"WARNING: the number of steps we are able to execute is less than the max_steps specified in the env_runner. This is because the model is prompted. This leaves only {total_available_action_steps_after_prompting} to execute which is less than the specified max steps of {self.max_steps}")
                # TODO: probably should make this an error rather than a warning

        goal_images_all = None
        if (
            self.use_goal_image
            and policy is not None
            and not self.replay_dataset_action
        ):
            print("Loading goal images...")
            train_goal_ds = LiberoReplayImageDataset(
                shape_meta=self.shape_meta,
                dataset_splits=[self.dataset_name],
                dataset_name=self.dataset_name,
                replay_buffer=self.replay_buffer,
                action_padding=True, # this parameter is not used with only_goal_image, so it doesn't matter what we set it to
                use_cache=True,
                overwrite_cache=False,
                cache_dir=self.cache_dir,
                seed=0,
                val_ratio=0,
                sample_type="task",
                only_goal_image=True,
                max_segments=-1,
                name_suffix=self.demo_name,
                include_file_filters=[self.demo_name],
                training_split_info=(
                    policy.get_training_split_info() if self.is_seen else None
                ),
            )
            self.replay_buffer = train_goal_ds.replay_buffer

            n_inits = len(self.env_init_fn_dills)

            def load_goal_batch(ds, bs: int) -> torch.Tensor:
                dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
                batch = next(iter(dl))
                batch = move_batch_to_device(batch, device=device)
                return batch["obs"]["goal_image"]

            # Training-split goals only (same spirit as prompts); test rollouts reuse train-split goals.
            goal_padded = pad_dataset_to_length(train_goal_ds, n_inits)
            goal_images_all = load_goal_batch(goal_padded, n_inits)

            assert goal_images_all.shape[0] == n_inits, (
                f"goal images {goal_images_all.shape[0]} != n_inits {n_inits}"
            )
            print("Finished loading goal images.")

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # in the case the n_inits doesn't divide evenly into n_envs, we pad the goal images to match the number of envs
            chunk_goal_images = None
            if goal_images_all is not None:
                chunk_goal_images = goal_images_all[this_global_slice]
                if chunk_goal_images.shape[0] < n_envs:
                    pad_n = n_envs - chunk_goal_images.shape[0]
                    chunk_goal_images = torch.cat(
                        [
                            chunk_goal_images,
                            chunk_goal_images[:1].expand(
                                pad_n, *chunk_goal_images.shape[1:]
                            ),
                        ],
                        dim=0,
                    )

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # init policy
            if policy is not None:
                policy.reset(action_exec_horizon=self.exec_action_horizon)

            if self.use_prompting:
                policy.prompt(prompts_batched)

            pbar = tqdm.tqdm(
                total=min(self.max_steps, total_available_action_steps_after_prompting) if self.use_prompting and total_available_action_steps_after_prompting is not None else self.max_steps,
                desc=f"Eval (chunk {chunk_idx+1}/{n_chunks}) \"{self.task.language}\"",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            def prepare_obs_dict(obs, chunk_goal_images=chunk_goal_images):
                """Prepares the observations for the policy inference.
                Returns two versions of the observation dict. The first is the numpy version which is used for visualization (contains full observation history). The second is the torch version which is used for policy inference (contains the format the policy expects).
                """
                visualization_obs_dict = dict(obs)
                for img_key in self.sim_rgb_keys:
                    runner_img_key = self.obs_key_to_runner_key[img_key]
                    visualization_obs_dict[runner_img_key] = (visualization_obs_dict[runner_img_key] / 255).transpose(0,1,4,2,3) # (B, T, H, W, C) (0-255) -> (B, T, C, H, W) (0-1)

                # rename simulator keys to match policy keys
                for obs_key in list(visualization_obs_dict.keys()):
                    policy_key_name = self.runner_key_to_obs_key[obs_key]
                    # rename observation from sim name to policy name
                    visualization_obs_dict[policy_key_name] = visualization_obs_dict.pop(obs_key)
                
                # convert rotation representation for ee_ori from quat wxyz from simulator to rot6d for policy
                if 'ee_ori' in visualization_obs_dict:
                    quat_xyzw = visualization_obs_dict['ee_ori']
                    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]] 
                    rot6d = self.rotation_transformer_quatwxyz_to_rot6d.forward(quat_wxyz)
                    visualization_obs_dict['ee_ori'] = rot6d

                # device transfer
                policy_obs_dict = dict_apply(
                    visualization_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # trim keys to the correct horizon
                for obs_key in policy_obs_dict.keys():
                    horizon = self.key_horizon[obs_key] if self.prompt_sample_mode != 'sequence' else self.exec_action_horizon
                    policy_obs_dict[obs_key] = policy_obs_dict[obs_key][:, -horizon:]

                # resize the rgb keys to the appropriate size
                for rgb_key in self.sim_rgb_keys:
                    # resize image to be 224x224
                    resize = 224
                    B, T, C, H, W = policy_obs_dict[rgb_key].shape
                    resized_tensor = F.interpolate(
                        policy_obs_dict[rgb_key].reshape(B * T, C, H, W),
                        size=(resize, resize),
                        mode="bilinear",
                        align_corners=False,
                    )
                    policy_obs_dict[rgb_key] = resized_tensor.view(
                        B, T, C, resize, resize
                    )

                if self.use_goal_image and chunk_goal_images is not None:
                    policy_obs_dict["goal_image"] = chunk_goal_images

                # insert language goal
                if self.using_language:
                    token_ids = np.tile(self.task_language_token_ids, (n_envs, 1, 1))  # (n_envs, 1, 77)
                    policy_obs_dict['task_language'] = torch.from_numpy(token_ids).to(device=device)

                return visualization_obs_dict, policy_obs_dict

            # reset env
            obs = env.reset() # this will call env.set_init_state behind the scenes not the normal sim reset

            # weirdly it seems during data collection the gripper is fully open, but in the test environments the gripper can be initialized to be partially closed. This can lead to out of distribution issues, so we just open the gripper all the way
            if self.action_rep == 'delta':
                open_gripper_action = np.zeros((n_envs, self.exec_action_horizon, 7))
                open_gripper_action[:, :, 6] = -1
            else: # absolute
                # we command the robot to its current position with a gripper open action
                pos = obs['robot0_eef_pos'][:, -1] # (B, 3)
                quat_xyzw = obs['robot0_eef_quat'][:, -1] # (B, 4)
                quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]] # (B, 4)
                axis_angle = self.rotation_transformer_quatwxyz_to_axisangle.forward(quat_wxyz) # (B, 3)
                rot_mat = self.rotation_transformer_axis_angle_to_matrix.forward(axis_angle) # (B, 3, 3)

                # rotate -90 degrees about z (for some reason eef_quat needs this rotation)
                rot_z_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
                                   [np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
                                   [0, 0, 1]])
                rot_mat = rot_mat @ rot_z_90
                axis_angle = self.rotation_transformer_axis_angle_to_matrix.inverse(rot_mat) # (B, 3)

                gripper_action = np.full((n_envs, 1), -1)
                open_gripper_action = np.concatenate([pos, axis_angle, gripper_action], axis=-1) # (B, 7)

                # insert horizon
                open_gripper_action = np.expand_dims(open_gripper_action, axis=1) # (B, 1, 7)
                open_gripper_action = np.tile(open_gripper_action, (1, self.exec_action_horizon, 1)) # (B, exec_action_horizon, 7)

            num_steps_wait = 0
            while num_steps_wait < self.num_open_gripper_steps:
                obs, reward, done, info = env.step(open_gripper_action)
                num_steps_wait += self.exec_action_horizon
            assert num_steps_wait == self.num_open_gripper_steps
            # obs, reward, done, info = env.step(np.zeros((n_envs, self.exec_action_horizon, 7))) # I observe the robot moves a little bit even with zero actions when doing action, perhaps since the initial state has some initial velocity. Add this line back in to observe this effect.

            visualization_obs_dict, policy_obs_dict = prepare_obs_dict(obs)

            # store initial proprioception
            cur_proprio_rollout = np.concatenate([visualization_obs_dict[key] for key in self.proprio_key_names], axis=-1) # (B, T, proprio_dim)
            all_proprio_rollout = cur_proprio_rollout[:, :1] # (B, 1, proprio_dim); only take 1 step of the initial proprio since we are at the start of the rollout and only have observed a single state

            # start rollout
            all_execed_actions = None # (B, T, action_dim)
            done = False
            pred_steps = 0
            if vis_attention_map and chunk_idx == 0:
                attn_logger = PromptAttentionLogger(policy, self.vis_env_index)
            while not done:
                # get action either from policy or from replay dataset
                if self.replay_dataset_action:
                    action = self.pre_collected_actions[:self.exec_action_horizon] # self.pre_collected_actions is (T_demonstration, action_dim) and contains actions for just the first training env
                    action = np.expand_dims(action, axis=0) # (1, T_demonstration, action_dim)
                    
                    # pad with zeros for rest of the envs
                    n_pad = n_envs - action.shape[0]
                    if n_pad > 0:
                        action = np.pad(action, ((0, n_pad), (0,0), (0,0)), mode='constant')

                    # pad horizon if necessary
                    if action.shape[1] < self.exec_action_horizon:
                        action = self.transform_action_for_policy(action, undo=False) # axis-angle -> rot6d (B, num_pred_steps, 10)
                        action = pad_action(action, self.action_rep, self.exec_action_horizon)
                        action = self.transform_action_for_policy(action, undo=True) # rot6d -> axis-angle (B, num_pred_steps, 7)

                    # check that the shape is correct
                    assert action.shape[0] == n_envs and action.shape[1] == self.exec_action_horizon, f"cur_action.shape: {action.shape} is not expected shape"

                    self.pre_collected_actions = self.pre_collected_actions[self.exec_action_horizon:]
                else:
                    with torch.inference_mode():
                        kwargs = dict()
                        if vis_attention_map and chunk_idx == 0:
                            kwargs['need_weights'] = True
                            kwargs['average_attn_weights'] = True
                        action_dict = policy.predict_action(
                            policy_obs_dict,
                            **kwargs,
                        )
                    
                    action = action_dict["action"].detach().to("cpu").numpy()  # (B, num_pred_steps, 10)
                    action = self.transform_action_for_policy(action, undo=True) # rot6d -> axis-angle (B, num_pred_steps, 7)
                    action = action[:, :self.exec_action_horizon] # exec only part of the full action trajectory (B, exec_action_horizon, 7)

                    if vis_attention_map and chunk_idx == 0:
                        attn_logger.log(action_dict)

                # step env
                obs, reward, done, info = env.step(action)

                for i in range(len(reward)):
                    if reward[i] == 1:
                        done[i] = True

                done = np.all(done)

                # prepare obs for next iteration
                visualization_obs_dict, policy_obs_dict = prepare_obs_dict(obs)

                if self.vis_proprio_rollout or self.vis_trajectory_video:
                    cur_proprio_rollout = np.concatenate([visualization_obs_dict[key] for key in self.proprio_key_names], axis=-1) # (B, T, proprio_dim)
                    assert cur_proprio_rollout.shape[1] == self.exec_action_horizon, f"cur_proprio_rollout.shape: {cur_proprio_rollout.shape} is not expected shape"
                    all_proprio_rollout = np.concatenate([all_proprio_rollout, cur_proprio_rollout], axis=1) # concat along time dimension
                
                if self.vis_actions_rollout:
                    if all_execed_actions is None:
                        all_execed_actions = action[:, :self.exec_action_horizon]    
                    else:
                        all_execed_actions = np.concatenate([all_execed_actions, action[:, :self.exec_action_horizon]], axis=1)

                # update pbar
                pbar.update(action.shape[1])
                pred_steps += action.shape[1]

                if self.use_prompting and total_available_action_steps_after_prompting is not None:
                    # prompting models have a fixed sequence length so we have to stop once we reach the sequence length
                    if pred_steps >= total_available_action_steps_after_prompting:
                        done = True
                
                if self.replay_dataset_action:
                    if len(self.pre_collected_actions) == 0:
                        done = True
            
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]

            if self.use_prompting and total_available_action_steps_after_prompting is not None and total_available_action_steps_after_prompting < self.max_steps:
                assert policy.num_available_actions() < self.exec_action_horizon, 'if we are limited by the number of steps we can execute, then the number of steps we can execute should be less than the number of steps we can predict at the very end of execution'

        # clear out video buffer
        _ = env.reset()
        # clear out policy buffers
        if policy is not None:
            policy.reset(action_exec_horizon=self.exec_action_horizon)

        vis_demo_name = os.path.basename(all_video_paths[self.vis_env_index]).replace('.mp4', '')

        vis_out_dir = os.path.join(self.output_dir, "vis")
        os.makedirs(vis_out_dir, exist_ok=True)

        # visualize action rollout
        if self.vis_actions_rollout:
            assert n_chunks == 1, 'only supported if we are running a single chunk'
            first_execed_action = all_execed_actions[self.vis_env_index]

            # convert rotation in first_execed_action from axis-angle to rot6d to match the prompt format
            pos = first_execed_action[:, :3]
            axisangle = first_execed_action[:, 3:6]
            gripper_action = first_execed_action[:, 6:]
            rot6d = self.rotation_transformer_axisangle_to_rot6d.forward(axisangle)
            first_execed_action = np.concatenate([pos, rot6d, gripper_action], axis=-1)

            prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index) if self.use_prompting else None
            vis_actions_rollout(os.path.join(vis_out_dir, f"actions_rollout_{vis_demo_name}.png"), first_execed_action, prompt=prompt_for_vis)

        if self.replay_dataset_action:
            assert n_chunks == 1, 'only supported if we are running a single chunk'
            if 'ee_ori' in self.proprio_key_names:
                self.pre_collected_obs['ee_ori'] = self.rotation_transformer_axisangle_to_rot6d.forward(self.pre_collected_obs['ee_ori'])
            all_proprio_replay = np.concatenate([self.pre_collected_obs[key] for key in self.proprio_key_names], axis=-1) # (T, proprio_dim) for the first env only

        # visualize proprio rollout
        if self.vis_proprio_rollout:
            assert n_chunks == 1, 'only supported if we are running a single chunk'
            first_proprio = all_proprio_rollout[self.vis_env_index]
            proprio_list = [first_proprio]
            proprio_legend_names = ["Rollout observations"] if not self.use_prompting else ["Prompt then rollout observations"]

            if self.replay_dataset_action:
                proprio_list.append(all_proprio_replay)
                proprio_list[0] = proprio_list[0][:len(proprio_list[1])] # the proprio rollout is longer than the precollected actions because the full exec horizon happens each iteration which may exceed the length of the precollected proprio, so we just trim it
                proprio_legend_names.append("Replay dataset observations")

            prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index) if self.use_prompting else None
            vis_proprio_rollout(
                os.path.join(vis_out_dir, f"proprio_rollout_{vis_demo_name}.png"),
                proprio_list=proprio_list,
                proprio_key_names=self.proprio_key_names, 
                proprio_key_sizes=self.proprio_key_sizes, 
                prompt=prompt_for_vis,
                proprio_legend_names=proprio_legend_names
            )

        # visulize the trajectory of the prompt and the rollout
        if self.vis_trajectory_video:
            assert n_chunks == 1, 'only supported if we are running a single chunk'
            assert self.proprio_key_names[0] == 'ee_ori' and self.proprio_key_names[1] == 'ee_pos', 'proprio_key_names must be ee_ori and then ee_pos in that ordering for this visualization to work properly'

            def get_trajectory(proprio):
                rot6d = proprio[:, :6]
                axisangle = self.rotation_transformer_axisangle_to_rot6d.inverse(rot6d)
                pos = proprio[:, 6:9]
                poses = np.concatenate([pos, axisangle], axis=-1)
                poses = pose_6d_to_4x4(poses)
                return poses

            # save the proprio rollout to a video
            first_proprio = all_proprio_rollout[self.vis_env_index]
            rollout_poses = get_trajectory(first_proprio)
            trajectories = [rollout_poses]

            if self.replay_dataset_action:
                # get the precollected trajectory
                replay_trajectory = get_trajectory(all_proprio_replay)
                trajectories.append(replay_trajectory)
                trajectories[0] = trajectories[0][:len(trajectories[1])] # the proprio rollout is longer than the precollected actions because the full exec horizon happens each iteration which may exceed the length of the precollected proprio, so we just trim it

            # for some reason there was an EGL initiation error, so we do it in a separate process
            _p = Process(target=vis_trajectories, args=(os.path.join(vis_out_dir, f"proprio_rollout_{vis_demo_name}.mp4"), trajectories), kwargs={'axis_every_n_steps': 10})
            _p.start()
            _p.join()
            
            if self.use_prompting:
                prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index)
                ee_pos = prompt_for_vis['obs']['ee_pos'].cpu().numpy()
                ee_ori = prompt_for_vis['obs']['ee_ori'].cpu().numpy()

                if len(ee_ori.shape) == 3: # remove the num_pred_steps dimension if present
                    ee_ori = ee_ori[:, 0]
                    ee_pos = ee_pos[:, 0]

                ee_ori = self.rotation_transformer_axisangle_to_rot6d.inverse(ee_ori)
                prompt_poses = np.concatenate([ee_pos, ee_ori], axis=-1)
                prompt_poses = pose_6d_to_4x4(prompt_poses)
                chunk_n_action = prompt_for_vis['action'].shape[1]
                prompt_poses = prompt_poses.repeat(chunk_n_action, axis=0)
                _p = Process(target=vis_trajectories, args=(os.path.join(vis_out_dir, f"proprio_prompt_{vis_demo_name}.mp4"), prompt_poses), kwargs={'axis_every_n_steps': 1})
                _p.start()
                _p.join()

                # pad the poses
                n_pad_prompt = len(first_proprio)
                n_pad_rollout = len(prompt_poses)
                prompt_poses_padded = np.concatenate([prompt_poses, np.tile(prompt_poses[-1:], (n_pad_prompt, 1, 1))], axis=0)
                rollout_poses_padded = np.concatenate([np.tile(rollout_poses[:1], (n_pad_rollout, 1, 1)), rollout_poses], axis=0)

                _p = Process(target=vis_trajectories, args=(os.path.join(vis_out_dir, f"proprio_prompt_and_rollout_{vis_demo_name}.mp4"), [prompt_poses_padded, rollout_poses_padded]), kwargs={'axis_every_n_steps': [1,10]})
                _p.start()
                _p.join()

        # log
        log_data = dict()
        max_rewards = collections.defaultdict(list)
        is_first_train_video = True
        is_first_test_video = True
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            rewards_prefix = 'score/' + prefix
            max_rewards[rewards_prefix].append(max_reward)
            log_data[rewards_prefix + str(seed)] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                # because we do the gripper open action, the video will start with the gripper opening. We want to cut these frames from the video so that we have alignment between the policy actions taken and the video frames
                assert self.num_open_gripper_steps % self.steps_per_render == 0, 'num_open_gripper_steps must be divisible by steps_per_render'
                cut_first_n_frames(video_path, self.num_open_gripper_steps // self.steps_per_render, skip_if_no_frames=True, skip_if_too_few_frames=True) # sometimes the video perhaps isn't fully flushed which was causing the video reader to not find any frames. In this case just skip cutting the gripper open frames, it's not that important. Also sometimes the init state for the environment will already somehow have the object in the correct target location based on the randomization which makes the video have very few frames. In this case just skip cutting the gripper open frames, it's not that important.

                sim_video = wandb.Video(video_path, format="mp4")
                videos_prefix = 'videos/' + prefix
                log_data[videos_prefix + str(seed)] = sim_video

                log_as_single = False
                if prefix.startswith('train/') and is_first_train_video:
                    log_as_single = True
                    is_first_train_video = False
                elif prefix.startswith('test/') and is_first_test_video:
                    log_as_single = True
                    is_first_test_video = False

                if log_as_single:
                    videos_prefix = 'videos_single/' + prefix
                    log_data[videos_prefix + str(seed)] = sim_video
        
        # visualize attention map
        if vis_attention_map:
            prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index)
            prompt_for_vis = move_batch_to_numpy(prompt_for_vis)
            rgb_keys = [key for key in self.sim_rgb_keys]
            attention_and_rollout_vis_path = os.path.join(vis_out_dir, f"attention_and_rollout_{vis_demo_name}.mp4")

            attn_logger.vis(attention_and_rollout_vis_path, prompt_for_vis, all_video_paths[self.vis_env_index], rgb_keys, steps_per_render=self.steps_per_render, exec_action_horizon=self.exec_action_horizon, fps=self.fps, cmap=self.attention_map_cmap, save_weights=self.save_attention_weights, scale_attention_weights=self.scale_attention_weights, use_proportion_of_colormap=self.use_proportion_of_colormap, stack_prompt_imgs="off", rollout_width_fraction=1/2)

            print(f'Saved attention and rollout visualization to {attention_and_rollout_vis_path}')

            seed = self.env_seeds[self.vis_env_index]
            prefix = self.env_prefixs[self.vis_env_index]
            sim_video = wandb.Video(attention_and_rollout_vis_path, format="mp4")
            attention_map_prefix = 'attention_map/' + prefix
            log_data[attention_map_prefix + str(seed)] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            prefix = prefix.replace('score/', 'mean_score/')
            name = prefix[:-1]
            value = np.mean(value)
            log_data[name] = value

        return log_data
    
    def close(self):
        self.env.close()

    def transform_action_for_policy(self, action, undo=False):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        if undo:
            rot = self.rotation_transformer_axisangle_to_rot6d.inverse(rot)
        else:
            rot = self.rotation_transformer_axisangle_to_rot6d.forward(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
