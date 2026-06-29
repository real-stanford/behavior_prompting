import os
import hashlib
import time
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import cv2
from typing import Optional
from torch.utils.data import Subset, ConcatDataset
import wandb.sdk.data_types.video as wv
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import get_training_task_names_from_training_split_info
from behavior_prompting.train_network.dataset.draw_image_dataset import DrawImageDataset
from behavior_prompting.train_network.env.draw.draw_env import DrawEnv
from behavior_prompting.train_network.gym_util.async_vector_env import AsyncVectorEnv
from behavior_prompting.train_network.gym_util.multistep_wrapper import MultiStepWrapper
from behavior_prompting.train_network.gym_util.video_recording_wrapper import VideoRecordingWrapper
from behavior_prompting.train_network.utils.dataset_util import pad_dataset_to_length
from behavior_prompting.train_network.utils.plot_util import PromptAttentionLogger
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder
from behavior_prompting.common.pytorch_util import add_batch_dim, move_batch_to_device, move_batch_to_numpy, remove_batch_dim_from_prompt

from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.env_runner.base_runner import BaseRunner
from behavior_prompting.train_network.utils.draw_util import vis_prompt
from behavior_prompting.train_network.utils.prompt_util import collate_prompts

from behavior_prompting.train_network.env.draw.draw_env import BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH

def task_name_to_boundary_angle(task_name: str) -> float:
    """Map task name to a deterministic boundary angle in [BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH]."""
    h = int(hashlib.sha256(task_name.encode()).hexdigest(), 16)
    t = (h % 1000000) / 1000000.0  # [0, 1)
    return BOUNDARY_ANGLE_LOW + t * (BOUNDARY_ANGLE_HIGH - BOUNDARY_ANGLE_LOW)

def get_draw_env(shape_meta, n_train, n_test, boundary_angle, fps, crf, exec_action_horizon, max_steps, draw_env=None, use_async_vector_env=True, overlay_action_cross=True, overlay_reward=True, overlay_target_drawing=True, render_cache_size=None, **kwargs):
    n_envs = n_train + n_test
    steps_per_render = 1

    use_prompting = shape_meta['use_prompting']
    prompt_sample_mode = shape_meta['prompt_sample_mode'] if use_prompting else None

    if prompt_sample_mode == 'sequence':
        # sequence prompting policies need to get all observations that occur during the rollout so we need to have the env runner recording observations for the entire batch of the past actions executed for each step
        max_obs_horizon = exec_action_horizon
    else:
        # for other policies (non-prompting or pair prompting) we need to record history of observations with length of the longest observation horizon
        obs_horizons = [attr['horizon'] for attr in shape_meta['obs'].values()]
        max_obs_horizon = max(obs_horizons)

    def env_fn():
        return MultiStepWrapper(
            VideoRecordingWrapper(
                DrawEnv(
                    boundary_angle=boundary_angle,
                    overlay_action_cross=overlay_action_cross,
                    overlay_reward=overlay_reward,
                    overlay_target_drawing=overlay_target_drawing,
                    render_cache_size=render_cache_size
                ) if draw_env is None else draw_env,
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='AUTO',
                    thread_count=0
                ),
                mode='rgb_array' if draw_env is None else draw_env.mode,
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=max_obs_horizon,
            n_action_steps=exec_action_horizon,
            max_episode_steps=max_steps
        )

    env_fns = [env_fn] * n_envs
    if use_async_vector_env:
        env = AsyncVectorEnv(env_fns)
    else:
        assert n_envs == 1, 'if use_async_vector_env is False, then n_envs must be 1'
        env = env_fns[0]()
    return env

class DrawRunner(BaseRunner):
    def __init__(self,
            output_dir,
            env: AsyncVectorEnv,
            replay_buffer: ReplayBuffer,
            task_name,
            shape_meta: dict,
            is_eval_dataset: bool,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            fps=10,
            crf=22,
            tqdm_interval_sec=5.0,
            boundary_angle=None,
            set_boundary_angle_from_task_name: bool=False,
            exec_action_horizon=12,
            vis_prompt: bool=False,
            vis_attention_map: bool=False,
            attention_map_version: str='v2',
            vis_goal_image: bool=False,
            vis_env_index: int=0,
            overlay_action_cross: bool=True, # used in get_draw_env
            overlay_reward: bool=True, # used in get_draw_env
            overlay_target_drawing: bool=True, # used in get_draw_env
            render_cache_size: Optional[int]=None, # used in get_draw_env
            save_attention_weights: bool=False,
            save_canvas_step: Optional[int]=None,
            restore_canvas_step: Optional[int]=None
        ):
        super().__init__(output_dir)
        n_envs = n_train + n_test

        vis_task_name = task_name.replace(' ', '_')
        assert vis_task_name.startswith('draw_'), 'vis_task_name must start with "draw_"'
        end_of_task_name = vis_task_name[len('draw_'):]
        if end_of_task_name.upper() == end_of_task_name:
            vis_task_name += '_upper'
        elif end_of_task_name.lower() == end_of_task_name:
            vis_task_name += '_lower'
        else:
            assert False
        self.vis_task_name = vis_task_name
        
        self.use_prompting = shape_meta['use_prompting']
        self.use_goal_image = 'goal_image' in shape_meta['obs'] and not self.use_prompting
        self.prompt_sample_mode = shape_meta['prompt_sample_mode'] if self.use_prompting else None
        assert exec_action_horizon <= shape_meta['action']['horizon'], 'number of steps to execute must not be greater than the action horizon predicted by the policy'
        
        self.env = env
        self.is_async_vector_env = isinstance(env, AsyncVectorEnv)
        self.n_envs = n_envs
        self.fps = fps
        self.crf = crf
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.task_name = task_name
        self.set_boundary_angle_from_task_name = set_boundary_angle_from_task_name
        assert not (set_boundary_angle_from_task_name and boundary_angle is not None), (
            'set_boundary_angle_from_task_name and boundary_angle cannot both be set; '
            'the former overwrites the latter at run time.')
        self.replay_buffer = replay_buffer
        self.exec_action_horizon = exec_action_horizon
        self.vis_prompt = vis_prompt
        self.prompt_sequence_length = shape_meta['prompt_sequence_length']
        self.vis_attention_map = vis_attention_map
        self.attention_map_version = attention_map_version
        self.vis_env_index = vis_env_index
        self.shape_meta = shape_meta
        self.is_eval_dataset = is_eval_dataset
        self.n_train = n_train
        self.n_test = n_test
        self.n_train_vis = n_train_vis
        self.n_test_vis = n_test_vis
        self.train_start_seed = train_start_seed
        self.test_start_seed = test_start_seed
        self.vis_goal_image = vis_goal_image
        self.save_attention_weights = save_attention_weights
        self.save_canvas_step = save_canvas_step
        self.restore_canvas_step = restore_canvas_step

        self._validate_canvas_step(self.save_canvas_step, 'save_canvas_step')
        self._validate_canvas_step(self.restore_canvas_step, 'restore_canvas_step')
        if save_canvas_step is not None and restore_canvas_step is not None:
            assert save_canvas_step < restore_canvas_step, 'save_canvas_step must be less than restore_canvas_step'

        self.rgb_keys = []
        self.lowdim_keys = []
        for key in shape_meta['obs']:
            if shape_meta['obs'][key]['type'] == 'rgb':
                self.rgb_keys.append(key)
            elif shape_meta['obs'][key]['type'] == 'low_dim':
                self.lowdim_keys.append(key)

    def run(self, policy: BasePolicy, enable_expensive_vis: bool=True):
        print(f"\n=== Started DrawRunner run for task \"{self.task_name}\" ===")
        if not policy.supports_prompting() or self.prompt_sample_mode == 'sequence':
            self.vis_attention_map = False

        vis_attention_map = self.vis_attention_map and enable_expensive_vis

        # if is_eval_dataset is True, then we need to make sure we didn't train on this task
        training_task_names = get_training_task_names_from_training_split_info(policy.get_training_split_info())
        if self.is_eval_dataset:
            assert self.task_name not in training_task_names, f"task \"{self.task_name}\" should not be in training dataset when doing eval run"
        else:
            assert self.task_name in training_task_names, f"task \"{self.task_name}\" should be in training dataset when doing train run"

        device = policy.device
        dtype = policy.dtype
        env = self.env

        vis_out_dir = os.path.join(self.output_dir, "vis")
        os.makedirs(vis_out_dir, exist_ok=True)

        # plan for rollout
        # TODO: we currently don't support multiple chunks yet so we will have evaluate all environments at once. This could be fixed later.
        n_chunks = 1
        n_envs = self.n_envs
        n_inits = n_envs

        # allocate data
        all_video_paths = [None] * n_envs
        all_rewards = [None] * n_envs

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            policy.reset(action_exec_horizon=self.exec_action_horizon)

            if self.use_goal_image:
                start_time = time.time()
                # load the dataset replay buffer
                train_goal_image_dataset = DrawImageDataset(
                    shape_meta=self.shape_meta,
                    replay_buffer=self.replay_buffer,
                    seed=0,
                    sample_type='task',
                    only_goal_image=True,
                    training_split_info=policy.get_training_split_info() if not self.is_eval_dataset else None,
                    val_ratio=0,
                    only_task_names=[self.task_name]
                )
                orig_goal_image_dataset = train_goal_image_dataset

                # if train dataset, then we have trained on some demonstrations and have some validation demonstrations. So we use training demonstration goal images for train environments and validation goal images for test environments
                # if eval dataset, then we have not trained on any demonstrations, so we use all the goal images for both train and test environments
                if self.is_eval_dataset:
                    # expand train dataset to match the number of envs
                    goal_image_dataset = pad_dataset_to_length(train_goal_image_dataset, n_envs)
                else:
                    test_goal_image_dataset = train_goal_image_dataset.get_validation_dataset()
                    # expand datasets if needed to match the number of envs
                    train_goal_image_dataset = pad_dataset_to_length(train_goal_image_dataset, self.n_train)
                    test_goal_image_dataset = pad_dataset_to_length(test_goal_image_dataset, self.n_test)

                def get_goal_images_from_dataset(dataset: DrawImageDataset, n_envs: int) -> torch.Tensor:
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=n_envs,
                        shuffle=False,
                        num_workers=0 # all the data is loaded into one batch so we don't need to use multiple workers
                    )
                    batch = next(iter(dataloader))
                    batch = move_batch_to_device(batch, device=device)
                    goal_images_batched = batch['obs']['goal_image'] # (n_envs, 1, 3, 224, 224)

                    target_drawings_batched = []
                    target_boundary_angles_batched = []
                    for i in range(n_envs):
                        target_drawing, boundary_angle = orig_goal_image_dataset.get_target_drawing_image_for_task_idx(batch['metadata']['goal_image_task_idx'][i].item())
                        target_drawings_batched.append(target_drawing)
                        target_boundary_angles_batched.append(boundary_angle)
                    target_drawings_batched = np.array(target_drawings_batched) # (n_envs, 512, 512, 3)
                    target_boundary_angles_batched = np.array(target_boundary_angles_batched) # (n_envs,)

                    return goal_images_batched, target_drawings_batched, target_boundary_angles_batched

                if self.is_eval_dataset:
                    goal_images, target_drawings, boundary_angles = get_goal_images_from_dataset(goal_image_dataset, n_envs)
                    train_goal_images = goal_images[:self.n_train]
                    test_goal_images = goal_images[self.n_train:]
                    train_target_drawings = target_drawings[:self.n_train]
                    test_target_drawings = target_drawings[self.n_train:]
                    train_boundary_angles = boundary_angles[:self.n_train]
                    test_boundary_angles = boundary_angles[self.n_train:]
                else:
                    train_goal_images, train_target_drawings, train_boundary_angles = get_goal_images_from_dataset(train_goal_image_dataset, self.n_train)
                    if self.n_test > 0:
                        test_goal_images, test_target_drawings, test_boundary_angles = get_goal_images_from_dataset(test_goal_image_dataset, self.n_test)
                    else:
                        test_goal_images = train_goal_images.new_empty(0, *train_goal_images.shape[1:])
                        test_target_drawings = np.array([], dtype=np.object_)
                        test_boundary_angles = np.array([])

                    goal_images = torch.cat([train_goal_images, test_goal_images], dim=0).to(device)
                    assert len(goal_images) == n_envs, 'the number of goal images must be the same as the number of envs'

                end_time = time.time()
                print(f"Time taken to load goal images: {end_time - start_time} seconds")
            elif self.use_prompting:
                start_time = time.time()
                train_prompt_dataset = DrawImageDataset(
                    shape_meta=self.shape_meta,
                    replay_buffer=self.replay_buffer,
                    seed=0,
                    sample_type='task',
                    only_prompt=True,
                    training_split_info=policy.get_training_split_info() if not self.is_eval_dataset else None,
                    val_ratio=0, # used only when self.is_eval_dataset is True. In this case we haven't seen any of the demos during training, so we just consider any of the demos to be eligible for prompting
                    only_task_names=[self.task_name]
                )

                orig_train_prompt_dataset = train_prompt_dataset

                if self.prompt_sample_mode == 'sequence':
                    # since sequence prompting inference code requires that all prompts are the same length, it's unfortuantely challenging to use validation prompts for test envs or even different prompts for different environments. Thus we just use the training prompt for both train and test envs
                    print('WARNING: for sequence prompting we only support using a training prompt for both train and test envs! We do not support using validation prompts for the test envs!')
                    prompt_sample_index = 0
                    prompt_dataset_to_sample = Subset(train_prompt_dataset, [prompt_sample_index] * n_envs)
                else:
                    # for pair prompting we use the training prompt for train envs and the validation prompt for test envs if is_eval_dataset is False. If is_eval_dataset is True, then all the demonstrations are unseen so we use all the prompts for both train and test envs
                    assert self.prompt_sample_mode == 'pair'

                    if self.is_eval_dataset:
                        # one dataset that contains all the prompts for the unseen task
                        prompt_dataset_to_sample = train_prompt_dataset
                        prompt_dataset_to_sample = pad_dataset_to_length(prompt_dataset_to_sample, n_envs)
                    else:
                        # separate datasets for train and test envs with training and validation prompts, respectively
                        test_prompt_dataset = train_prompt_dataset.get_validation_dataset()
                        train_prompt_dataset = pad_dataset_to_length(train_prompt_dataset, self.n_train)
                        test_prompt_dataset = pad_dataset_to_length(test_prompt_dataset, self.n_test)
                        prompt_dataset_to_sample = ConcatDataset([train_prompt_dataset, test_prompt_dataset])

                print('Starting to load prompts and then going to prompt model...')
                prompt_dataloader = torch.utils.data.DataLoader(
                    prompt_dataset_to_sample,
                    batch_size=n_envs,
                    shuffle=False,
                    num_workers=0, # we just load one batch from each dataloader we create so we don't need to use multiple workers
                    collate_fn=collate_prompts
                )
                prompts_batched = next(iter(prompt_dataloader))
                prompts_batched = move_batch_to_device(prompts_batched, device=device)
                prompts_batched = prompts_batched['obs']['prompt']
                policy.prompt(prompts_batched)

                end_time = time.time()
                print(f"Finished prompting model! Time taken: {end_time - start_time} seconds")

                # determine target drawings corresponding to the prompts
                train_target_drawings = []
                train_boundary_angles = []
                test_target_drawings = []
                test_boundary_angles = []
                for i in range(n_envs):
                    task_idx = prompts_batched['metadata']['task_indices'][i].item()
                    target_drawing, boundary_angle = orig_train_prompt_dataset.get_target_drawing_image_for_task_idx(task_idx)
                    if i < self.n_train:
                        train_target_drawings.append(target_drawing)
                        train_boundary_angles.append(boundary_angle)
                    else:
                        test_target_drawings.append(target_drawing)
                        test_boundary_angles.append(boundary_angle)

                if self.vis_prompt:
                    prompt_for_vis = move_batch_to_numpy(remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index))

                    vis_prompt(prompt_for_vis, os.path.join(vis_out_dir, f"prompt_{self.vis_task_name}.mp4"))

                total_available_action_steps_after_prompting = policy.num_available_actions()
                if total_available_action_steps_after_prompting is not None and total_available_action_steps_after_prompting < self.max_steps:
                    print(f"WARNING: the number of steps we are able to execute is less than the max_steps specified in the env_runner. This is because the model is prompted. This leaves only {total_available_action_steps_after_prompting} to execute which is less than the specified max steps of {self.max_steps}")
                    # TODO: probably should make this an error rather than a warning

            assert len(train_target_drawings) == self.n_train, 'the number of target drawings must be the same as the number of train envs'
            assert len(test_target_drawings) == self.n_test, 'the number of target drawings must be the same as the number of test envs'
            assert len(train_boundary_angles) == self.n_train, 'the number of boundary angles must be the same as the number of train envs'
            assert len(test_boundary_angles) == self.n_test, 'the number of boundary angles must be the same as the number of test envs'
    
            # === init envs ===
            boundary_angle_from_task = task_name_to_boundary_angle(self.task_name) if self.set_boundary_angle_from_task_name else None
            env_seeds = list()
            env_prefixs = list()
            env_init_fn_dills = list()
            
            # train
            for i in range(self.n_train):
                seed = self.train_start_seed + i
                enable_render = i < self.n_train_vis
                target_drawing = train_target_drawings[i]
                target_boundary_angle = train_boundary_angles[i]
                output_dir = self.output_dir
                vis_task_name = self.vis_task_name

                def init_fn(env, seed=seed, enable_render=enable_render, target_drawing=target_drawing, target_boundary_angle=target_boundary_angle, boundary_angle_from_task=boundary_angle_from_task):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f'train_{vis_task_name}_{wv.util.generate_id()}.mp4')
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    if boundary_angle_from_task is not None:
                        env.env.env.boundary_angle = boundary_angle_from_task
                        env.env.env.randomize_boundary_angle = False

                    # set target drawing
                    env.env.env.set_target_drawing(target_drawing, target_boundary_angle)

                    # TODO: for train environments theoretically we should set the board angle and cursor position to be from the start of a training demonstration

                    # set seed
                    assert isinstance(env, MultiStepWrapper)
                    env.seed(seed)
                
                env_seeds.append(seed)
                env_prefixs.append(f'train/{self.vis_task_name}_')
                env_init_fn_dills.append(dill.dumps(init_fn))

            # test
            for i in range(self.n_test):
                seed = self.test_start_seed + i
                enable_render = i < self.n_test_vis
                target_drawing = test_target_drawings[i]
                target_boundary_angle = test_boundary_angles[i]
                output_dir = self.output_dir
                vis_task_name = self.vis_task_name

                def init_fn(env, seed=seed, enable_render=enable_render, target_drawing=target_drawing, target_boundary_angle=target_boundary_angle, boundary_angle_from_task=boundary_angle_from_task):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f'test_{vis_task_name}_{wv.util.generate_id()}.mp4')
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    if boundary_angle_from_task is not None:
                        env.env.env.boundary_angle = boundary_angle_from_task
                        env.env.env.randomize_boundary_angle = False

                    # set target drawing
                    env.env.env.set_target_drawing(target_drawing, target_boundary_angle)

                    # set seed
                    assert isinstance(env, MultiStepWrapper)
                    env.seed(seed)
                
                env_seeds.append(seed)
                env_prefixs.append(f'test/{self.vis_task_name}_')
                env_init_fn_dills.append(dill.dumps(init_fn))
            self.env_seeds = env_seeds
            self.env_prefixs = env_prefixs
            self.env_init_fn_dills = env_init_fn_dills

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs using the target drawings
            if self.is_async_vector_env:
                env.call_each('run_dill_function', 
                    args_list=[(x,) for x in this_init_fns])
            else:
                assert n_envs == 1, 'if use_async_vector_env is False, then n_envs must be 1'
                init_fn = dill.loads(this_init_fns[0])
                init_fn(env)

            # === start rollout ===
            obs = env.reset()

            pbar = tqdm.tqdm(total=min(self.max_steps, total_available_action_steps_after_prompting) if self.use_prompting and total_available_action_steps_after_prompting is not None else self.max_steps, desc=f"Eval \"{self.task_name}\" DrawRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            pred_steps = 0
            time_since_cursor_not_moved = 0
            cursor_start_pos = None
            if vis_attention_map:
                attn_logger = PromptAttentionLogger(policy, self.vis_env_index)
            save_canvas_done = False
            restore_canvas_done = False

            # handle special case where save_canvas_step is 0
            if self.save_canvas_step == 0:
                self._invoke_env_method('save_canvas_state')
                save_canvas_done = True

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)

                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                if not self.is_async_vector_env:
                    obs_dict = add_batch_dim(obs_dict)
                
                # if goal image is used, add it to the obs dict
                if self.use_goal_image:
                    obs_dict['goal_image'] = goal_images

                # run policy
                with torch.inference_mode():
                    kwargs = dict()
                    if vis_attention_map:
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = True
                    action_dict = policy.predict_action(obs_dict, **kwargs)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                action = action[:, :self.exec_action_horizon]

                if vis_attention_map:
                    attn_logger.log(action_dict)

                if self.is_async_vector_env:
                    action_for_policy = action
                else:
                    action_for_policy = action[0]

                # step env
                obs, reward, done, info = env.step(action_for_policy)
                done = np.all(done)

                # update pbar
                pbar.update(action.shape[1])
                pred_steps += action.shape[1]

                # handle canvas state saving and restoring
                if (self.save_canvas_step is not None
                    and not save_canvas_done
                    and pred_steps >= self.save_canvas_step
                ):
                    self._invoke_env_method('save_canvas_state')
                    save_canvas_done = True

                if (self.restore_canvas_step is not None
                    and not restore_canvas_done
                    and pred_steps >= self.restore_canvas_step
                ):
                    assert save_canvas_done, 'save_canvas_step must be before restore_canvas_step'
                    self._invoke_env_method('skip_next_video_reset') # we don't want the video recorder to stop, so we do special handling here
                    obs = env.reset() # this reset call will restore the canvas state to the saved state rather than performing a full reset because we previously called save_canvas_state
                    restore_canvas_done = True

                # handle early stop for prior baseline prompt models that have a fixed sequence length
                if self.use_prompting and total_available_action_steps_after_prompting is not None:
                    # prompting models have a fixed sequence length so we have to stop once we reach the sequence length
                    if pred_steps >= total_available_action_steps_after_prompting:
                        done = True

                # if none of the envs have moved the cursor by a significant amount, then we should stop early # TODO: it would make sense to do this within the environment itself since that would allow each environment to independently stop rather than having to wait until all environments are stopped
                if time_since_cursor_not_moved == 0:
                    cursor_start_pos = action[:, 0, :2] # action is (B, T, 3) -> (B, 2)
                
                dist_from_start = np.linalg.norm(action[:,:,:2] - np.expand_dims(cursor_start_pos, axis=1), axis=2) # (B, T)
                not_moved = dist_from_start < 20 # 20 pixels distance from start is what we consider movement
                if np.all(not_moved):
                    time_since_cursor_not_moved += action.shape[1]
                else:
                    time_since_cursor_not_moved = 0
                
                if time_since_cursor_not_moved >= 20: # 20 steps at 10Hz is 2 seconds. If the cursor hasn't moved for 2 seconds, then we stop
                    done = True

                # for the canvas save/restore feature if we reset the env then the multi step wrapper will reset its count of the steps so the check for max steps within the wrapper won't properly work, so we need to manually handle it here
                if self.restore_canvas_step is not None and pred_steps >= self.max_steps:
                    done = True

            pbar.close()

            if self.is_async_vector_env:
                all_video_paths[this_global_slice] = env.render()[this_local_slice]
                all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            else:
                all_video_paths[0] = env.render()
                all_rewards[0] = reward

            if self.use_prompting and total_available_action_steps_after_prompting is not None and total_available_action_steps_after_prompting < self.max_steps:
                assert policy.num_available_actions() < self.exec_action_horizon, 'if we are limited by the number of steps we can execute, then the number of steps we can execute should be less than the number of steps we can predict at the very end of execution'

        # clear out video buffer
        _ = env.reset()
        # clear out policy buffers
        policy.reset(action_exec_horizon=self.exec_action_horizon)

        log_data = dict()
        vis_demo_name = os.path.basename(all_video_paths[self.vis_env_index]).replace('.mp4', '')

        # TODO add visualization for goal image and rollout for goal image condition policy to verify that goal image orientation doesn't need to match rollout target drawing orientation

        # visualize attention map
        if vis_attention_map:
            prompt_for_vis = remove_batch_dim_from_prompt(prompts_batched, index_to_keep=self.vis_env_index)
            prompt_for_vis = move_batch_to_numpy(prompt_for_vis)
            rgb_keys = [key for key in self.rgb_keys if key != 'goal_image']
            attention_and_rollout_vis_path = os.path.join(vis_out_dir, f"attention_and_rollout_{vis_demo_name}.mp4")

            attn_logger.vis(
                attention_and_rollout_vis_path,
                prompt_for_vis,
                all_video_paths[self.vis_env_index],
                rgb_keys,
                steps_per_render=1,
                exec_action_horizon=self.exec_action_horizon,
                fps=self.fps,
                version=self.attention_map_version,
                save_weights=self.save_attention_weights,
            )

            print(f'Saved attention and rollout visualization to {attention_and_rollout_vis_path}')

            seed = self.env_seeds[self.vis_env_index]
            prefix = self.env_prefixs[self.vis_env_index]
            sim_video = wandb.Video(attention_and_rollout_vis_path, format="mp4")
            attention_map_prefix = prefix.replace('train/', 'train/attention_map/').replace('test/', 'test/attention_map/')
            log_data[attention_map_prefix + f'seed_{seed}'] = sim_video

        # save goal images
        if self.vis_goal_image:
            for i in range(n_envs):
                seed = self.env_seeds[i]
                video_path = all_video_paths[i]
                if video_path is not None:
                    # Get base video name without extension
                    base_name = os.path.basename(video_path).replace('.mp4', '')
                    
                    goal_image = (goal_images[i].detach().cpu().numpy() * 255).astype(np.uint8)[0]
                    
                    # Save goal image with matching name in vis folder
                    goal_img_path = os.path.join(vis_out_dir, f"goal_{base_name}.png")
                    # Convert from CxHxW to HxWxC if needed and save
                    if goal_image.shape[0] == 3:
                        goal_image = goal_image.transpose(1,2,0)
                    
                    cv2.imwrite(goal_img_path, goal_image[:,:,::-1])

        # log
        max_rewards = collections.defaultdict(list)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            rewards_prefix = prefix.replace('train/', 'train/rewards/').replace('test/', 'test/rewards/')
            max_rewards[rewards_prefix].append(max_reward)
            log_data[rewards_prefix + f'seed_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path, format="mp4")
                videos_prefix = prefix.replace('train/', 'train/videos/').replace('test/', 'test/videos/')
                log_data[videos_prefix + f'seed_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        print(f"=== Finished DrawRunner run for task \"{self.task_name}\" ===\n")

        return log_data

    def _validate_canvas_step(self, value: Optional[int], name: str):
        if value is None:
            return
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        if value % self.exec_action_horizon != 0:
            raise ValueError(
                f"{name} ({value}) must be a multiple of exec_action_horizon ({self.exec_action_horizon})."
            )

    def _invoke_env_method(self, method_name: str, *args, **kwargs):
        """Call a method on every underlying env, regardless of vectorization."""
        if self.is_async_vector_env:
            return self.env.call(method_name, *args, **kwargs)
        method = getattr(self.env, method_name)
        return method(*args, **kwargs)
