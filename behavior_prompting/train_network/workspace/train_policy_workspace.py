import os
from pathlib import Path
import shutil
import time
from typing import Optional
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, ConcatDataset
import copy
import random
import pickle
import tqdm
import numpy as np
import accelerate
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from accelerate.utils import send_to_device
from diffusers.training_utils import EMAModel

from behavior_prompting.train_network.workspace.base_workspace import BaseWorkspace
from behavior_prompting.train_network.dataset.batched_dataset import BatchedByTaskDataset
from behavior_prompting.train_network.utils.load_env import load_env_runner, env_rollout
from behavior_prompting.train_network.common.checkpoint_util import TopKCheckpointManager
from behavior_prompting.train_network.model.common.lr_scheduler import get_scheduler
from behavior_prompting.train_network.utils.training_utils import validate_optimizer_parameters
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.utils.prompt_util import collate_prompts

class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.model)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes

        timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=120)) # two hour timeout on multi GPU NCCL (default is 30 min). This helps with long setup before training where processes have to wait for env_runners to initialize
        dynamo_plugin = hydra.utils.instantiate(cfg.training.compilation_plugin)
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[timeout], mixed_precision=cfg.training.mixed_precision, gradient_accumulation_steps=cfg.training.gradient_accumulate_every, dynamo_plugin=dynamo_plugin)
        
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # ensure all processes use the same output directory
        output_dirs = accelerate.utils.gather_object([self.output_dir] if accelerator.is_main_process else [''])
        main_output_dir = [x for x in output_dirs if x][0]
        self._output_dir = main_output_dir
        accelerator.print(f'Started training. Run dir: {self.output_dir}')

        # configure optimizer
        for key, value in cfg.optimizer.items():
            if key == 'lr' or key == 'obs_encoder_lr':
                cfg.optimizer[key] *= accelerator.num_processes # scale LR by num GPUs; see see https://huggingface.co/docs/accelerate/concept_guides/performance

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        validate_optimizer_parameters(self.optimizer, self.model)

        # configure training state
        self.global_step = 0
        self.num_grad_steps = 0
        self.epoch = 0
        self.num_rollout_completed = 0

        # Initialize the dataset
        print(f'Initializing dataset...')
        start_time = time.time()
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        end_time = time.time()
        print(f'Time taken to initialize dataset: {end_time - start_time} seconds')

        training_split_info = dataset.get_training_split_info()
        self.model.register_training_split_info(training_split_info)

        # configure env runners
        if cfg.training.rollout_every != -1 and (cfg.rollout.distribute or accelerator.is_main_process):
            accelerator.print('Starting to load env runners')
            start_time = time.time()
            env_runners = load_env_runner(cfg, self.output_dir, dataset, accelerator=accelerator if cfg.rollout.distribute else None)
            end_time = time.time()
            accelerator.print(f"Time taken to load env runners: {end_time - start_time} seconds")

        # configure train and validation datasets
        prompt_dataset: Optional[BaseDataset] = None
        if cfg.training.group_dataloader_by_task.enabled and cfg.task.shape_meta.use_prompting and cfg.task.shape_meta.prompt_sample_mode == 'pair':
            prompt_dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
            prompt_dataset_cfg['only_prompt'] = True
            prompt_dataset_cfg['training_split_info'] = training_split_info # we need to pass the training split info to the prompt dataset so that it can sample prompts from the same training split as the dataset
            prompt_dataset_cfg['replay_buffer'] = dataset.replay_buffer # reuse the replay buffer from the dataset
            prompt_dataset_cfg['dataset_path'] = None # no need for the path since we provide the replay buffer directly
            start_time = time.time()
            accelerator.print(f'Loading prompt dataset...')
            prompt_dataset = hydra.utils.instantiate(prompt_dataset_cfg)
            end_time = time.time()
            accelerator.print(f'Time taken to load prompt dataset: {end_time - start_time} seconds')
        
        start_time = time.time()
        accelerator.print(f'Loading validation dataset...')
        val_dataset: BaseDataset = dataset.get_validation_dataset()
        end_time = time.time()
        accelerator.print(f'Time taken to load validation dataset: {end_time - start_time} seconds')

        train_dataloader_cfg = {**cfg.dataloader, 'collate_fn': collate_prompts}
        val_dataloader_cfg = {**cfg.val_dataloader, 'collate_fn': collate_prompts}

        train_dataloader: DataLoader
        val_dataloader: DataLoader
        lr_scheduler = None
        train_action_mse_dataloader_by_task = None
        val_action_mse_dataloader_by_task = None
        def setup_dataloaders_for_epoch():
            nonlocal dataset, val_dataset, prompt_dataset, train_dataloader, val_dataloader, lr_scheduler, train_action_mse_dataloader_by_task, val_action_mse_dataloader_by_task

            if prompt_dataset is not None:
                dataset.set_ignore_prompt(True) # when we do pair prompting and grouping, we can just load one prompt per entire batch, so we tell the dataset to ignore the prompt and then load single prompts from a separate dataset

            # shuffle the dataset if it requires shuffling
            if dataset.requires_epoch_shuffle():
                if 'persistent_workers' in train_dataloader_cfg:
                    assert not train_dataloader_cfg['persistent_workers'], 'persistent workers must be disabled when shuffling the dataset. Since you are recreating the dataloaders each epoch it does not make sense to have persistent workers and will likely lead to memory leaks and unreleased workers'
                if 'persistent_workers' in val_dataloader_cfg:
                    assert not val_dataloader_cfg['persistent_workers']
                
                if self.epoch > 0:
                    # the dataset will already have an initial ordering from when it was created, so we only need to shuffle it after the first epoch
                    dataset.shuffle_data_ordering(seed=self.epoch)
                    val_dataset.shuffle_data_ordering(seed=self.epoch)

            if self.epoch == 0 or dataset.requires_epoch_shuffle():
                # need to recreate dataloaders
                if cfg.training.group_dataloader_by_task.enabled:
                    # the main idea is that we have an intermediate dataloader which loads batches from a single task. Then we wrap that in a torch DataLoader which will sample these batches (thus we use a batch size of 1 at the final dataloader level)
                    dataset_for_train_dataloader = BatchedByTaskDataset(dataset, train_dataloader_cfg, cfg.task.shape_meta, cfg.training.group_dataloader_by_task.balanced, prompt_dataset)
                    config_for_train_dataloader = train_dataloader_cfg.copy()
                    config_for_train_dataloader['batch_size'] = 1
                    config_for_train_dataloader['collate_fn'] = lambda x: x[0] # since there will only be a single item in the batch, we can just return the first item so we don't add an extra batch dimension
                else:
                    dataset_for_train_dataloader = dataset
                    config_for_train_dataloader = train_dataloader_cfg

                if cfg.training.repeat_dataset_n_times > 1:
                    dataset_for_train_dataloader = ConcatDataset([dataset_for_train_dataloader] * cfg.training.repeat_dataset_n_times)
                
                train_dataloader = DataLoader(dataset_for_train_dataloader, **config_for_train_dataloader)

                if cfg.training.max_val_steps is not None:
                    cur_steps = len(val_dataset) // val_dataloader_cfg['batch_size']
                    if cur_steps > cfg.training.max_val_steps:
                        rng = np.random.RandomState(seed=0)
                        selected_indices = rng.choice(len(val_dataset), size=cfg.training.max_val_steps * val_dataloader_cfg['batch_size'], replace=False)
                        dataset_for_val_dataloader = Subset(val_dataset, selected_indices)
                    else:
                        dataset_for_val_dataloader = val_dataset
                else:
                    dataset_for_val_dataloader = val_dataset

                if len(dataset_for_val_dataloader) == 0:
                    val_dataloader_cfg['shuffle'] = False # causes an error to shuffle if the dataset is empty

                val_dataloader = DataLoader(dataset_for_val_dataloader, **val_dataloader_cfg)

                if self.epoch == 0:
                    # configure lr scheduler (needs to happen before accelerator.prepare)
                    total_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
                    lr_scheduler = get_scheduler(
                        cfg.training.lr_scheduler,
                        optimizer=self.optimizer,
                        num_warmup_steps=int(cfg.training.lr_warmup_proportion * total_steps),
                        num_training_steps=total_steps,
                        # pytorch assumes stepping LRScheduler every epoch
                        # however huggingface diffusers steps it every batch
                        last_epoch=-1
                    )

                train_dataloader, val_dataloader = accelerator.prepare(
                    train_dataloader, val_dataloader
                )

                # setup the dataloaders for action MSE error
                if cfg.training.sample_every != -1 and accelerator.is_main_process:
                    def prepare_action_mse_dataloader_by_task(dataset: BaseDataset):
                        if len(dataset) == 0:
                            return {}

                        action_mse_dataloader_cfg = copy.deepcopy(val_dataloader_cfg) # we use val_dataloader cfg here because we want to use the batch size of the val_dataloader since we are doing evaluations
                        action_mse_dataloader_cfg['shuffle'] = False # always compute action MSE error on the same section of the dataset
                        action_mse_dataloader_cfg['persistent_workers'] = False # don't want a bunch of persistent workers for just the action MSE dataloaders
                        action_mse_dataloader_cfg['drop_last'] = False # if the sample size is smaller than the batch size, we don't want to drop the last batch

                        if dataset.is_multi_task():
                            unique_task_name_to_dataset_indices = dataset.get_unique_task_name_to_dataset_indices()

                            if cfg.training.sample_max_tasks is not None and len(unique_task_name_to_dataset_indices) > cfg.training.sample_max_tasks:
                                selected_task_names = sorted(list(unique_task_name_to_dataset_indices.keys()))
                                rng = np.random.RandomState(seed=0)
                                selected_task_names = rng.choice(selected_task_names, size=cfg.training.sample_max_tasks, replace=False)
                                unique_task_name_to_dataset_indices = {task_name: unique_task_name_to_dataset_indices[task_name] for task_name in selected_task_names}
                        else:
                            unique_task_name_to_dataset_indices = {}
                        unique_task_name_to_dataset_indices['all'] = list(range(len(dataset)))

                        # create dataloaders for each task and for 'all'
                        action_mse_dataloader_by_task = {}
                        for unique_task_name, dataset_indices in unique_task_name_to_dataset_indices.items():
                            rng = np.random.RandomState(seed=cfg.training.seed)
                            selected_indices = rng.choice(dataset_indices, size=cfg.training.sample_size, replace=True) # we set replace=True to handle the case where the dataset is smaller than the sample size even though this might mean we sample the same index multiple times
                            subset = Subset(dataset, selected_indices)
                            subset_dataloader = DataLoader(subset, **action_mse_dataloader_cfg)
                            action_mse_dataloader_by_task[unique_task_name] = subset_dataloader
                        return action_mse_dataloader_by_task

                    if prompt_dataset is not None:
                        dataset.set_ignore_prompt(False) # action MSE datasets need prompt

                    train_action_mse_dataloader_by_task = prepare_action_mse_dataloader_by_task(dataset)
                    val_action_mse_dataloader_by_task = prepare_action_mse_dataloader_by_task(val_dataset)

                    if prompt_dataset is not None:
                        dataset.set_ignore_prompt(True) # return to ignoring prompt for training
                
                accelerator.print(f'\nEpoch {self.epoch}. Recreated dataloaders')
                accelerator.print('train dataset length:', len(dataset), 'train dataloader length:', len(train_dataloader))
                accelerator.print('val dataset length:', len(val_dataset), 'val dataloader length:', len(val_dataloader), '\n')

        setup_dataloaders_for_epoch()

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            if cfg.training.use_cached_normalizer:
                cached_normalizer_path = Path(cfg.task.dataset.dataset_path).absolute().with_suffix('').with_suffix('').as_posix() + '_normalizer.pkl'
                if cfg.training.override_cached_normalizer and os.path.exists(cached_normalizer_path):
                    os.remove(cached_normalizer_path)
            if cfg.training.use_cached_normalizer and os.path.exists(cached_normalizer_path):
                shutil.copy(cached_normalizer_path, normalizer_path)
                accelerator.print(f'Loaded cached normalizer from {cached_normalizer_path}')
            else:
                accelerator.print(f'Computing normalizer on main process...')
                start_time = time.time()
                normalizer = dataset.get_normalizer()
                end_time = time.time()
                accelerator.print(f'Time taken to compute normalizer: {end_time - start_time} seconds')
                pickle.dump(normalizer, open(normalizer_path, 'wb'))

                if cfg.training.use_cached_normalizer:
                    shutil.copy(normalizer_path, cached_normalizer_path)
                    accelerator.print(f'Cached normalizer to {cached_normalizer_path}')

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)

        # configure checkpoint and make sure the monitor_key is valid
        old_monitor_key = cfg.checkpoint.topk.monitor_key
        old_mode = cfg.checkpoint.topk.mode
        if cfg.checkpoint.topk.monitor_key in ['train_mean_score', 'test_mean_score', 'eval_test_mean_score']:
            if cfg.checkpoint.topk.monitor_key == 'eval_test_mean_score' and cfg.task.eval_dataset_path is None:
                cfg.checkpoint.topk.monitor_key = 'test_mean_score'

            if cfg.training.rollout_every == -1:
                # invalid, try action MSE error
                split = 'val' if len(val_dataloader) > 0 else 'train'
                cfg.checkpoint.topk.monitor_key = f'{split}_action_mse_error_all'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use rollout every
                checkpoint_every = cfg.training.rollout_every
        if 'action_mse_error' in cfg.checkpoint.topk.monitor_key:
            if cfg.training.sample_every == -1:
                # invalid, try val loss
                cfg.checkpoint.topk.monitor_key = 'val_loss'
                cfg.checkpoint.topk.mode = 'min'
            elif cfg.checkpoint.topk.monitor_key == 'val_action_mse_error_all' and len(val_dataset) == 0:
                # invalid to use val, but valid to use train action mse error
                cfg.checkpoint.topk.monitor_key = 'train_action_mse_error_all'
                cfg.checkpoint.topk.mode = 'min'
                checkpoint_every = cfg.training.sample_every
            else:
                # valid, use sample every
                checkpoint_every = cfg.training.sample_every
        if cfg.checkpoint.topk.monitor_key == 'val_loss':
            if cfg.training.val_every == -1 or len(val_dataset) == 0:
                # invalid, try train loss
                cfg.checkpoint.topk.monitor_key = 'train_loss'
                cfg.checkpoint.topk.mode = 'min'
            else:
                # valid, use val every
                checkpoint_every = cfg.training.val_every
        if cfg.checkpoint.topk.monitor_key == 'train_loss':
            # train_loss is always a valid metric to checkpoint on
            checkpoint_every = 1

        if cfg.checkpoint.topk.monitor_key != old_monitor_key:
            accelerator.print(f'NOTE: updating checkpoint monitor key from `{old_monitor_key}` ({old_mode}) to `{cfg.checkpoint.topk.monitor_key}` ({cfg.checkpoint.topk.mode})')
            cfg.checkpoint.topk.format_str = cfg.checkpoint.topk.format_str.replace(old_monitor_key, cfg.checkpoint.topk.monitor_key)
        
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        ) if cfg.checkpoint.topk.monitor_key != 'train_loss' else None
        topk_manager_train_loss = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='train_loss',
            mode='min',
            k=cfg.checkpoint.topk.k,
            format_str='epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt'
        )

        # accelerator
        self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device

        # setup ema model
        ema_model: Optional[EMAModel] = hydra.utils.instantiate(cfg.ema, parameters=self.model.parameters()) if cfg.training.use_ema else None
        if ema_model is not None:
            ema_model.to(device)

        # ========= training loop ==========
        batch_frequency_by_task = dict()
        for local_epoch_idx in range(cfg.training.num_epochs):
            is_last_epoch = self.epoch == cfg.training.num_epochs - 1
            accelerator.wait_for_everyone() # since some of the evaluations can take a while (env rollout happens only on main process), halt all processes until it's finished to prevent timeout

            self.model.train()

            # ========= train for this epoch ==========
            if cfg.training.freeze_encoder:
                assert False, "this leverages model specific assumption that this variable exists. Can put this variable in the BaseModel class to address this" # TODO: fix this
                self.model.obs_encoder.eval()
                self.model.obs_encoder.requires_grad_(False)

            if self.epoch > 0:
                setup_dataloaders_for_epoch()

            step_log = dict()
            train_losses = list()
            batch_end_time = time.time()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    step_log = {}
                    batch_start_time = batch_end_time

                    with accelerator.accumulate(self.model):
                        # compute loss
                        loss = self.model(batch)
                        accelerator.backward(loss)

                        # clip and log grad norms
                        if accelerator.sync_gradients:
                            if cfg.training.clip_grad_norm:
                                step_log['grad_norm'] = accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad_norm).item()
                                step_log['grad_norm_clipped'] = min(step_log['grad_norm'], cfg.training.clip_grad_norm)

                            self.num_grad_steps += 1

                        # step optimizer
                        self.optimizer.step()
                        lr_scheduler.step()
                        self.optimizer.zero_grad()
                    
                    # update ema
                    if ema_model is not None and accelerator.sync_gradients:
                        ema_model.step(accelerator.unwrap_model(self.model).parameters())

                    # Calculate iterations per second for this batch
                    # Measure from end of previous batch to end of current batch
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    iterations_per_second = 1.0 / batch_time if batch_time > 0 else 0

                    # logging
                    loss_cpu = loss.item()
                    tepoch.set_postfix(loss=loss_cpu, refresh=False)
                    train_losses.append(loss_cpu)
                    step_log.update({
                        'train_loss': loss_cpu,
                        'global_step': self.global_step,
                        'grad_steps': self.num_grad_steps,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0],
                        'iterations_per_second': iterations_per_second
                    })

                    # log frequency by task if using grouped dataloader
                    if cfg.training.group_dataloader_by_task.enabled and cfg.training.group_dataloader_by_task.log_frequency:
                        task_name = batch['metadata']['task_name']
                        task_indices = batch['metadata']['task_idx']
                        for task_idx in task_indices:
                            key = f'task_frequency/{task_name}/task_{task_idx}'
                            batch_frequency_by_task[key] = batch_frequency_by_task.get(key, 0) + 1

                            key2 = f'task_frequency/{task_name}'
                            batch_frequency_by_task[key2] = batch_frequency_by_task.get(key2, 0) + 1
                        
                        if 'prompt' in batch['obs']:
                            prompt_task_idx = batch['obs']['prompt']['metadata']['task_indices'].item()
                            key = f'task_frequency/prompt/{task_name}/task_{prompt_task_idx}'
                            batch_frequency_by_task[key] = batch_frequency_by_task.get(key, 0) + 1

                            key2 = f'task_frequency/prompt/{task_name}'
                            batch_frequency_by_task[key2] = batch_frequency_by_task.get(key2, 0) + 1
                        step_log.update(batch_frequency_by_task)

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break
                    
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        accelerator.log(step_log, step=self.global_step)
                        self.global_step += 1

            if len(train_losses) > 0:
                del loss, batch
            
            # at the end of each epoch log epoch average train loss
            train_loss_average = np.mean(train_losses)
            step_log['train_loss_epoch_average'] = train_loss_average # also sets 'train_loss' to epoch average in case we checkpoint based on 'train_loss' we should be using the epoch average not just one batch

            self.optimizer.zero_grad() # ensure optimizer is not holding any gradients so that we free up GPU memory for rollout (we need this since we might not end on a gradient accumulation step where we step the model and then zero grads); this is here in the case that the final step is not a gradient update step (due to multiple steps of gradient accumulation)

            # ========= validation for this epoch (distributed manner) ==========
            self.model.eval()
            if cfg.training.val_every != -1 and ((self.epoch % cfg.training.val_every == 0) or is_last_epoch) and len(val_dataloader) > 0:
                with torch.inference_mode():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            loss = self.model(batch)
                            val_losses.append(loss.detach())

                    if len(val_losses) > 0:
                        del batch

                    # Stack losses and gather from all processes
                    val_losses_tensor = torch.stack(val_losses)  # shape: [num_batches_per_process]
                    gathered_losses = accelerator.gather_for_metrics(val_losses_tensor)  # shape: [total_batches_all_processes]
                    
                    if accelerator.is_main_process:
                        val_loss = gathered_losses.mean().item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # ========= EVALUATION ==========
            policy = accelerator.unwrap_model(self.model)
            policy.eval()
            if ema_model is not None:
                ema_model.store(policy.parameters())
                ema_model.copy_to(policy.parameters())

            # ========= rollout for this epoch (non-distributed manner) ==========
            if ((self.epoch % cfg.training.rollout_every) == 0 or is_last_epoch) and cfg.training.rollout_every != -1 and (cfg.rollout.distribute or accelerator.is_main_process):
                condition1 = cfg.training.rollout.enable_expensive_vis_every != -1 and self.num_rollout_completed % cfg.training.rollout.enable_expensive_vis_every == 0
                condition2 = cfg.training.rollout.enable_expensive_vis_last_epoch and is_last_epoch
                enable_expensive_vis = condition1 or condition2

                if is_last_epoch:
                    max_tasks = None
                    init_runner_kwargs = None
                else:
                    max_tasks = cfg.training.rollout.intermediate_max_tasks
                    init_runner_kwargs = cfg.training.rollout.intermediate_init_runner_kwargs
                    init_runner_kwargs = OmegaConf.to_container(init_runner_kwargs, resolve=True) if init_runner_kwargs is not None else None

                start_time = time.time()
                runner_log = env_rollout(cfg, env_runners, policy, enable_expensive_vis=enable_expensive_vis, accelerator=accelerator if cfg.rollout.distribute else None, max_tasks=max_tasks, init_runner_kwargs=init_runner_kwargs)
                end_time = time.time()
                accelerator.print(f"Time taken to rollout policy: {end_time - start_time} seconds")

                step_log.update(runner_log)
                self.num_rollout_completed += 1

            # ========= action MSE error for this epoch (non-distributed manner) ==========
            # compute action MSE error on both training and validation datasets (and for each task if multi-task training)
            def compute_action_mse(category, task, pred_action, gt_action):
                log = {}
                
                # Overall MSE
                log[f'{category}/action_mse_error/{task}'] = torch.nn.functional.mse_loss(pred_action, gt_action).item()

                action_components = cfg.task.shape_meta.action.components
                current_action_dim = 0
                for component in action_components:
                    component_dim = component.size
                    component_name = component.type
                    log[f'{category}/action_mse_error_{component_name}/{task}'] = torch.nn.functional.mse_loss(pred_action[..., current_action_dim:current_action_dim+component_dim], gt_action[..., current_action_dim:current_action_dim+component_dim]).item()
                    current_action_dim += component_dim
                
                return log
            
            if cfg.training.sample_every != -1 and ((self.epoch % cfg.training.sample_every == 0) or (self.epoch == cfg.training.num_epochs-1)) and accelerator.is_main_process:
                with torch.inference_mode():
                    def compute_action_mse_on_dataset(split_name: str, task_name_to_dataloader: dict):
                        """Handles logging per-task MSE if doing multi-task training. Also handles only computing loss on non-prompt actions if prompting policy."""
                        pbar = tqdm.tqdm(task_name_to_dataloader.items(), desc=f"Computing action MSE error on split \"{split_name}\"", leave=False, mininterval=cfg.training.tqdm_interval_sec)
                        for unique_task_name, dataloader in pbar:
                            pbar.set_description(f"Computing action MSE error for \"{unique_task_name}\" on split \"{split_name}\"")
                            num_task_observed = 0
                            pred_actions = []
                            gt_actions = []
                            loader_iter = iter(dataloader)
                            try:
                                for batch in loader_iter:
                                    batch = send_to_device(batch, device=policy.device)
                                    B = batch['action'].shape[0]
                                    num_task_observed += B

                                    pred_action = policy.predict_action_training(batch['obs'])['action_pred'].cpu() # (B, T [optional; only if sequence prompting], num_pred_steps, action_dim)
                                    gt_action = batch['action'].cpu() # (B, T [optional; only if sequence prompting], num_pred_steps, action_dim)

                                    if 'action_mask' in batch.get('metadata', {}):
                                        # action_mask is (B,) and contains the indices of the first valid action for each batch entry
                                        for batch_i in range(B):
                                            first_valid_action_index = batch['metadata']['action_mask'][batch_i]
                                            pred_actions.append(pred_action[batch_i, first_valid_action_index:])
                                            gt_actions.append(gt_action[batch_i, first_valid_action_index:])
                                    else:
                                        pred_actions.append(pred_action)
                                        gt_actions.append(gt_action)

                                    if num_task_observed >= cfg.training.sample_size:
                                        break
                            finally:
                                del loader_iter

                            assert num_task_observed >= cfg.training.sample_size, f'for task {unique_task_name}, num_task_observed: {num_task_observed}, cfg.training.sample_size: {cfg.training.sample_size} on split {split_name}'

                            pred_actions = torch.cat(pred_actions, dim=0)
                            gt_actions = torch.cat(gt_actions, dim=0)
                            
                            # now average over the entire batch
                            mse_log = compute_action_mse(split_name, unique_task_name.replace(' ', '_'), pred_actions, gt_actions)

                            step_log.update(mse_log)

                    if prompt_dataset is not None:
                        # if during training we are using a separate dataset for prompts, during evaluation we need to make sure that prompts are loaded in the normal dataset without using a separate dataset for prompting
                        dataset.set_ignore_prompt(False)
                    
                    compute_action_mse_on_dataset('train', train_action_mse_dataloader_by_task) 
                    compute_action_mse_on_dataset('val', val_action_mse_dataloader_by_task)
            
            # ========= checkpoint ==========
            if accelerator.is_main_process:
                # unwrap the model to save ckpt
                model_ddp = self.model
                self.model = policy # if EMA is used, then this is the unwrapped model with the EMA weights copied in, which is what we will save into the checkpoint

                # checkpointing
                if cfg.checkpoint.save_last_ckpt or is_last_epoch:
                    reason = 'final epoch' if is_last_epoch and not cfg.checkpoint.save_last_ckpt else 'save_last_ckpt is True'
                    path = self.save_checkpoint()
                    accelerator.print(f'Saved latest checkpoint ({reason}): {path}')
                if cfg.checkpoint.save_last_snapshot:
                    path = self.save_snapshot()
                    accelerator.print(f'Saved latest snapshot (save_last_snapshot is True): {path} ')

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                metric_dict['train_loss'] = step_log['train_loss_epoch_average'] # use the epoch average train loss for checkpointing rather than the noisy train loss

                # save `train_loss` checkpoint every N epochs (skipped entirely if not configured)
                save_train_loss_ckpt_every = cfg.checkpoint.get('save_train_loss_ckpt_every', None)
                if save_train_loss_ckpt_every is not None and \
                        ((self.epoch % save_train_loss_ckpt_every) == 0 or is_last_epoch):
                    train_loss_reason = 'last epoch' if is_last_epoch and (self.epoch % save_train_loss_ckpt_every) != 0 else f'epoch % {save_train_loss_ckpt_every} == 0'
                    topk_ckpt_path = topk_manager_train_loss.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        accelerator.print(f'Saved train_loss top-k checkpoint ({train_loss_reason}): {topk_ckpt_path}')

                if ((self.epoch % checkpoint_every) == 0 or is_last_epoch) and topk_manager is not None:
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_reason = 'last epoch' if is_last_epoch and (self.epoch % checkpoint_every) != 0 else f'epoch % {checkpoint_every} == 0'
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        accelerator.print(f'Saved {cfg.checkpoint.topk.monitor_key} top-k checkpoint ({topk_reason}): {topk_ckpt_path}')

                # recover the DDP model
                self.model = model_ddp
            
            if ema_model is not None:
                ema_model.restore(policy.parameters())

            # ========= eval end for this epoch ==========
            # end of epoch
            # log of last step is combined with validation and rollout
            accelerator.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1

        accelerator.print(f'Finished training. Run dir: {self.output_dir}')
        accelerator.end_training()
        accelerator.print(f'Finished training. Run dir: {self.output_dir}')
