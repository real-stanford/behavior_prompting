import os
import hydra
import torch
import time
import random
import copy
import numpy as np
from accelerate import Accelerator
from omegaconf import OmegaConf
from behavior_prompting.train_network.workspace.base_workspace import BaseWorkspace
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.utils.load_env import load_env_runner, env_rollout

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RolloutPolicyWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if not cfg.rollout.replay_dataset_action:
            # configure model
            self.model: BasePolicy = hydra.utils.instantiate(cfg.model)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes

        accelerator = Accelerator(log_with='wandb', mixed_precision=cfg.training.mixed_precision)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        accelerator.print(f'Started rollout. Run dir: {self.output_dir}')

        if not accelerator.is_main_process and not cfg.rollout.distribute:
            raise ValueError('should not run multi GPU accelerate if distribute eval is disabled')
        
        # TODO: the design of this checkpoint loading actually has some pretty big limitations. For example we don't actually load the config from the model checkpoint file. Instead we expect that the user sets up the config in their run of this workspace exactly how they used to train the model. In reality we should be loading the config for this run from the checkpoint file rather than having the user set it up correctly.
        
        # load checkpoint
        if cfg.rollout.checkpoint_path:
            print(f"Going to evaluate checkpoint {cfg.rollout.checkpoint_path}")
            saved_output_dir = self.output_dir
            self.load_checkpoint(path=cfg.rollout.checkpoint_path, exclude_keys=['optimizer'])
            self._output_dir = saved_output_dir
        elif cfg.rollout.policy_only_checkpoint_path:
            print(f"Going to evaluate policy only checkpoint {cfg.rollout.policy_only_checkpoint_path}")
            checkpoint = torch.load(cfg.rollout.policy_only_checkpoint_path, map_location='cpu')
            
            self.model.load_state_dict(checkpoint)
        else:
            assert cfg.rollout.replay_dataset_action, 'must provide a checkpoint path or replay dataset actions'

        # setup policy if not doing replay dataset action
        if not cfg.rollout.replay_dataset_action:
            policy = self.model
            policy.eval()
            policy.to(accelerator.device)
        else:
            policy = None
            print('Going to replay dataset actions')

        # load env runners
        accelerator.print('Starting to load env runners')
        start_time = time.time()

        env_runners = load_env_runner(cfg, self.output_dir, accelerator=accelerator if cfg.rollout.distribute else None)
        end_time = time.time()
        accelerator.print(f"Time taken to load env runners: {end_time - start_time} seconds")

        # rollout policy in env
        accelerator.print('Starting to rollout policy')
        start_time = time.time()
        step_log = env_rollout(cfg, env_runners, policy, enable_expensive_vis=cfg.rollout.enable_expensive_vis, accelerator=accelerator if cfg.rollout.distribute else None)
        end_time = time.time()
        accelerator.print(f"Time taken to rollout policy: {end_time - start_time} seconds")
        
        step_log['epoch'] = 0 # helps with logging on wandb
        accelerator.log(step_log, step=0)

        accelerator.print(f'Finished rollout. Run dir: {self.output_dir}')
        accelerator.end_training()
        accelerator.print(f'Finished rollout. Run dir: {self.output_dir}')
