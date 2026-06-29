from typing import Dict
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.train_network.model.common.base_obs_encoder import BaseTokenizedObsEncoder
from behavior_prompting.train_network.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.utils.prompt_util import normalize_obs_with_optional_prompt

class DiffusionTransformerPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: BaseTokenizedObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            diffusion_decoder_n_layer=7,
            diffusion_decoder_n_head=8,
            diffusion_encoder_enabled=False,
            diffusion_encoder_n_layer=4,
            diffusion_encoder_n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            name: str = 'diffusion_transformer',
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']

        max_token_count = obs_encoder.get_max_token_count()
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            decoder_n_layer=diffusion_decoder_n_layer,
            decoder_n_head=diffusion_decoder_n_head,
            encoder_n_layer=diffusion_encoder_n_layer,
            encoder_n_head=diffusion_encoder_n_head,
            enable_encoder=diffusion_encoder_enabled,
            n_emb=n_emb,
            max_cond_tokens=max_token_count,
            p_drop_attn=p_drop_attn
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = Normalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.init_weights()
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None, memory_key_padding_mask=None,
            # keyword arguments to scheduler.step
            need_weights=False,
            average_attn_weights=False,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        cross_attn_weights = []

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            ret = model(trajectory, t, cond, memory_key_padding_mask=memory_key_padding_mask, need_weights=need_weights, average_attn_weights=average_attn_weights)

            if need_weights:
                model_output, cur_cross_attn_weights = ret
                cross_attn_weights.append(cur_cross_attn_weights)
            else:
                model_output = ret

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        if need_weights:
            return trajectory, torch.stack(cross_attn_weights, dim=0)
        else:
            return trajectory
        
    def supports_prompting(self):
        return self.obs_encoder.supports_prompting()
    
    @torch.inference_mode()
    def prompt(self, prompt_dict: Dict):
        obs_dict = {'prompt': prompt_dict}
        obs_dict = normalize_obs_with_optional_prompt(obs_dict, self.normalizer)
        prompt_dict = obs_dict['prompt']
        self.obs_encoder.prompt(prompt_dict)

    def reset(self, action_exec_horizon=None):
        self.obs_encoder.reset()

        assert action_exec_horizon is None or action_exec_horizon <= self.action_horizon, 'action_exec_horizon must be None or less than or equal to the action_horizon'

    @torch.inference_mode()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], need_weights: bool=False, average_attn_weights: bool=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        obs_dict = obs_dict.copy()
        assert 'past_action' not in obs_dict # not implemented yet

        # normalize input
        obs_dict = normalize_obs_with_optional_prompt(obs_dict, self.normalizer)

        cond, obs_encoder_metadata = self.obs_encoder(obs_dict, need_weights=need_weights, average_attn_weights=average_attn_weights)
        memory_key_padding_mask = obs_encoder_metadata.pop('token_mask')

        del obs_dict
        B = cond.shape[0]
        
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # run sampling
        ret = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=cond,
            memory_key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            **self.kwargs)
        
        if need_weights:
            nsample, diffusion_attention_weights = ret
        else:
            nsample = ret
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'obs_encoder_metadata': obs_encoder_metadata
        }

        if need_weights:
            result['diffusion_attention_weights'] = diffusion_attention_weights

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            **kwargs
        ) -> torch.optim.Optimizer:
        optim_groups = []
        optim_groups.extend(self.model.get_optim_groups(weight_decay=weight_decay))
        optim_groups.extend(self.obs_encoder.get_optim_groups(lr=lr,  weight_decay=weight_decay))        

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay, **kwargs)
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        
        obs_dict = batch['obs']
        obs_dict = normalize_obs_with_optional_prompt(obs_dict, self.normalizer)
        cond, obs_encoder_metadata = self.obs_encoder(obs_dict)
        memory_key_padding_mask = obs_encoder_metadata.pop('token_mask')

        del obs_dict
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            cond=cond,
            memory_key_padding_mask=memory_key_padding_mask
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
        
    def get_diffusion_cross_attn_dim_names(self, obs_len: int):
        obs_encoder_token_names = self.obs_encoder.get_output_token_names(obs_len)
        return ['diffusion timestep'] + obs_encoder_token_names

    def init_weights(self):
        # no need to init weights here because they are initialized in the constructor of the obs encoder and model
        pass
