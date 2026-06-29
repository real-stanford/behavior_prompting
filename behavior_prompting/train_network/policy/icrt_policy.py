import torch
from typing import Dict
from omegaconf import DictConfig

from behavior_prompting.common.pytorch_util import dict_apply

from icrt.util.model_constructor import vision_encoder_constructor
from icrt.util import misc

from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import ICRTPromptObsEncoder
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.model.prompt.icrt import ICRT
from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker, normalize_obs_with_optional_prompt
from behavior_prompting.train_network.model.common.normalizer import Normalizer

class ICRTPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            prompt_encoder: ICRTPromptObsEncoder,
            icrt_config: DictConfig,
            vision_encoder_config: DictConfig,
            name: str
        ):
        super().__init__()

        self.shape_meta = shape_meta
        self.prompt_encoder = prompt_encoder

        self.sequence_length = shape_meta['prompt_sequence_length']
        self.max_batch_size = icrt_config['max_batch_size']
        self.action_dim = shape_meta['action']['shape'][0]
        self.action_horizon = shape_meta['action']['horizon']
        self.chunk_n_actions = shape_meta['prompt_chunk_n_actions']

        assert self.sequence_length % self.chunk_n_actions == 0, f'sequence length {self.sequence_length} must be divisible by prompt chunk n actions {self.chunk_n_actions}'
        self.chunked_sequence_length = self.sequence_length // self.chunk_n_actions
        
        # construct vision encoder
        vision_encoder = vision_encoder_constructor(vision_encoder_cfg=vision_encoder_config)

        # construct ICRT model
        self.icrt_model = ICRT(
            vision_encoder=vision_encoder,
            num_cameras=prompt_encoder.num_cameras,
            proprio_dim=prompt_encoder.proprio_dim,
            action_dim=self.action_dim + 1, # we add 1 due to a part of the ICRT code that original assumed there was an EOS number appended to the action space which ended up not being included in the final model (the constructor for ICRT subtracts 1 to account for this)
            seq_length=self.chunked_sequence_length,
            num_pred_steps=self.action_horizon,
            chunk_n_actions=self.chunk_n_actions,
            **icrt_config
        )

        self.normalizer = Normalizer()
        self.has_inference_been_prompted = False
        self.just_prompted = False
        self.prompt_action_chunker = PromptActionChunker(shape_meta)

        assert self.shape_meta['prompt_sample_mode'] == 'sequence', 'ICRT only supports sequence prompting'

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # === Inference time ===
    def reset(self, action_exec_horizon=None):
        if action_exec_horizon is None:
            action_exec_horizon = self.action_horizon
        assert action_exec_horizon <= self.action_horizon, f'action exec horizon {action_exec_horizon} must be less than or equal to the action prediction horizon {self.action_horizon}'
        assert action_exec_horizon % self.chunk_n_actions == 0, 'action exec horizon must be divisible by prompt chunk n actions'
        self.has_inference_been_prompted = False
        # TODO: will need to update this to be the actual exec horizon not the horizon produced by the policy when we want to support the temporal averaging feature
        self.icrt_model.reset(action_exec_horizon=action_exec_horizon)
        self.action_exec_horizon = action_exec_horizon

        initial_mode = self.icrt_model.training
        self.icrt_model.train(True) # this clears out the KV cache in the underlying llama model
        self.icrt_model.train(initial_mode) # restore the previous mode

    def supports_prompting(self) -> bool:
        return True
    
    @torch.inference_mode()
    def prompt(self, prompt_dict: Dict[str, torch.Tensor]) -> int:
        """Right now for batch prompting we assume that all prompts are the same length. This could mean just using the same prompt across the entire batch or trimming the prompts to a fixed length. This is because the llama implementation in ICRT does not support specifying `position_ids` which would let us specify the true position subject to padding for inputs according to the padded prompt lengths.
        
        The proprioception and actions you pass here can either include the `num_pred_steps` chunking or not. The underlying code will remove it before actually prompting the model since the input to the model does not include `num_pred_steps`.
        """
        assert not self.has_inference_been_prompted, 'need to call `reset` before prompting again'

        prompt_dict = prompt_dict.copy()
        obs_dict = {
            'prompt': prompt_dict
        }
        
        prompt_dict = normalize_obs_with_optional_prompt(obs_dict, self.normalizer)['prompt']
        encoded_obs = self.prompt_encoder(prompt_dict)
        assert not torch.any(encoded_obs['prompt_mask']), 'sequence provided to prompt should be all marked as a prompt in the prompt mask'

        batch_size = encoded_obs["proprio"].shape[0]
        sequence_length = encoded_obs["proprio"].shape[1]
        assert sequence_length <= self.chunked_sequence_length, 'prompt cannot be longer than the sequence length of the model'
        assert batch_size <= self.max_batch_size, f'batch size {batch_size} exceeds max batch size {self.max_batch_size} specified in model initialization'

        self.icrt_model.prompt(encoded_obs)
        self.has_inference_been_prompted = True
        self.just_prompted = True

    def num_available_actions(self) -> int:
        num_tokens_per_inference = self.action_exec_horizon // self.chunk_n_actions
        num_tokens_remaining = self.chunked_sequence_length - self.icrt_model.start_pos // 2
        number_of_valid_times_to_run_inference = num_tokens_remaining // num_tokens_per_inference
        available_actions = number_of_valid_times_to_run_inference * self.action_exec_horizon
        if self.icrt_model.first_obs:
            available_actions += self.action_exec_horizon
        return available_actions
    
    @torch.inference_mode()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        this should be used during rollout of the policy using a observations with a history length equal to `self.exec_action_horizon` (but right after prompting you only need 1 observation not a history). This method handles the action chunking.
        obs_dict: directly contains keys of observations
        fixed_action_prefix: unnormalized action prefix
        result: diction with "action" and "action_pred" keys
        """
        assert self.has_inference_been_prompted
        assert 'past_action' not in obs_dict # not implemented yet
        assert self.num_available_actions() >= self.action_exec_horizon, 'trying to predict actions beyond the sequence length of the model which is not supported as it will be out of the training distribution'

        # if just prompted then we are going to receive an observation that has a history equal to `self.exec_action_horizon`, but we don't have associated actions for that history of observations (the observations are just duplicated across history steps at the very start). So solution is to just trim the observations down to a single step of history
        if self.just_prompted:
            self.just_prompted = False
            obs_dict = dict_apply(obs_dict, lambda x: x[:, -1:, ...]) # trim to just the last observation (single step of observation)
        else:
            prompt_dict = {
                'obs': obs_dict,
            }
            # all the entries in obs_dict should have shape (B, T, ...). The environment provides a set of observations corresponding to the previous actions executed.
            # we want to provide the model with the latest observations so we actually need to select the last observation in each chunk rather than the first observation in each chunk.
            # example: chunk 2 every actions, predict 4 actions and exec 4 actions. * indicates this observation that will be send to the policy, brackets indicate a chunk of observations returned by the environment after executing the previous action predictions
            # example: [o* (from just_prompted case above)], a1, [o, a2, o*, a3, o, a4, o*], a1, [o, a2, o*, a3, o, a4, o*], etc.
            # importantly, this means that from a given set of observations returned by the environment, we need to select the last observation in each chunk rather than the first observation in each chunk. So if we get [o, o, o, o], we need to select the last observation in each chunk, so we get [o, o*, o, o*]. However, the chunk_prompt selects the first observations in each chunk [o*, o, o*, o]. To handle this we temporarily flip the ordering of the observations and then flip it back after chunking to select the right entries.

            for key in prompt_dict['obs']:
                T = prompt_dict['obs'][key].shape[1]
                flipped_ordering = torch.arange(T-1, -1, -1) # torch doesn't support negative step, so we just manually create indexes in flipped order
                prompt_dict['obs'][key] = prompt_dict['obs'][key][:, flipped_ordering] # reverse the order of the observations

            chunked_prompt_dict = self.prompt_action_chunker.chunk_prompt(prompt_dict)

            for key in chunked_prompt_dict['obs']:
                new_T = chunked_prompt_dict['obs'][key].shape[1]
                new_flipped_ordering = torch.arange(new_T-1, -1, -1) # torch doesn't support negative step, so we just manually create indexes in flipped order
                chunked_prompt_dict['obs'][key] = chunked_prompt_dict['obs'][key][:, new_flipped_ordering] # reverse the order of the chunked observations to bring them back to the original order

            obs_dict = chunked_prompt_dict['obs']

        nobs = normalize_obs_with_optional_prompt(obs_dict, self.normalizer) # note there is no prompt in obs_dict, but this function will normalize the non-prompt obs
        B = next(iter(nobs.values())).shape[0]

        # encode obs
        encoded_obs = self.prompt_encoder({'obs': nobs})
        # predict action
        action_pred = self.icrt_model.get_action(encoded_obs)

        # unnormalize prediction
        assert action_pred.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(action_pred)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # === Training time ===
    def compute_loss(self, batch):
        """
        Batch should contain a 'prompt' key that contains the prompt data.
        Batch should contain an 'action' key that contains the action data that does not have pred steps horizon (though this is not actually needed for computing the loss).
        Instead we pull actions from batch['obs']['prompt']['action'] which is the version of the action data that has the pred steps horizon, which is the one used by the model to compute loss.
        """
        batch = batch.copy()
        batch['obs'] = normalize_obs_with_optional_prompt(batch['obs'], self.normalizer) # this also handles the normalization of the actions that are in the prompt
        encoded_batch = self.prompt_encoder(batch['obs']['prompt'])
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_dict = self.icrt_model(encoded_batch)
        return loss

    def predict_action_training(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        During training we need to predict actions for every step in the prompt and rollout at once. This differs from during evaluation time when we first prompt then step by step run inference using `prompt` and `predict_action`. Note that we hope that action output by the model matches the action that is input to the model. The reason the model can't just copy the actions from the input is that the model is autoregressive so it only will see the correct action after it has already predicted the action for the previous step.
        
        obs_dict should contain a 'prompt' key that contains the prompt data.
        """

        prompt = normalize_obs_with_optional_prompt(obs_dict, self.normalizer)['prompt']
        encoded_batch = self.prompt_encoder(prompt)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            action_pred = self.icrt_model.forward_action(encoded_batch)
        action_pred = self.normalizer['action'].unnormalize(action_pred) # (B, T, num_pred_steps, action_dim)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    def forward(self, batch):
        return self.compute_loss(batch)

    def get_optimizer(self, lr, betas, weight_decay):
        # following timm: set wd as 0 for bias and norm layers
        param_groups = misc.add_weight_decay(self, weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return optimizer
