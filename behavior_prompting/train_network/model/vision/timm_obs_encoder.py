"""
This is a modified version of the timm_obs_encoder in UMI diffusion_policy.
It adds the following features:
- use vision normalization
- use separate train_image_transforms and eval_image_transforms rather than single augmentation
"""

import copy
from typing import Dict, List, Optional

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from behavior_prompting.train_network.utils.model_util import replace_submodules
from behavior_prompting.train_network.common.augmentation import ImageAugmentation
from behavior_prompting.train_network.model.common.base_obs_encoder import BaseObsEncoder
from behavior_prompting.train_network.utils.model_util import init_weights
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(BaseObsEncoder):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            train_image_transforms: Optional[ImageAugmentation],
            eval_image_transforms: Optional[ImageAugmentation],
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            use_vision_norm: bool=True,
            flatten_time_dimension: bool=True,
            max_image_model_chunk_n: Optional[int]=None, # if not None, then the image will be split into batches of this size before passing through the vision encoder. This only applies during training and performs gradient checkpointing to save memory. This is useful to reduce memory usage when using using a large batch of images with a large vision encoder during training at the cost of compute.
            pretrained_lr_scale: float=0.1,
            text_encoder_model_name: Optional[str]=None,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_train_transform_map = nn.ModuleDict()
        key_eval_transform_map = nn.ModuleDict()

        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )

        model_data_config = timm.data.resolve_data_config(model.pretrained_cfg)
        model_normalization_transform = torchvision.transforms.Normalize(
            mean=model_data_config['mean'],
            std=model_data_config['std']
        )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # we want to build the vision models for the keys that share vision encoders after we build the models for the keys that don't share vision encoders
        obs_keys = list(obs_shape_meta.keys())
        ordered_obs_keys = list()
        for key in obs_keys:
            attr = obs_shape_meta[key]
            if 'share_rgb_model' in attr:
                ordered_obs_keys.append(key)
            else:
                ordered_obs_keys.insert(0, key)

        for key in ordered_obs_keys:
            attr = obs_shape_meta[key]
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')

            if attr.get('ignore_by_policy', False):
                continue

            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                if 'share_rgb_model' in attr:
                    to_share_key = attr['share_rgb_model']
                    this_model = key_model_map[to_share_key]
                else: # fall back to using global value for share_rgb_model if not specified at key level
                    this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
                key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        # Store repeat_n_times for each key
        key_repeat_n_times_map = dict()
        for key in obs_keys:
            attr = obs_shape_meta[key]
            if 'repeat_n_times' in attr:
                key_repeat_n_times_map[key] = attr['repeat_n_times']
            else:
                key_repeat_n_times_map[key] = None

        self.model_name = model_name
        self.pretrained = pretrained
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_train_transform_map = key_train_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.model_normalization_transform = model_normalization_transform
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.key_repeat_n_times_map = key_repeat_n_times_map
        self.feature_aggregation = feature_aggregation
        self.use_vision_norm = use_vision_norm
        self.flatten_time_dimension = flatten_time_dimension
        self.max_image_model_chunk_n = max_image_model_chunk_n
        self.pretrained_lr_scale = pretrained_lr_scale

        self.using_language = 'task_language' in shape_meta['obs'] and not shape_meta['obs']['task_language'].get('ignore_by_policy', False)
        self.text_encoder_model_name = text_encoder_model_name

        if self.using_language:
            self.clip_language_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_model_name)
            # Get pad_token_id from tokenizer (needed for attention mask creation)
            tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
            self.clip_pad_token_id = tokenizer.pad_token_id
        else:
            self.clip_language_encoder = None
            self.clip_pad_token_id = None

        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self.init_weights()

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict, *args, metadata=None, **kwargs):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])

            if self.training:
                img = self.key_train_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            if self.use_vision_norm:
                img = self.model_normalization_transform(img) # apply image normalization
            
            if self.max_image_model_chunk_n is not None and self.training:
                # Process images in chunks and apply activation checkpointing to avoid OOM
                chunks = []
                for i in range(0, img.shape[0], self.max_image_model_chunk_n):
                    chunk = img[i:i+self.max_image_model_chunk_n]
                    # use checkpoint to save memory
                    chunk = torch.utils.checkpoint.checkpoint(
                        self.key_model_map[key],
                        chunk,
                        use_reentrant=False
                    )
                    chunks.append(chunk)
                raw_feature = torch.cat(chunks, dim=0)
            else:
                raw_feature = self.key_model_map[key](img)
            
            feature = self.aggregate_feature(raw_feature) # (B*T, D)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            if self.flatten_time_dimension:
                features.append(feature.reshape(B, -1))
            else:
                features.append(feature.reshape(B, T, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            
            # Check if this is task_language and we're using finetuned encoder
            if key == 'task_language' and self.using_language:
                # task_language should contain token IDs when using finetuned encoder
                # Shape should be (B, horizon, seq_len) or (B, horizon*seq_len)
                B = batch_size
                expected_horizon = 1
                assert self.shape_meta['obs']['task_language']['horizon'] == 1, 'horizon should be 1 for language'
                
                # Reshape token IDs if needed
                if len(data.shape) == 3:  # (B, horizon, seq_len)
                    input_ids = data  # (B, horizon, seq_len)
                    assert input_ids.shape[1] == 1, 'horizon should be 1 for language'
                else:
                    raise ValueError(f"Unexpected shape for task_language token IDs: {data.shape}")
                
                # Create attention mask from token IDs using pad_token_id from config
                attention_mask = (input_ids != self.clip_pad_token_id).long()  # (B, horizon, seq_len)
                
                # Flatten to (B*horizon, seq_len) for batch processing
                B_horizon = B * expected_horizon
                input_ids_flat = input_ids.reshape(B_horizon, -1)  # (B*horizon, seq_len)
                attention_mask_flat = attention_mask.reshape(B_horizon, -1)  # (B*horizon, seq_len)
                
                # Encode with CLIP
                inputs = {
                    'input_ids': input_ids_flat.to(self.device),
                    'attention_mask': attention_mask_flat.to(self.device)
                }
                
                # Get embeddings
                text_outputs = self.clip_language_encoder(**inputs)
                text_embeds = text_outputs.text_embeds
                
                # Reshape back to (B, horizon, D) then to final shape
                text_embeds = text_embeds.reshape(B, expected_horizon, -1)  # (B, horizon, D)
                
                if self.flatten_time_dimension:
                    # Reshape to (B, horizon*D)
                    feature = text_embeds.reshape(B, -1)
                else:
                    # Keep as (B, horizon, D)
                    feature = text_embeds
            else:
                # Regular lowdim processing (backward compatibility)
                B, T = data.shape[:2]
                assert B == batch_size
                assert data.shape[2:] == self.key_shape_map[key]
                if self.flatten_time_dimension:
                    feature = data.reshape(B, -1)
                else:
                    feature = data.reshape(B, T, -1)
            
            # Apply repeat_n_times if specified
            repeat_n_times = self.key_repeat_n_times_map.get(key)
            if repeat_n_times is not None:
                # Repeat along the last dimension (feature dimension)
                feature = feature.repeat_interleave(repeat_n_times, dim=-1)
            
            features.append(feature)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result, {}
    
    @torch.inference_mode()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            # Use int64 for task_language when using finetuned encoder (token IDs)
            if key == 'task_language' and self.using_language:
                dtype = torch.int64
            else:
                dtype = self.dtype
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _ = self.forward(example_obs_dict)
        if self.flatten_time_dimension:
            assert len(example_output.shape) == 2
        else:
            assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1
        
        return example_output.shape
    
    def init_weights(self):
        if not self.pretrained:
            # only do weight initialization if we are not using a pretrained model because we don't want to overwrite the pretrained weights
            self.apply(init_weights)
        else:
            # there are no parameters besides the vision encoder parameters to initialize
            pass
    
    """BaseObsEncoder methods"""
    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        optim_groups = []

        backbone_lr = lr * self.pretrained_lr_scale if self.pretrained else lr # for fine tuning if pretrained
        clip_lr = lr * self.pretrained_lr_scale if self.using_language else None

        # obs encoder parameters
        backbone_params = list()
        clip_params = list()
        other_obs_params = list()
        for key, value in self.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            elif self.using_language and key.startswith('clip_language_encoder'):
                clip_params.append(value)
            else:
                if key.endswith('_dummy_variable'):
                    continue
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": 0, # we disable weight decay for now since we don't have an implementation to separate out the parameters that should and should not experience weight decay
            "lr": backbone_lr
        })

        # Add CLIP encoder parameters with pretrained LR scale
        if self.using_language and len(clip_params) > 0:
            optim_groups.append({
                "params": clip_params,
                "weight_decay": 0,
                "lr": clip_lr
            })

        # TODO: this assertion isn't really correct depending on the aggregation method, so this can be fixed later when it's an issue
        assert len(other_obs_params) == 0, 'the only parameters are from the vision models in the backbone'
        
        return optim_groups
    
if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        eval_image_transforms=None
    )
