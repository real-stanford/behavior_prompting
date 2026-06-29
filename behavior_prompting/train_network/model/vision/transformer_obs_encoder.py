"""
This is a modified version of the transformer_obs_encoder in UMI diffusion_policy.
It adds the following features:
- use vision normalization
- use separate train_image_transforms and eval_image_transforms rather than single augmentation
"""

import copy
from typing import Dict, List, Optional, Tuple

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from behavior_prompting.train_network.utils.model_util import replace_submodules
from behavior_prompting.train_network.common.augmentation import ImageAugmentation
from behavior_prompting.train_network.model.common.base_obs_encoder import BaseTokenizedObsEncoder
from behavior_prompting.train_network.utils.model_util import init_weights

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
    

class TransformerObsEncoder(BaseTokenizedObsEncoder):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            train_image_transforms: Optional[ImageAugmentation]=None,
            non_prompt_train_image_transforms: Optional[ImageAugmentation]=None,
            eval_image_transforms: Optional[ImageAugmentation]=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            use_vision_norm: bool=True,
            concat_time_dimension: bool=True,
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
        key_non_prompt_train_image_transform_map = nn.ModuleDict()
        key_eval_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        self.model_name = model_name

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
            
        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation is None:
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        #TODO: add support for only initializing for keys that are used by the policy
        # right now we intialize for all the keys in shape_meta, whether they are ever used or not
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
            if obs_shape_meta[key].get('ignore_by_policy', False):
                continue
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                if 'share_rgb_model' in attr:
                    to_share_key = attr['share_rgb_model']
                    this_model = key_model_map[to_share_key]
                else:
                    this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_train_transform_map[key] = train_image_transforms.get_transform(key) if train_image_transforms is not None else nn.Identity()
                key_eval_transform_map[key] = eval_image_transforms.get_transform(key) if eval_image_transforms is not None else nn.Identity()
                key_non_prompt_train_image_transform_map[key] = non_prompt_train_image_transforms.get_transform(key) if non_prompt_train_image_transforms is not None else nn.Identity()
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
            elif type == 'low_dim':
                if key == 'task_language':
                    # projection from CLIP output dim -> n_emb will be set after loading CLIP
                    low_dim_keys.append(key)
                else:
                    dim = np.prod(shape)
                    proj = nn.Identity()
                    if dim != n_emb:
                        proj = nn.Linear(in_features=dim, out_features=n_emb)
                    key_projection_map[key] = proj
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]

        # load CLIP language encoder and finish projection setup
        self.using_language = 'task_language' in shape_meta['obs'] and not shape_meta['obs']['task_language'].get('ignore_by_policy', False)
        if self.using_language:
            self.clip_language_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_model_name)
            tokenizer = CLIPTokenizer.from_pretrained(text_encoder_model_name)
            self.clip_pad_token_id = tokenizer.pad_token_id
            clip_projection_dim = self.clip_language_encoder.config.projection_dim
            proj = nn.Identity() if clip_projection_dim == n_emb else nn.Linear(clip_projection_dim, n_emb)
            key_projection_map['task_language'] = proj
        else:
            self.clip_language_encoder = None
            self.clip_pad_token_id = None

        # sort is very important to ensure we put the output tokens in a consistent order
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_train_transform_map = key_train_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.key_non_prompt_train_image_transform_map = key_non_prompt_train_image_transform_map
        self.model_normalization_transform = model_normalization_transform
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.ordered_rgb_keys = rgb_keys
        self.ordered_low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_vision_norm = use_vision_norm
        self.concat_time_dimension = concat_time_dimension
        self.pretrained = pretrained
        self.pretrained_lr_scale = pretrained_lr_scale

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        self.init_weights()

    def aggregate_feature(self, feature):
        # Return: B, N, C
        
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            
            # or use all tokens
            assert self.feature_aggregation is None 
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict, rgb_keys=None, low_dim_keys=None, is_prompt: bool=False, *args, **kwargs):
        rgb_keys, low_dim_keys = self.get_present_ordered_keys(rgb_keys, low_dim_keys)

        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])

            if self.training:
                if not is_prompt:
                    img = self.key_non_prompt_train_image_transform_map[key](img)
                img = self.key_train_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            if self.use_vision_norm:
                img = self.model_normalization_transform(img) # apply image normalization

            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B,-1,self.n_emb)
            embeddings.append(emb)

        # process lowdim input
        for key in low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            if key == 'task_language' and self.using_language:
                assert T == 1, 'language horizon must be 1'
                input_ids = data.squeeze(1)  # (B, seq_len)
                attention_mask = (input_ids != self.clip_pad_token_id).long()
                text_outputs = self.clip_language_encoder(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )
                emb = self.key_projection_map[key](text_outputs.text_embeds)  # (B, n_emb)
                emb = emb.unsqueeze(1)  # (B, 1, n_emb)
            else:
                assert data.shape[2:] == self.key_shape_map[key]
                data = data.reshape(B, T, -1)
                emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # concatenate all features along t
        if self.concat_time_dimension:
            result = torch.cat(embeddings, dim=1) # (B, T, n_emb)
        else:
            result = torch.stack(embeddings, dim=2) # (B, T, num_obs_types, n_emb)
        return result, {'token_mask': None}

    @torch.no_grad()
    def output_shape(self, rgb_keys=None, low_dim_keys=None):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            dtype = torch.int64 if key == 'task_language' and self.using_language else self.dtype
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _ = self.forward(example_obs_dict, rgb_keys, low_dim_keys)
        if self.concat_time_dimension:
            assert len(example_output.shape) == 3
        else:
            assert len(example_output.shape) == 4
        assert example_output.shape[0] == 1
        assert example_output.shape[-1] == self.n_emb

        return example_output.shape
    
    def get_obs_names(self, rgb_keys=None, low_dim_keys=None) -> List[str]:
        """provides the names of the observation modalities when we are not concatenating them across the time dimension. Provides the keys in order of the tokens that will be output by the forward function."""
        present_ordered_rgb_keys, present_ordered_low_dim_keys = self.get_present_ordered_keys(rgb_keys, low_dim_keys)

        assert not self.concat_time_dimension

        present_ordered_obs_keys = present_ordered_rgb_keys + present_ordered_low_dim_keys
        return present_ordered_obs_keys
    
    def get_present_ordered_keys(self, rgb_keys: Optional[List[str]], low_dim_keys: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        """Selects the keys from the obs_dict that are in rgb_keys or low_dim_keys. Ensures that the ordering of the keys matches self.rgb_keys and self.low_dim_keys regardless of the order of rgb_keys and low_dim_keys which is important for ensuring the inputs are handled in a consistent manner."""
        if rgb_keys is None:
            result_rgb_keys = self.ordered_rgb_keys
        else:
            result_rgb_keys = [x for x in self.ordered_rgb_keys if x in rgb_keys]
            assert len(result_rgb_keys) == len(rgb_keys), f'{result_rgb_keys} != {rgb_keys}'
        
        if low_dim_keys is None:
            result_low_dim_keys = self.ordered_low_dim_keys
        else:
            result_low_dim_keys = [x for x in self.ordered_low_dim_keys if x in low_dim_keys]
            assert len(result_low_dim_keys) == len(low_dim_keys)
    
        return result_rgb_keys, result_low_dim_keys
    
    def init_weights(self):
        if not self.pretrained:
            # only do weight initialization if we are not using a pretrained model because we don't want to overwrite the pretrained weights
            self.apply(init_weights)
        else:
            # note in this case we don't apply the weight initialization to the key_projection_map (not necessarily ideal, but likely doesn't matter)
            pass

    """BaseObsEncoder methods"""
    def reset(self):
        pass

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        optim_groups = []

        backbone_lr = lr * self.pretrained_lr_scale if self.pretrained else lr
        clip_lr = lr * self.pretrained_lr_scale if self.using_language else None

        backbone_params = list()
        clip_params = list()
        other_obs_params = list()
        for key, value in self.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            elif self.using_language and key.startswith('clip_language_encoder'):
                clip_params.append(value)
            else:
                assert key.startswith('key_projection_map') or key == '_dummy_variable', f'unexpected key: {key}'
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": 0,
            "lr": backbone_lr
        })
        if self.using_language and len(clip_params) > 0:
            optim_groups.append({
                "params": clip_params,
                "weight_decay": 0,
                "lr": clip_lr
            })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": weight_decay,
            "lr": lr
        })

        return optim_groups
    
    """BaseTokenizedObsEncoder methods"""
    def get_max_token_count(self):
        assert self.concat_time_dimension
        output_shape = self.output_shape()
        return output_shape[1]
    
    def get_output_token_names(self, obs_len: int) -> List[str]:
        raise NotImplementedError
