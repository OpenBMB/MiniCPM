# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from typing_extensions import TypedDict
from transformers.configuration_utils import PretrainedConfig


class CPMMistralConfig(PretrainedConfig):
    model_type = "cpmmistral"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "num_kv_heads",
        "hidden_act": "activate_fn",
        "hidden_size": "dim_model",
        "num_attention_heads": "num_heads",
        "intermediate_size": "dim_ff",
        "num_hidden_layers": "num_layers",
        "vocab_size": "vocab_size",
        "rms_norm_eps": "eps",
        "scale_emb": "scale_emb",
        "scale_depth": "scale_depth",
        "scale": "scale",
        "attention_scale": "attention_scale"
    }

    def __init__(
        self,
        vocab_size=32000,
        dim_model=4096,
        num_heads=32,
        num_kv_heads=32,
        dim_head=128,
        dim_ff=11008,
        num_layers=32,
        dropout_p=0.0,
        activate_fn="silu",
        scale=True,
        scale_emb: float=1.,
        scale_depth: float=-1,
        dim_model_base:int=None,
        eps=1e-5,
        init_std=0.02,
        half: bool = True,
        half_type = 'bf16',
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        use_flash_attn: bool = True,
        flash_attn_mask_shape="1d",
        flash_impl="cuda",
        base=10000,
        non_checkpointing_layers_num:int = 0,
        attention_scale=1,
        max_position_embeddings=8192,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.activate_fn = activate_fn
        self.scale = scale
        self.scale_emb = scale_emb
        self.half = half
        self.half_type = half_type
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.eps = eps
        self.init_std = init_std
        self.flash_impl = flash_impl
        self.mask_modules = mask_modules
        self.use_flash_attn = use_flash_attn
        self.flash_attn_mask_shape = flash_attn_mask_shape
        self.base = base
        self.attention_scale=attention_scale
        self.max_position_embeddings = max_position_embeddings
        self.non_checkpointing_layers_num = non_checkpointing_layers_num
        self.rope_scaling = rope_scaling
        super().__init__(architectures=["CPMMistralForCausalLM"])
    
    @property
    def scale_width(self,):
        if self.scale:
            return self.dim_model / self.dim_model_base
        else:
            return 1.
    
    @property
    def dtype(self, ):
        if self.half:
            if self.half_type == 'bf16':
                return torch.bfloat16
            else:
                return torch.half
        else:
            return torch.float
    