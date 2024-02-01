# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE

# from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs import CPMDragonflyConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]



class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
        old_dtype = hidden.dtype
        variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
        result = hidden * weight
        return result

    def forward(self, hidden, residual=None):
        if residual is None:
            hidden = self.rms_layernorm(hidden=hidden, weight=self.weight, eps=self.variance_epsilon)
            return hidden
        else:
            residual = hidden + residual
            hidden = self.rms_layernorm(hidden=residual, weight=self.weight, eps=self.variance_epsilon)
            return hidden, residual

class CPMFFN(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
        config = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.w_out = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.w_out(x)
        return x


class CPMAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        config = None
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.attention_out = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        if config.qk_norm == True:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            base=self.rope_theta,
            max_position=self.max_position_embeddings,
            rotary_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            rope_scaling=rope_scaling)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        
        # def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
        #     old_dtype = hidden.dtype
        #     variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        #     hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
        #     result = hidden * weight
        #     return result


        if self.q_norm is not None:
            org_q_shape = q.shape
            q = q.reshape(org_q_shape[0], org_q_shape[1], org_q_shape[2]//self.head_dim, self.head_dim)
            q = self.q_norm(q)
            q = q.reshape(*org_q_shape)
            

        if self.k_norm is not None:
            org_k_shape = k.shape
            k = k.reshape(org_k_shape[0], org_k_shape[1], org_k_shape[2]//self.head_dim, self.head_dim)
            k = self.k_norm(k)
            k = k.reshape(*org_k_shape)
            
    
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.attention_out(attn_output)
        return output


class CPMAttentionBlock(nn.Module):
    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        # print(f"rope_scaling: {rope_scaling}")
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.layernorm_before_attention = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.self_attention = CPMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            config=config
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.layernorm_before_attention(hidden_states)
        else:
            hidden_states, residual = self.layernorm_before_attention(
                hidden_states, residual)


        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        if self.config.scale_depth > 0:
            hidden_states = self.config.scale_depth / math.sqrt(self.num_layers) * hidden_states
        
        return hidden_states, residual
        

class CPMFFNBlock(nn.Module):
    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None,):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.layernorm_before_ffn = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
        self.ffn = CPMFFN(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
            config=config
        )

    def forward(self, hidden_states, residual):
        hidden_states, residual = self.layernorm_before_ffn(hidden_states, residual)
        hidden_states = self.ffn(hidden_states)
        if self.config.scale_depth > 0:
            hidden_states = self.config.scale_depth / math.sqrt(self.num_layers) * hidden_states
        return hidden_states, residual


class CPMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: CPMDragonflyConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.self_att = CPMAttentionBlock(config=config,
            linear_method=linear_method
        )
        self.ffn = CPMFFNBlock(config=config,
            linear_method=linear_method
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        hidden_states, residual = self.self_att(positions, 
                      hidden_states,
                      kv_cache,
                      input_metadata,
                      cache_event,
                      residual)
        
        hidden_states, residual = self.ffn(hidden_states, residual)
        return hidden_states, residual


class CPMModel(nn.Module):

    def __init__(
        self,
        config: CPMDragonflyConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # self.embed_tokens = VocabParallelEmbedding(
        #     config.vocab_size,
        #     config.hidden_size,
        # )
        self.layers = nn.ModuleList([
            CPMDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.output_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        
        if self.config.scale:
            hidden_states = hidden_states * self.config.scale_emb
        residual = None
        for i in range(len(self.layers)):
            cache_event = None if cache_events is None else cache_events[i]
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
                residual,
            )

        hidden_states, _ = self.output_layernorm(hidden_states, residual)
        if self.config.scale:
            hidden_states /= self.config.scale_width
        return hidden_states


class CPMDragonflyForCausalLM(nn.Module):

    def __init__(
        self,
        config: CPMDragonflyConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.input_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.encoder = CPMModel(config, linear_method)
        if self.config.tie_lm_head == False:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.input_embedding(input_ids)
        hidden_states = self.encoder(hidden_states, positions, kv_caches,
                                   input_metadata, cache_events) 
        if self.config.tie_lm_head:
            lm_head_weight = self.input_embedding.weight   
        else:
            lm_head_weight = self.lm_head.weight 
        next_tokens = self.sampler(lm_head_weight, hidden_states,
                                   input_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "project_q", "q"),
            ("qkv_proj", "project_k", "k"),
            ("qkv_proj", "project_v", "v"),
            ("gate_up_proj", "w_in.w_0", 0),
            ("gate_up_proj", "w_in.w_1", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                param = params_dict[name]
                    
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
