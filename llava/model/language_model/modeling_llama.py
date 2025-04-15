# coding=utf-8
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
"""PyTorch LLaMA model."""

""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_llama import LlamaConfig





# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    Expands 2D mask [batch, seq_len] to 4D [batch, 1, tgt_len, src_len].
    Which means:
    - 2D mask: [batch, seq_len] , where seq_len is the length of the sequence and batch is the number of sequences.
    - 4D mask: [batch, 1, tgt_len, src_len] , where tgt_len is the length of the target sequence and src_len is the length of the source sequence.

    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)




class zero_init(nn.Module):
    # Simple module that initializes weights to zero
    # Used for stabilizing training of fusion modules
    def __init__(self,idx=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        return torch.tanh(self.weight) * hidden_states


class LlamaRMSNorm(nn.Module):
    '''
    Root Mean Square normalization
    Simpler alternative to LayerNorm
    Formula: output = weight * input / sqrt(mean(input^2) + eps)
        - eps: small constant to avoid division by zero
        - weight: learnable scaling factor
        - input: input tensor to normalize
        - hidden_states: input tensor to normalize
    '''

    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

# Position Embeddings
class LlamaRotaryEmbedding(nn.Module):
    '''
    Rotary Position Embeddings (RoPE)
    Purpose: Encode positional information in a way that's efficient and effective
    Key features:
    - Uses sine/cosine functions (sin/cos)
    - Caches computed values for efficiency (register_buffer)
    - Supports different scaling strategies (LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding)
    '''

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    '''
    MLP used in the transformer layers
    Contains: 
    - gate (gate_proj): Linear layer for gating
    - up (up_proj): Linear layer for upscaling
    - down (down_proj): Linear layer for downscaling
    - act_fn: Activation function from config
    '''
    def __init__(self, config):
        # Step 1: Basic Setup
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size        # Size of input/output
        self.intermediate_size = config.intermediate_size  # Size of middle layer
        
        # Step 2: Create Three Linear Layers
        # These are like simple transformations of the data
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # Gate layer -> Produces a gating signal that controls what information to pass through. It's like a filter that decides which parts of the input are important
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # Up layer -> Scales up the input to the intermediate size, but doesn't apply the activation function like the gate layer does
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # Down layer -> Scales down the intermediate size to the hidden size
        
        # Step 3: Get the activation function (like ReLU, but configurable)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # Path: Check if we need to split the computation across multiple GPUs
        if self.config.pretraining_tp > 1:
            # Step 1: Split the weights into smaller pieces
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # Step 2: Process each piece separately and combine results
            # Step 2.1: Apply the gate and up projections, these are the same operations but with different weights ofc
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            # Step 2.2: 
            # Now, The activation function (self.act_fn) is applied to the output of the gate_proj. (This produces values between 0 and 1 (depending on the activation function, like GELU or ReLU).)
            # Then Element-Wise Multiplication on act_fn(gate_proj) with up_proj --> This is to act like a filter, where the activation function acts as a gate.
            # This is the gating  mechanism — parts of the input can be amplified or suppressed based on the activation
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            
            # Step 2.3: Apply the down to compressed back down to the original size. 
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # If not using multiple GPUs, do simple forward pass:
            # 1. Pass through gate layer and activate
            # 2. Pass through up layer
            # 3. Multiply results
            # 4. Pass through down layer
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)






class LlamaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    '''
    Handles cross-attention between text and image
    Key components:
    - Query (q_proj): Query projection from text
    - Key (k_proj): Key projection from image
    - Value (v_proj): Value projection from image
    - Attention computation with masking (cross_attn_mask)
    - Output projection (o_proj)
    - Rotary position embedding (rotary_emb)
    '''

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # Step 1: Basic Setup
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        # Step 2: Configure Attention Parameters

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Step 3: Prep data for attention
        # Text input (hidden_states) is converted to queries using q_proj.
        # Image input (image_feature) is converted to keys and values using k_proj and v_proj.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)


    def forward(
        self,
        hidden_states: torch.Tensor,
        image_feature: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # Step 1: Get Dimensions
        bsz, q_len, _ = hidden_states.size() # Batch size, sequence length
        _, img_len,_ = image_feature.size() # Image feature length
        padding_length = hidden_states.shape[1] - image_feature.shape[1]

        # Step 2: Prep Attention Mask
        if attention_mask is not None:

            cross_attn_mask = attention_mask[:, :, :, :img_len]
        else:
            cross_attn_mask = None

        image_feature = image_feature

        # Step 3: Project Text to Queries
        query_states = self.q_proj(hidden_states)

        # Step 4: Project Image to Keys and Values
        if past_key_value is None:
            key_states = self.k_proj(image_feature)
            value_states = self.v_proj(image_feature)
            # Step 4.1: Transpose for .... rope?
            key_states = self._transpose_for_scores(key_states)
            value_states = self._transpose_for_scores(value_states)
        else:
            # Or Step 4.2: If we have past key values, we can use them to save computation
            key_states, value_states = past_key_value

        # Step 5: Reshape for Multi-Head Attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        past_key_value = (key_states, value_states) if use_cache else None
        
        
        kv_seq_len = key_states.shape[-2]
       
        # Step 6: Compute Attention Scores
        # This is like calculating how much each text token should pay attention to each image feature
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Step 7: Apply Attention Mask
        if cross_attn_mask is not None:
            if cross_attn_mask.size() != (bsz, 1, q_len, kv_seq_len):

                if attention_mask.size(-1) < kv_seq_len:
                    padding_size = kv_seq_len - attention_mask.size(-1)
                    attention_mask = torch.cat([attention_mask, attention_mask[:, :, :, -1:].expand(-1, -1, -1, padding_size)], dim=-1)
                    cross_attn_mask = attention_mask[:, :, :, :kv_seq_len]
                else:
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
            attn_weights = attn_weights + cross_attn_mask

        # Step 8: Normalize Attention Scores (Convert to Probabilities) and Apply Dropout
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Step 9: Generate Attention Output
        # attention scores are used to combine the values (image features)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Step 10: Project Attention Output
        # bring it back to the original size
        if self.config.pretraining_tp > 1:
            # Step 10.1: If we have multiple GPUs, we need to split the attention output
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            # Step 10.2: Split the output projection weights
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            # Step 10.3: Project the attention output
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            # Step 10.1: Project Attention Output
            # This is to project the attention output back to the original size
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    - Query (q_proj): Linear layer for query. Query projection from text
    - Key (k_proj): Linear layer for key. Key projection from image
    - Value (v_proj): Linear layer for value. Value projection from image
    - Output (o_proj): Linear layer for output. Output projection
    - Rotary position embedding (rotary_emb): Rotary position embedding
    """
    '''
    Handles self-attention within the text sequence. 
    Data flow: 
    - Text features are projected to queries, keys, and values
    - Rotary position embeddings are applied to queries and keys
    - Attention scores are calculated between all pairs of tokens
    - These scores determine how much each token should focus on other tokens
    - The weighted sum of values produces the final output
    '''

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # Step 1: Basic Setup
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        # Step 2: Check if this layer should use cross-attention, True if it is the last layer of the first third of the layers, False otherwise
        self.is_cross = self.layer_idx == config.num_hidden_layers//3-1 or self.layer_idx == config.num_hidden_layers//3*2 -1

        # Step 3: Configure Attention Parameters
        self.attention_dropout = config.attention_dropout  # How much to randomly ignore connections during training
        self.hidden_size = config.hidden_size  # Size of the hidden representation
        self.num_heads = config.num_attention_heads  # Number of parallel attention heads
        self.head_dim = self.hidden_size // self.num_heads  # Size of each attention head
        self.num_key_value_heads = config.num_key_value_heads  # Number of key/value heads (for efficiency)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # How many heads share the same key/value
        self.max_position_embeddings = config.max_position_embeddings  # Maximum sequence length
        self.rope_theta = config.rope_theta  # Base for rotary embeddings
        self.is_causal = True  # Whether to use causal masking (only look at previous tokens, no future tokens)

        # Step 4: Validate that the hidden size is divisible by the number of heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Step 5: Init the projection layers for attention
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        
        # Step 6: Init RoPE
        self._init_rope()


    def _init_rope(self):
        '''
        Initialize rotary position embeddings based on configuration
        - If no scaling is provided, use the default RoPE
        - If scaling is provided, use the specified scaling type and factor
        '''
        if self.config.rope_scaling is None:
            # Path 1: Initialize the default RoPE
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # Path 2: Initialize the specified RoPE
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                # Path 2.1: Initialize the linear scaling RoPE
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # Path 2.2: Initialize the dynamic scaling RoPE
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Step 1: Get Dimensions
        bsz, q_len, _ = hidden_states.size() # Batch size, sequence length
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # Step 2: Project our hidden states to queries, keys, and values
        if self.config.pretraining_tp > 1:
            # Check if we have multiple GPUs
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)


        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Step 3: Reshape the queries, keys, and values
        # This splits the representation into multiple heads, each focusing on different aspects
        # This is done to allow the model to attend to different parts of the input simultaneously
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
         
        # Step 4: Get the key-value sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # If we have cached key-values, add their length
            kv_seq_len += past_key_value[0].shape[-2]      
            
        # Step 5: Apply RoPE to the queries and keys
        # cos, sin = self.rotary_emb(value_states, position_ids)
        cos, sin = self.rotary_emb(value_states,seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
   
        # Step 6: If we have cached key-values, concatenate them with the new key-values. Useful for faster generation/decoding
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)        
        past_key_value = (key_states, value_states) if use_cache else None

        # Step 7: Repeat the key-values for each group of heads if needed...?
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Step 8: Compute the attention scores. How much each token should pay attention to each other tokens
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Step 9: Apply the attention mask if provided
        # this mask out certain positions (e.g. padding tokens or future tokens)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Step 10: Normalize the attention scores (convert to probabilities) and apply dropout
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Step 11: Generate attention output. Use the attention scores to weight the values
        attn_output = torch.matmul(attn_weights, value_states)

        # Step 12: Validate output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Step 13: Reshape the output
        # Combine the heads back together
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Step 14: Project the output back to the original size
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        
        # Step 15: Return the output, attention weights (if requested), and cached key-values
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value





class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    '''
    Handles scaled dot-product attention using torch.nn.functional.scaled_dot_product_attention.
    Inherits from LlamaAttention.
    Optimized version using torch's scaled dot product attention.
    '''

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]      
        cos, sin = self.rotary_emb(value_states,seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
   
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)        
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    '''
    Main transformer layer
    Contains:
    - Self-attention (self_attn): Handles self-attention between text tokens
    - Cross-attention (optional): Handles cross-attention between text and image
    - MLP (mlp): Handles feed-forward network
    - Layer normalization (input_layernorm, post_attention_layernorm): Handles normalization
    - Residual connections (residual, residual1): Handles residual connections
    '''
    def __init__(self, config: LlamaConfig, layer_idx: int):
        # Step 1: Basic setup - initialize the layer with configuration and layer index
        super().__init__()
        self.hidden_size = config.hidden_size
     
        self.layer_idx = layer_idx # What layer this is in the stack of layers
        self.config = config
        
        # Step 2: Create the self-attention module based on the implementation type (eager or sdpa)
        # This is the part that lets the model look at previous words to understand context
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        # Step 3: Set ViT layer selection strategy and image-text fusion strategy
        self.layer_using_strategy = config.layer_using_strategy    
        self.layer_fusing_strategy = config.layer_fusing_strategy   
        
        # Step 4: Check if this layer needs X-attention
        # Dependent on the strategy, determine where and if this layer should use cross-attention (looking at images)
        # Different strategies place cross-attention at different layers (eg. latter would do cross-attention at the last 12 layers)
        if "I" in self.layer_fusing_strategy:
            # Strategy 1 (Single): Only at layer 18/24
            if self.layer_using_strategy == '18':
                self.has_cross = layer_idx in [int(18*config.num_hidden_layers/24-1)]
            # Strategy 2 (Double): At layers 3/24 and 18/24 
            if self.layer_using_strategy == '3-18':
                self.has_cross = layer_idx in [int(3*config.num_hidden_layers/24-1),int(18*config.num_hidden_layers/24-1)]    
            # Strategy 3 (Triple): At layers 3/24, 18/24, and 23/24
            if self.layer_using_strategy == '3-18-23':
                self.has_cross = layer_idx in [int(3*config.num_hidden_layers/24-1),int(18*config.num_hidden_layers/24-1),int(23*config.num_hidden_layers/24-1)]
            # Strategy 4 (Former): In the first 12 layers
            if self.layer_using_strategy == 'former':                 
                self.has_cross = layer_idx in [0,1,2,3,4,5,6,7,8,9,10,11]
            # Strategy 5 (Latter): In the last 12 layers
            if self.layer_using_strategy == 'latter':
                self.has_cross = layer_idx in [config.num_hidden_layers - 12,config.num_hidden_layers - 11, config.num_hidden_layers - 10, config.num_hidden_layers - 9, config.num_hidden_layers - 8, config.num_hidden_layers - 7, config.num_hidden_layers - 6, config.num_hidden_layers - 5, config.num_hidden_layers - 4, config.num_hidden_layers - 3, config.num_hidden_layers - 2, config.num_hidden_layers - 1]
            # Strategy 6: In both first and last 12 layers
            if self.layer_using_strategy == 'all':
                self.has_cross = layer_idx in [0,1,2,3,4,5,6,7,8,9,10,11,config.num_hidden_layers - 12, config.num_hidden_layers - 11, config.num_hidden_layers - 10, config.num_hidden_layers - 9, config.num_hidden_layers - 8, config.num_hidden_layers - 7, config.num_hidden_layers - 6, config.num_hidden_layers - 5, config.num_hidden_layers - 4, config.num_hidden_layers - 3, config.num_hidden_layers - 2, config.num_hidden_layers - 1]
        else:
            # If no image fusion strategy is specified, this layer doesn't use cross-attention
            self.has_cross = False

        # Step 5: If this layer uses cross-attention and the fusion strategy is "I_M" (Internal Modular)
        # Create the necessary modules for cross-attention and fusion
        if self.has_cross and self.layer_fusing_strategy=="I_M":
            # only Internal Modualr Fusion needs extra modules in the LLM.
            # Create zero-initialized modules for fusion (these start with zero weights)
            self.ucross_xattn = zero_init(self.layer_idx)
            # Create cross-attention module (lets text look at image features)
            self.ucross_attn = LlamaCrossAttention(config=self.config, layer_idx=self.layer_idx)
            # Create another zero-initialized module for MLP fusion
            self.ucross_xmlp = zero_init(self.layer_idx)
            # Create MLP for cross-attention output
            self.ucross_mlp = LlamaMLP(config)

        # Step 6: Init standard MLP (feed-forward network) and Norm layers
        self.mlp = LlamaMLP(config)
        # Normalize inputs before self-attention
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Normalize after self-attention but before MLP
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        image_token_mask: Optional[torch.Tensor] = None,
        image_feature: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # For External Fusion, it always takes Path A because:
        # 1. self.has_cross is False (set in __init__)

        # Process the layer. PAths: Cross-attention or not
        # Path A: Standard processing without cross-attention but with "I_D" (Direct Insertion) strategy
        if not self.has_cross or "I_D" in self.layer_fusing_strategy:
            # Step 1: Res Connection: Initialize residual connection!!! --> From DeepStack paper
            residual = hidden_states

            # Normalize the input
            hidden_states = self.input_layernorm(hidden_states)

            # Step 2: Apply self-attention (look at previous words)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value[:2] if past_key_value is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
                )
            # Step 3: Res Connection: Add the residual connection (the oringal information + processed hidden state)
            hidden_states = residual + hidden_states

            # Fully Connected
            # Step 4: Apply the MLP (feed-forward network)
            # Res Connection: Save the input for another residual connection
            residual = hidden_states
            # Normalize after self-attention
            hidden_states = self.post_attention_layernorm(hidden_states)
            # Apply the MLP
            hidden_states = self.mlp(hidden_states)
            # Res Connection: Add the residual connection (the oringal information + processed hidden state)
            hidden_states = residual + hidden_states

            # Step 5: Prepare the output
            outputs = (hidden_states,)

            # Include attention weights if requested
            if output_attentions:
                outputs += (self_attn_weights,)

            # Include cached key-values if requested (for faster generation)
            if use_cache:
                outputs += (present_key_value,)

            return outputs
        # Path B: Processing with cross-attention (for "I_M" - Internal Modular strategy)
        else:
            # Step 1: Res Connection: Save the input for the final residual connection
            residual = hidden_states
            # Res Connection: Normalize the input. but make residual1 to use this for now hidden_states
            residual1 = self.input_layernorm(hidden_states)

            # Step 2: Apply cross-attention (look at image features)
            hidden_states, self_ucross_attn_weights, present_key_value_cross = self.ucross_attn(
                    hidden_states=residual1,
                    image_feature = image_feature,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value[-2:] if past_key_value is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
            )    
            
            # w zero
            # Step 3: Apply the zero-initialized fusion module and add residual
            # This gradually learns how to combine text and image information
            hidden_states = residual1 + self.ucross_xattn(hidden_states)

            # Step 4: Apply the cross-attention MLP with fusion
            # Res Connection: Again make a new res con, but now for ucross_xmlp ????
            residual1 = hidden_states
            # Apply MLP and fusion, then add residual
            hidden_states = residual1 + self.ucross_xmlp(self.ucross_mlp(hidden_states))

            # Step 5: Apply self-attention (look at previous words)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value[:2] if past_key_value is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
            )  
            
            # Res Connection: Add the ORIGINAL residual connection
            hidden_states = residual + hidden_states

            # Step 6: Apply the final MLP
            # Normalize after self-attention
            hidden_states = self.post_attention_layernorm(hidden_states)

            # Fully Connected
            # Apply the MLP
            mlp_output = self.mlp(hidden_states)
            # Why do we add mlp out with hidden_states??
            hidden_states = mlp_output + hidden_states

            # Step 7: Prepare the output
            outputs = (hidden_states,)

            # Include attention weights if requested
            if output_attentions:
                outputs += (self_attn_weights,)

            # Include cached key-values if requested (for faster generation)
            if use_cache:
                outputs += (present_key_value+present_key_value_cross,)

            return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    '''
    Base class for all Llama models
    Handles:
        - Weight initialization (function: _init_weights)
        - Model configuration (config_class)
        - Common utilities (base_model_prefix, supports_gradient_checkpointing, _no_split_modules, _skip_keys_device_placement, _supports_flash_attn_2, _supports_sdpa, _supports_cache_class)
    '''
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    '''
    Main model architecture
    Features:
        - Token embeddings (embed_tokens): Converts input IDs to embeddings
        - Stack of decoder layers (layers): Contains multiple LlamaDecoderLayer instances
        - Final normalization (norm): Applies final layer normalization
        - Support for different fusion strategies (self.layer_using_strategy)
    '''

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.layer_using_strategy = config.layer_using_strategy
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.log_len = []
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # self._use_sdpa = False #config._attn_implementation == "sdpa"
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value



        
    def load_cross_attn_weights(self,pretrain_mm_mlp_adapter):
        if pretrain_mm_mlp_adapter == None:
            return
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        cross_attn_weights = torch.load(pretrain_mm_mlp_adapter.replace('mm_projector.bin', 'ucross.bin'), map_location='cpu')
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'ucross_attn'):
                layer.ucross_attn.load_state_dict(get_w(cross_attn_weights, f'layers.{idx}.ucross_attn'), strict=False)
                print(f'ucross_attn in layer {idx} is loaded!!!')
                for param in layer.ucross_attn.parameters():
                    param.requires_grad = True
                layer.ucross_xattn.load_state_dict(get_w(cross_attn_weights, f'layers.{idx}.ucross_xattn'), strict=False)
                print(f'ucross_xattn in layer {idx} is loaded!!!')                
                for param in layer.ucross_xattn.parameters():
                    param.requires_grad = True
                # print(f'ucross_xattn in layer {idx}.requires_grad::::',layer.ucross_xattn.requires_grad)
                
                layer.ucross_mlp.load_state_dict(get_w(cross_attn_weights, f'layers.{idx}.ucross_mlp'), strict=False)
                print(f'ucross_mlp in layer {idx} is loaded!!!')
                for param in layer.ucross_mlp.parameters():
                    param.requires_grad = True
                layer.ucross_xmlp.load_state_dict(get_w(cross_attn_weights, f'layers.{idx}.ucross_xmlp'), strict=False)
                print(f'ucross_xmlp in layer {idx} is loaded!!!')                
                for param in layer.ucross_xmlp.parameters():
                    param.requires_grad = True
                # print(f'ucross_xattn in layer {idx}.requires_grad::::',layer.ucross_xattn.requires_grad)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        image_token_mask: torch.Tensor = None,
        images_features: List[List[torch.Tensor]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        

        seq_length_with_past = seq_length

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
            
            

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        try:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        except:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )       
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        


        # embed positions
        hidden_states = inputs_embeds
        # print("hidden_state.shape:",hidden_states.shape)
        self.log_len.append(hidden_states.shape[1])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None


        count = 0 
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if not decoder_layer.has_cross:
                layer_outputs = decoder_layer(
                        hidden_states,
                        image_token_mask = None,
                        image_feature = None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                )
            else:
                

                length = len(self.layers)
                if self.layer_using_strategy == '18':
                    layer_indices = [int(18*length/24-1)]
                if self.layer_using_strategy == '3-18':
                    layer_indices = [int(3*length/24-1),int(18*length/24-1)]    
                if self.layer_using_strategy == '3-18-23':
                    layer_indices = [int(3*length/24-1),int(18*length/24-1),int(23*length/24-1)]
                if self.layer_using_strategy == 'former':                 
                    layer_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
                if self.layer_using_strategy == 'latter':
                    layer_indices = [length - 12,length - 11, length - 10, length - 9, length - 8, length - 7, length - 6, length - 5, length - 4, length - 3, length - 2, length - 1]
                if self.layer_using_strategy == 'all':
                    layer_indices = [0,1,2,3,4,5,6,7,8,9,10,11,length - 12, length - 11, length - 10, length - 9, length - 8, length - 7, length - 6, length - 5, length - 4, length - 3, length - 2, length - 1]
                if self.layer_using_strategy == 'last':
                    layer_indices = [length - 1]





                # Map the indices of LLM decoder layers to the actual ViT layer indices
                # adjusted_layer_idx = decoder_layer.self_attn.layer_idx + 1  # ViT layers are counted starting from 1
                adjusted_layer_idx = decoder_layer.self_attn.layer_idx
                # Find the position of adjusted_layer_idx in layer_indices
                if adjusted_layer_idx in layer_indices:
                    index = layer_indices.index(adjusted_layer_idx)
                    image_f = images_features[index]
                if self.layer_using_strategy == 'I_D':    
                    if past_key_value is None and decoder_layer.has_cross:
                        for batch_idx in range(hidden_states.shape[0]):
                            cur_image_mask = image_token_mask[batch_idx]
                            img_token_indices = torch.where(cur_image_mask == 1)[0]
                            if img_token_indices.shape[0]!=0:
                                hidden_states[batch_idx, img_token_indices[0]:img_token_indices[0]+img_token_indices.shape[0]] = hidden_states[batch_idx, img_token_indices[0]:img_token_indices[0]+img_token_indices.shape[0]]+image_f[batch_idx]
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )    

                elif self.layer_using_strategy == 'I_M':  
                    layer_outputs = decoder_layer(
                            hidden_states,
                            image_token_mask = image_token_mask,
                            image_feature = image_f,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                    )     

            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)


            if output_attentions:
                all_self_attns += (layer_outputs[1],)  
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):

    '''
    Language model with causal masking
    Adds:
        - Language modeling head (lm_head): Applies final linear layer to hidden states
        - Generation utilities (prepare_inputs_for_generation, _reorder_cache, _prepare_past_key_values)
        - Loss computation (forward, prepare_inputs_for_generation)
    '''
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,image_token_mask = None, images_features = None, **kwargs
    ):

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_token_mask" : image_token_mask,
                "images_features" : images_features,

            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    '''
    Classification head on top of LLaMA
    Adds:
        - Classification head (score): Applies final linear layer to hidden states
        - Sequence classification logic (forward): Handles sequence classification
        - Loss computation for classification (forward): Computes loss for classification
    '''
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
