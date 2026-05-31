# coding=utf-8
"""
Whisper + Titans Integration — MAL (Memory as Layer) variant.
Fixed: BCE loss computed on gate_logits only. CE loss removed entirely.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

from MemoryModule.conponents.LMMBlock import LMMBlock
from MemoryModule.conponents.memory import MemoryState
from MemoryModule.conponents.config import TitansConfig

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 1
_CONFIG_FOR_DOC   = "WhisperConfig"
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"


# ─────────────────────────────────────────────────────────────────────────────
# Custom output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Seq2SeqLMOutputWithGate(Seq2SeqLMOutput):
    """
    Extends Seq2SeqLMOutput with a separate gate_logits field.

    Fields
    ------
    loss        : BCE scalar (only when labels are passed)
    logits      : (B, T) float  — gate sigmoid probabilities ∈ [0, 1]
                  This is what Seq2SeqTrainer reads as predictions.
    gate_logits : (B, T) float  — same as logits; explicit alias for clarity
                  in GateTrainer.compute_loss.
    """
    gate_logits: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError(f"channels must be divisible by 2, got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time    = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0]  = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("pad_token_id must be defined.")
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def _compute_mask_indices(shape, mask_prob, mask_length, attention_mask=None, min_masks=0):
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")
    if mask_length > sequence_length:
        raise ValueError(f"`mask_length` > `sequence_length`: {mask_length} > {sequence_length}")
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        num = int(mask_prob * input_length / mask_length + epsilon)
        num = max(num, min_masks)
        if num * mask_length > sequence_length:
            num = sequence_length // mask_length
        if input_length - (mask_length - 1) < num:
            num = max(input_length - (mask_length - 1), 0)
        return num

    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length] * batch_size
    )
    spec_aug_mask     = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []
    max_num_masked_span = compute_num_masked_span(sequence_length)
    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        num = compute_num_masked_span(input_length)
        idx = np.random.choice(np.arange(input_length - (mask_length - 1)), num, replace=False)
        if len(idx) == 0:
            idx = np.array([sequence_length - 1])
        idx = np.concatenate(
            [idx, np.ones(max_num_masked_span - num, dtype=np.int32) * idx[0]]
        )
        spec_aug_mask_idxs.append(idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    ).reshape(batch_size, max_num_masked_span * mask_length)
    offsets = np.broadcast_to(
        np.arange(mask_length)[None, None, :], (batch_size, max_num_masked_span, mask_length)
    ).reshape(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    return spec_aug_mask


# ─────────────────────────────────────────────────────────────────────────────
# Attention classes  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        return self.weight[position_ids]


class WhisperAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False,
                 bias=True, is_causal=False, layer_idx=None, config=None):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.dropout    = dropout
        self.head_dim   = embed_dim // num_heads
        self.config     = config
        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.scaling    = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal  = is_causal
        if layer_idx is None and is_decoder:
            logger.warning_once("Instantiating decoder attention without layer_idx.")
        self.layer_idx = layer_idx
        self.k_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj    = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _    = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states   = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states   = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cp = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cp}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Head mask size mismatch: {layer_head_mask.size()}")
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs  = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"attn_output size mismatch: {attn_output.size()}")
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class WhisperFlashAttention2(WhisperAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None):
        if isinstance(past_key_value, StaticCache):
            raise ValueError("StaticCache incompatible with flash_attention_2.")
        if output_attentions:
            raise ValueError("WhisperFlashAttention2 does not support output_attentions")
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _    = hidden_states.size()
        query_states = torch.reshape(
            self.q_proj(hidden_states), (bsz, tgt_len, self.num_heads, self.head_dim)
        )
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states   = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states   = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cp = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cp}
                )
        key_states   = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        causal_mask  = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states   = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = _flash_attention_forward(
            query_states, key_states, value_states, causal_mask, tgt_len,
            dropout=self.dropout, is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


class WhisperSdpaAttention(WhisperAttention):
    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None):
        if output_attentions or layer_head_mask is not None:
            logger.warning_once("WhisperSdpaAttention falling back to eager for unsupported args.")
            return super().forward(
                hidden_states, key_value_states=key_value_states,
                past_key_value=past_key_value, attention_mask=attention_mask,
                layer_head_mask=layer_head_mask, output_attentions=output_attentions,
                cache_position=cache_position,
            )
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _    = hidden_states.size()
        query_states       = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states   = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states   = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cp = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cp}
                )
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        is_causal   = True if self.is_causal and causal_mask is None and tgt_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"attn_output size mismatch: {attn_output.size()}")
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


WHISPER_ATTENTION_CLASSES = {
    "eager":             WhisperAttention,
    "flash_attention_2": WhisperFlashAttention2,
    "sdpa":              WhisperSdpaAttention,
}


# ─────────────────────────────────────────────────────────────────────────────
# Encoder layer  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout, config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout              = config.dropout
        self.activation_fn        = ACT2FN[config.activation_function]
        self.activation_dropout   = config.activation_dropout
        self.fc1                  = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2                  = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm     = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions=False):
        residual      = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            layer_head_mask=layer_head_mask, output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual      = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value   = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Decoder layer  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.layer_idx = layer_idx
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout, is_decoder=True, is_causal=True,
            layer_idx=layer_idx, config=config,
        )
        self.dropout              = config.dropout
        self.activation_fn        = ACT2FN[config.activation_function]
        self.activation_dropout   = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn         = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim, config.decoder_attention_heads,
            dropout=config.attention_dropout, is_decoder=True,
            layer_idx=layer_idx, config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1              = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2              = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.use_mem = (layer_idx is not None) and (layer_idx >= config.decoder_layers - 1)
        if self.use_mem:
            titans_cfg = TitansConfig(
                dim=config.d_model, num_heads=config.decoder_attention_heads,
                num_layers=config.decoder_layers, vocab_size=config.vocab_size,
                chunk_size=64, window_size=64, num_memory_layers=3,
            )
            self.mem_block       = LMMBlock(titans_cfg)
            self.router_proj_fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=True)
            self.router_proj_fc2 = nn.Linear(4 * config.d_model, 1,              bias=True)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, layer_head_mask=None,
                cross_attn_layer_head_mask=None, past_key_value=None,
                output_attentions=False, use_cache=True, cache_position=None,
                mem_state=None, lang_embed=None):
        # 1. Self-attention
        residual = hidden_states
        normed   = self.self_attn_layer_norm(hidden_states)
        sa_out, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=normed, past_key_value=past_key_value,
            attention_mask=attention_mask, layer_head_mask=layer_head_mask,
            output_attentions=output_attentions, cache_position=cache_position,
        )
        hidden_states = residual + nn.functional.dropout(
            sa_out, p=self.dropout, training=self.training
        )

        # 2. Cross-attention
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual      = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_pkv = self.encoder_attn(
                hidden_states=hidden_states, key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value, output_attentions=output_attentions,
            )
            hidden_states     = residual + nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            present_key_value = (present_key_value, cross_attn_pkv)

        # 3. FFN (all layers)
        residual      = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # 4. MAL block + gate (use_mem layers only)
        new_mem_state: Optional[MemoryState]  = None
        gate_output:   Optional[torch.Tensor] = None

        if self.use_mem:
            _, new_mem_state = self.mem_block(
                hidden_states, encoder_hidden_states=encoder_hidden_states, state=mem_state,
            )
            h           = F.relu(self.router_proj_fc1(hidden_states))
            gate_logits = self.router_proj_fc2(h * h)           # (B, T, 1)
            gate_output = torch.sigmoid(gate_logits)             # (B, T, 1) ∈ [0, 1]

        # 5. Pack outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (gate_output,)    # [-2]  None for non-MAL layers
        outputs += (new_mem_state,)  # [-1]  None for non-MAL layers
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# WhisperPreTrainedModel base  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperPreTrainedModel(PreTrainedModel):
    config_class                    = WhisperConfig
    base_model_prefix               = "model"
    main_input_name                 = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules               = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2          = True
    _supports_sdpa                  = True
    _supports_cache_class           = True
    _supports_static_cache          = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperEncoder):
            with torch.no_grad():
                ep = module.embed_positions.weight
                ep.copy_(sinusoids(*ep.shape))

    def _get_feat_extract_output_lengths(self, input_lengths):
        return (input_lengths - 1) // 2 + 1


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperEncoder(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout              = config.dropout
        self.layerdrop            = config.encoder_layerdrop
        embed_dim                 = config.d_model
        self.num_mel_bins         = config.num_mel_bins
        self.padding_idx          = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale          = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.conv1                = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2                = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions      = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers               = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm           = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self):    return self.conv1
    def set_input_embeddings(self, v): self.conv1 = v

    def forward(self, input_features, attention_mask=None, head_mask=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Expected mel length {expected_seq_length}, got {input_features.shape[-1]}"
            )
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        hidden_states = inputs_embeds + self.embed_positions.weight
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions    else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                if torch.rand([]) < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__, hidden_states, None,
                        head_mask[idx] if head_mask is not None else None,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states, None,
                        layer_head_mask=head_mask[idx] if head_mask is not None else None,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperDecoder(WhisperPreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout              = config.dropout
        self.layerdrop            = config.decoder_layerdrop
        self.padding_idx          = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale          = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens         = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions      = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model
        )
        self.layers = nn.ModuleList([
            WhisperDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)
        ])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa              = config._attn_implementation == "sdpa"
        self.layer_norm             = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.mem_states:            dict = {}
        self._last_gate_output:     Optional[torch.Tensor] = None
        self.lang_embed:            Optional[torch.Tensor] = None
        self.post_init()

    def reset_mem_states(self) -> None:
        self.mem_states        = {}
        self._last_gate_output = None

    def get_input_embeddings(self):    return self.embed_tokens
    def set_input_embeddings(self, v): self.embed_tokens = v

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                head_mask=None, cross_attn_head_mask=None, past_key_values=None,
                inputs_embeds=None, position_ids=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids   = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Specify decoder_input_ids or decoder_inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if input_shape[1] > 1:
            self.lang_embed = inputs_embeds[:, 1, :]

        return_legacy_cache = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                past_key_values     = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        src_for_pos = input_ids if input_ids is not None else inputs_embeds
        positions   = self.embed_positions(
            src_for_pos, past_key_values_length=past_key_values_length, position_ids=position_ids,
        )
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting False.")
            use_cache = False

        all_hidden_states    = () if output_hidden_states else None
        all_self_attns       = () if output_attentions    else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        last_gate_output: Optional[torch.Tensor] = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training and torch.rand([]) < self.layerdrop:
                continue

            current_mem_state = self.mem_states.get(idx, None)

            if self.gradient_checkpointing and self.training:
                _lang_embed = self.lang_embed
                def make_ckpt_call(layer, lang_embed):
                    def ckpt_call(hidden_states, causal_mask, encoder_hidden_states,
                                  _u1, layer_head_mask, cross_attn_layer_head_mask,
                                  _u2, output_attentions, use_cache, cache_position, mem_state):
                        return layer(
                            hidden_states, attention_mask=causal_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            layer_head_mask=layer_head_mask,
                            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                            past_key_value=None, output_attentions=output_attentions,
                            use_cache=use_cache, cache_position=cache_position,
                            mem_state=mem_state, lang_embed=lang_embed,
                        )
                    return ckpt_call
                layer_outputs = self._gradient_checkpointing_func(
                    make_ckpt_call(decoder_layer, _lang_embed),
                    hidden_states, causal_mask, encoder_hidden_states, None,
                    head_mask[idx]            if head_mask            is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None, output_attentions, use_cache, cache_position, current_mem_state,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states, attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=head_mask[idx]            if head_mask            is not None else None,
                    cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    past_key_value=past_key_values if use_cache else None,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position, mem_state=current_mem_state,
                    lang_embed=self.lang_embed,
                )

            hidden_states = layer_outputs[0]
            new_mem_state = layer_outputs[-1]
            gate_out      = layer_outputs[-2]

            if new_mem_state is not None:
                self.mem_states[idx] = new_mem_state
            if gate_out is not None:
                last_gate_output = gate_out

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        self._last_gate_output = last_gate_output

        next_cache = past_key_values if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions,
            ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states, past_key_values=next_cache,
            hidden_states=all_hidden_states, attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position,
                            past_key_values, output_attentions):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens   = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if (self.config._attn_implementation == "sdpa" and not using_static_cache
                and not output_attentions):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens, is_training=self.training,
            ):
                return None

        dtype, device   = input_tensor.dtype, input_tensor.device
        min_dtype       = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length   = (
            past_key_values.get_max_length() if using_static_cache else
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else
            past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("4D attention mask must be in inverted form (max==0)")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask  = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask  = causal_mask.clone()
                mask_length  = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask == 0, min_dtype
                )

        if (self.config._attn_implementation == "sdpa" and attention_mask is not None
                and attention_mask.device.type == "cuda" and not output_attentions):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# ─────────────────────────────────────────────────────────────────────────────
# WhisperModel  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        self.post_init()

    def get_input_embeddings(self):    return self.decoder.embed_tokens
    def set_input_embeddings(self, v): self.decoder.embed_tokens = v
    def get_encoder(self):             return self.encoder
    def get_decoder(self):             return self.decoder
    def freeze_encoder(self):          self.encoder._freeze_parameters()

    def _mask_input_features(self, input_features, attention_mask=None):
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features
        batch_size, hidden_size, sequence_length = input_features.size()
        if self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = torch.tensor(
                _compute_mask_indices(
                    (batch_size, sequence_length),
                    mask_prob=self.config.mask_time_prob,
                    mask_length=self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=self.config.mask_time_min_masks,
                ),
                device=input_features.device, dtype=torch.bool
            )
            input_features[mask_time_indices[:, None].expand(-1, hidden_size, -1)] = 0
        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = torch.tensor(
                _compute_mask_indices(
                    (batch_size, hidden_size),
                    mask_prob=self.config.mask_feature_prob,
                    mask_length=self.config.mask_feature_length,
                    min_masks=self.config.mask_feature_min_masks,
                ),
                device=input_features.device, dtype=torch.bool
            )
            input_features[mask_feature_indices] = 0
        return input_features

    def forward(self, input_features=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, head_mask=None, decoder_head_mask=None,
                cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None,
                decoder_inputs_embeds=None, decoder_position_ids=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features  = self._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.encoder(
                input_features, head_mask=head_mask,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]    if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0], head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds, position_ids=decoder_position_ids,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# WhisperForConditionalGeneration  — FIXED: BCE on gate_logits only
# ─────────────────────────────────────────────────────────────────────────────

class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    """
    Gate-training variant of Whisper.

    Forward contract
    ----------------
    inputs:
        input_features    : (B, n_mels, T_enc)
        decoder_input_ids : (B, T_dec)  int64   — full gold sequence, shifted right
        labels            : (B, T_dec)  float32 — gate targets 0.0/1.0, -100=ignore
                            Must be passed together with decoder_input_ids.
                            Do NOT pass labels alone — the model will NOT derive
                            decoder_input_ids from labels.

    outputs (Seq2SeqLMOutputWithGate):
        loss        : scalar BCE (only when labels provided)
        logits      : (B, T_dec) float32 ∈ [0,1] — gate sigmoid probabilities
                      Used by Seq2SeqTrainer as predictions for compute_metrics.
        gate_logits : same tensor as logits (explicit alias for GateTrainer)

    Loss
    ----
    ONLY binary cross-entropy on gate_logits vs labels.
    CrossEntropyLoss on lm_logits is intentionally ABSENT.
    proj_out is kept for weight-tying compatibility but its output is
    not part of the loss computation.
    """

    base_model_prefix  = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model    = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):              return self.model.get_encoder()
    def get_decoder(self):              return self.model.get_decoder()
    def get_output_embeddings(self):    return self.proj_out
    def set_output_embeddings(self, e): self.proj_out = e
    def get_input_embeddings(self):     return self.model.get_input_embeddings()
    def freeze_encoder(self):           self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,            # (B, T) float32: gate targets 0.0/1.0, -100=pad
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── IMPORTANT: decoder_input_ids must be supplied by the caller. ──────
        # During gate training the collator always provides decoder_input_ids
        # (the shifted full gold sequence).  We never derive decoder_input_ids
        # from labels because labels are float gate targets, not token IDs.
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                "When passing `labels` for gate training you must also pass "
                "`decoder_input_ids`.  The model will NOT derive decoder_input_ids "
                "from float gate labels."
            )

        # Reset Titans memory at the start of every new utterance.
        if past_key_values is None:
            self.model.decoder.reset_mem_states()

        # ── Run encoder + decoder ─────────────────────────────────────────────
        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # ── Gate logits ───────────────────────────────────────────────────────
        # Shape: (B, T, 1) → squeeze → (B, T)  ∈ [0, 1]
        # This is the ONLY output used for loss and metrics.
        gate_raw = self.model.decoder._last_gate_output   # (B, T, 1)
        if gate_raw is None:
            raise RuntimeError(
                "No MAL layer produced a gate output.  "
                "Check WhisperDecoderLayer.use_mem — it must be True for at least one layer."
            )
        # Cast before squeeze so fp16/bf16 training stays numerically stable
        gate_logits = gate_raw.squeeze(-1).to(outputs[0].dtype)   # (B, T) ∈ [0, 1]

        # ── BCE loss  (only gate_logits vs float labels; NO CE on lm_logits) ──
        loss = None
        if labels is not None:
            labels_f = labels.to(dtype=gate_logits.dtype, device=gate_logits.device)

            # Mask positions labelled -100 (HuggingFace padding convention)
            valid = labels_f != -100.0   # (B, T) bool

            if valid.any():
                # F.binary_cross_entropy expects probabilities (not raw logits)
                # because gate_logits is already passed through sigmoid above.
                loss = F.binary_cross_entropy(
                    gate_logits[valid],   # predicted probabilities ∈ [0, 1]
                    labels_f[valid],      # targets ∈ {0.0, 1.0}
                    reduction="mean",
                )
            else:
                # Edge case: all positions masked (should not occur in practice)
                loss = gate_logits.sum() * 0.0   # keeps graph alive, zero grad

        # ── proj_out forward (weight-tying contract; NOT used in loss) ────────
        # Called so that proj_out weights receive at least a forward pass,
        # preventing DDP unused-parameter errors.  Detached from the BCE graph.
        with torch.no_grad():
            _ = self.proj_out(outputs[0])

        if not return_dict:
            output = (gate_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputWithGate(
            loss=loss,
            logits=gate_logits,        # (B, T)  trainer reads this as predictions
            gate_logits=gate_logits,   # explicit alias for GateTrainer.compute_loss
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past_key_values=None, use_cache=None,
        encoder_outputs=None, attention_mask=None, decoder_attention_mask=None,
        cache_position=None, **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = (
                    cache_position[0] if cache_position is not None
                    else past_key_values.get_seq_length()
                )
            else:
                past_length = past_key_values[0][0].shape[2]
            remove_prefix_length = (
                past_length if decoder_input_ids.shape[1] > past_length
                else decoder_input_ids.shape[1] - 1
            )
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
            if (decoder_position_ids is not None
                    and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]):
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1],
                device=decoder_input_ids.device,
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1]:]

        return {
            "encoder_outputs":        encoder_outputs,
            "past_key_values":        past_key_values,
            "decoder_input_ids":      decoder_input_ids,
            "use_cache":              use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids":   decoder_position_ids,
            "cache_position":         cache_position,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Remaining classes  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperDecoderWrapper(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.decoder = WhisperDecoder(config)
    def get_input_embeddings(self):     return self.decoder.embed_tokens
    def set_input_embeddings(self, v):  self.decoder.embed_tokens = v
    def forward(self, *args, **kwargs): return self.decoder(*args, **kwargs)


class WhisperForCausalLM(WhisperPreTrainedModel):
    _tied_weights_keys = ["proj_out.weight"]
    main_input_name    = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.model    = WhisperDecoderWrapper(config)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):    return self.proj_out
    def set_output_embeddings(self, e): self.proj_out = e
    def get_input_embeddings(self):     return self.model.get_input_embeddings()
    def set_input_embeddings(self, v):  self.model.set_input_embeddings(v)
    def set_decoder(self, d):           self.model.decoder = d
    def get_decoder(self):              return self.model.decoder

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
                head_mask=None, cross_attn_head_mask=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, cache_position=None):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict
        if isinstance(encoder_outputs, (BaseModelOutput, tuple, list)):
            encoder_outputs = encoder_outputs[0]
        outputs = self.model.decoder(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs, head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position,
        )
        logits = self.proj_out(outputs[0])
        loss   = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss   = CrossEntropyLoss()(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None,
                                       encoder_outputs=None, attention_mask=None,
                                       cache_position=None, **kwargs):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, (Cache, EncoderDecoderCache)):
                past_length = (
                    cache_position[0] if cache_position is not None
                    else past_key_values.get_seq_length()
                )
            else:
                past_length = past_key_values[0][0].shape[2]
            remove_prefix_length = (
                past_length if input_ids.shape[1] > past_length
                else input_ids.shape[1] - 1
            )
            input_ids = input_ids[:, remove_prefix_length:]
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device,
            )
        elif use_cache:
            cache_position = cache_position[-input_ids.shape[1]:]
        return {
            "encoder_outputs": encoder_outputs, "past_key_values": past_key_values,
            "input_ids": input_ids, "use_cache": use_cache,
            "attention_mask": attention_mask, "cache_position": cache_position,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(s.index_select(0, beam_idx.to(s.device)) for s in lp)
            for lp in past_key_values
        )


class WhisperForAudioClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder  = WhisperEncoder(config)
        num_layers    = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector  = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_encoder(self):          self.encoder._freeze_parameters()
    def get_input_embeddings(self):    return self.encoder.get_input_embeddings()
    def set_input_embeddings(self, v): self.encoder.set_input_embeddings(v)

    def forward(self, input_features=None, head_mask=None, encoder_outputs=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features, head_mask=head_mask,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights  = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits        = self.classifier(pooled_output)
        loss          = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss   = CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    def __init__(self, config):
        super().__init__(config)
        self.encoder  = WhisperEncoder(config)
        num_layers    = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector  = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_encoder(self):          self.encoder._freeze_parameters()
    def get_input_embeddings(self):    return self.encoder.get_input_embeddings()
    def set_input_embeddings(self, v): self.encoder.set_input_embeddings(v)

    def forward(
        self,
        input_features=None,
        head_mask=None,
        encoder_outputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights  = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits        = self.classifier(pooled_output)
        loss          = None

        if labels is not None:
            labels = labels.to(logits.device)
            loss   = CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    def __init__(self, config):
        super().__init__(config)
        self.encoder  = WhisperEncoder(config)
        num_layers    = config.num_hidden_layers + 1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector  = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_encoder(self):          self.encoder._freeze_parameters()
    def get_input_embeddings(self):    return self.encoder.get_input_embeddings()
    def set_input_embeddings(self, v): self.encoder.set_input_embeddings(v)

    def forward(
        self, input_features=None, head_mask=None, encoder_outputs=None,
        labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features, head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights  = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits        = self.classifier(pooled_output)
        loss          = None

        if labels is not None:
            labels = labels.to(logits.device)
            loss   = CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )