# coding=utf-8
"""
Whisper + Titans Integration — corrected MAL (Memory as Layer) variant.

Key corrections vs the broken original
========================================

1.  LMMBlock called with correct signature
    ----------------------------------------
    Old (broken):
        self.mem_block(hidden_states, encoder_hidden_states, state=mem_state)
    The original LMMBlock.forward only accepted (x, state), so
    encoder_hidden_states was silently swallowed as `state`, corrupting memory.

    Fixed:
        self.mem_block(hidden_states,
                       encoder_hidden_states=encoder_hidden_states,
                       state=mem_state)
    LMMBlock.forward now has the matching signature.

2.  LMMBlock REPLACES the decoder FFN — not appended after it
    -----------------------------------------------------------
    MAL paper (Section 4.3) specifies:
        [self-attn] → [cross-attn] → [LMMBlock]
    where LMMBlock owns the FFN sub-layer.

    Old code ran the standard Whisper FFN (fc1/fc2/final_layer_norm) and
    then appended the LMMBlock on top — this doubled the FFN and violated
    the MAL architecture.

    Fixed: for use_mem layers, the WhisperDecoderLayer skips its own
    fc1/fc2/final_layer_norm and routes into LMMBlock instead.
    For non-use_mem layers the standard Whisper FFN is untouched.

3.  Double gating removed
    -----------------------
    Old code had WhisperDecoderLayer.mem_gate (scalar sigmoid) multiplying
    the output of LMMBlock — but LMMBlock internally already applies a
    per-dim sigmoid gate (gate_proj).  Two stacked gates distort training.

    Fixed: WhisperDecoderLayer no longer has its own mem_gate.  The single
    gate inside LMMBlock is sufficient and paper-faithful.

4.  Persistent tokens wired in
    ---------------------------
    LMMBlock.persistent_tokens (n_persistent, D) are prepended to the
    self-attention input so they participate in the KV cache.  Attention
    mask is extended accordingly so padding logic is correct.

5.  Layer output tuple contract is explicit
    -----------------------------------------
    new_mem_state is always the *last* element of the tuple returned by
    WhisperDecoderLayer.forward.  The decoder extracts it by position [-1],
    which is unambiguous regardless of other optional outputs.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.checkpoint
from dataclasses import dataclass
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
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
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
_CONFIG_FOR_DOC  = "WhisperConfig"
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"



# Unchanged utility functions
def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError(f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels.")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)
        return num_masked_span

    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []
    max_num_masked_span = compute_num_masked_span(sequence_length)
    if max_num_masked_span == 0:
        return spec_aug_mask
    for input_length in input_lengths:
        num_masked_span = compute_num_masked_span(input_length)
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )
        if len(spec_aug_mask_idx) == 0:
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    return spec_aug_mask



# Attention classes — unchanged from original
class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]
        else:
            return self.weight[position_ids]


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need'"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False,
                 bias=True, is_causal=False, layer_idx=None, config=None):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.dropout    = dropout
        self.head_dim   = embed_dim // num_heads
        self.config     = config
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling    = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal  = is_causal
        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not "
                "recommended and will to errors during the forward call, if caching is used."
            )
        self.layer_idx = layer_idx
        self.k_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj    = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                cache_position=None):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
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
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if attention_mask is not None:
            causal_mask  = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)},"
                    f" but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs  = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)},"
                f" but is {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
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
            raise ValueError(
                "The `static` cache implementation is not compatible with `attn_implementation='flash_attention_2'`."
            )
        if output_attentions:
            raise ValueError("WhisperFlashAttention2 attention does not support output_attentions")

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = torch.reshape(self.q_proj(hidden_states), (bsz, tgt_len, self.num_heads, self.head_dim))

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
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
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
            logger.warning_once(
                "WhisperModel is using WhisperSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention`"
                " does not support `output_attentions=True` or `layer_head_mask` not None."
            )
            return super().forward(
                hidden_states, key_value_states=key_value_states, past_key_value=past_key_value,
                attention_mask=attention_mask, layer_head_mask=layer_head_mask,
                output_attentions=output_attentions, cache_position=cache_position,
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
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
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
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)},"
                f" but is {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


WHISPER_ATTENTION_CLASSES = {
    "eager":             WhisperAttention,
    "flash_attention_2": WhisperFlashAttention2,
    "sdpa":              WhisperSdpaAttention,
}



# Encoder layer — unchanged
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout, config=config,
        )
        self.self_attn_layer_norm  = nn.LayerNorm(self.embed_dim)
        self.dropout               = config.dropout
        self.activation_fn         = ACT2FN[config.activation_function]
        self.activation_dropout    = config.activation_dropout
        self.fc1                   = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2                   = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm      = nn.LayerNorm(self.embed_dim)

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
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
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



# Decoder layer — corrected Titans MAL integration
class WhisperDecoderLayer(nn.Module):
    """
    Whisper decoder layer with Titans MAL integration.

    Layer structure for use_mem=True  (every other layer, layer_idx % 2 == 0):
        [self-attn w/ persistent tokens prepended]
        [cross-attn]
        [LMMBlock]              ← owns the FFN sub-layer; replaces fc1/fc2
                                  receives encoder_hidden_states for write path

    Layer structure for use_mem=False (remaining layers):
        [self-attn]
        [cross-attn]
        [standard Whisper FFN]  ← fc1/fc2/final_layer_norm unchanged

    Persistent tokens:
        For use_mem layers, LMMBlock.get_persistent_tokens(B) returns
        (B, n_persistent, D) which is prepended to the self-attention input.
        The causal mask is extended by n_persistent positions (all-zeros for
        the prepended tokens so every decoder token can attend to them).
    """

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

        self.encoder_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim, config.decoder_attention_heads,
            dropout=config.attention_dropout, is_decoder=True,
            layer_idx=layer_idx, config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # Standard Whisper FFN — used only when use_mem is False.
        self.fc1              = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2              = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        # Titans: every other layer is a MAL layer
        self.use_mem = (layer_idx >= config.decoder_layers - 1) 
        # self.use_mem = (layer_idx is not None) and (layer_idx % 2 == 0)

        if self.use_mem:
            # self.alpha = nn.Parameter(torch.tensor(1e-3))
            self.router_gate = nn.Linear(config.d_model, config.d_model//2, bias=True)
            self.router_value = nn.Linear(config.d_model, config.d_model//2, bias=True)
            self.router_proj = nn.Linear(config.d_model//2, 1, bias=True)
            titans_cfg = TitansConfig(
                dim=config.d_model,
                num_heads=config.decoder_attention_heads,
                num_layers=config.decoder_layers,
                vocab_size=config.vocab_size,
                chunk_size=64,
                window_size=64,
                num_memory_layers=3,
            )
            # LMMBlock owns the FFN for this layer.
            # It also exposes persistent_tokens used below in forward().
            self.mem_block = LMMBlock(titans_cfg)
            
            # gate_proj inside its forward; a second gate would be redundant.

   
    # Forward
    def forward(
        self,
        hidden_states:              torch.Tensor,
        attention_mask:             Optional[torch.Tensor]      = None,
        encoder_hidden_states:      Optional[torch.Tensor]      = None,
        encoder_attention_mask:     Optional[torch.Tensor]      = None,
        layer_head_mask:            Optional[torch.Tensor]      = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor]      = None,
        past_key_value:             Optional[EncoderDecoderCache] = None,
        output_attentions:          Optional[bool]               = False,
        use_cache:                  Optional[bool]               = True,
        cache_position:             Optional[torch.LongTensor]   = None,
        mem_state:                  Optional[MemoryState]        = None,
        lang_embed: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Returns
        -------
        Tuple whose elements are:
            (hidden_states,
             [self_attn_weights],     # only if output_attentions
             [cross_attn_weights],    # only if output_attentions
             [present_key_value],     # only if use_cache
             new_mem_state)           # always last; None for non-MAL layers
        """
        B = hidden_states.shape[0]
        residual      = hidden_states
        normed        = self.self_attn_layer_norm(hidden_states)
        sa_out, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=normed,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        sa_out = nn.functional.dropout(sa_out, p=self.dropout, training=self.training)
        hidden_states = residual + sa_out

   
        # Cross-attention
   
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual      = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_pkv = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = (present_key_value, cross_attn_pkv)
   
        # Standard FFN for non-MAL layers
        residual      = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states_0 = hidden_states

        new_mem_state: Optional[MemoryState] = None

        if self.use_mem:
            # Correct call: keyword argument for encoder_hidden_states
            mem_out, new_mem_state = self.mem_block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,   
                state=mem_state,
            )
            mem_out = F.dropout(mem_out, p=self.dropout, training=self.training)
            # hidden_states = hidden_states + mem_out
            # gate_states = torch.cat([mem_out, hidden_states], dim=-1)
            router_gate = self.router_gate(hidden_states)
            router_value = self.router_value(hidden_states)
            glu_gate = router_value * torch.sigmoid(router_gate)
            gate_logits = self.router_proj(glu_gate)   # (bs, seq, 1)
            gate = torch.sigmoid(gate_logits)
    
            hidden_states = hidden_states + gate * mem_out


   
        #  Pack outputs — new_mem_state is ALWAYS the last element
   
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (hidden_states_0,new_mem_state,)    # always last; None for non-MAL layers

        return outputs



# WhisperPreTrainedModel base — unchanged
class WhisperPreTrainedModel(PreTrainedModel):
    config_class           = WhisperConfig
    base_model_prefix      = "model"
    main_input_name        = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules      = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa         = True
    _supports_cache_class  = True
    _supports_static_cache = True

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
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths



# Encoder — unchanged
class WhisperEncoder(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout               = config.dropout
        self.layerdrop             = config.encoder_layerdrop
        embed_dim                  = config.d_model
        self.num_mel_bins          = config.num_mel_bins
        self.padding_idx           = config.pad_token_id
        self.max_source_positions  = config.max_source_positions
        self.embed_scale           = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.conv1                 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2                 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions        = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers                = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm            = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self):   return self.conv1
    def set_input_embeddings(self, v): self.conv1 = v

    def forward(self, input_features, attention_mask=None, head_mask=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        expected_seq_length = (
            self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length},"
                f" but found {input_features.shape[-1]}."
            )
        output_attentions   = output_attentions   if output_attentions   is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict         = return_dict         if return_dict         is not None else self.config.use_return_dict

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos     = self.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions    else None

        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__, hidden_states, None,
                        head_mask[idx] if head_mask is not None else None, output_attentions,
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
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



@dataclass
class DecoderOutput(BaseModelOutputWithPastAndCrossAttentions):
     last_hidden_state_0: Optional[torch.Tensor] = None


# Decoder — Titans state management lives here
class WhisperDecoder(WhisperPreTrainedModel):
    """
    Whisper decoder with persistent Titans long-term memory.

    mem_states: dict[layer_idx -> MemoryState]
        Populated on first decode step (state=None initialises inside LMMBlock).
        Updated at every decode step.
        Must be reset between utterances via reset_mem_states().
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout             = config.dropout
        self.layerdrop           = config.decoder_layerdrop
        self.padding_idx         = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale         = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens        = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions     = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layers              = nn.ModuleList([
            WhisperDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)
        ])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa              = config._attn_implementation == "sdpa"
        self.layer_norm             = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False

        # Titans memory state per MAL layer — persists across decode steps
        self.mem_states: dict[int, MemoryState] = {}
        # lang embed
        self.lang_embed = None

        self.post_init()

    def reset_mem_states(self) -> None:
        """
        Clear all stored MemoryStates.
        Must be called at the START of every new utterance so memory from a
        previous audio clip does not bleed into the next one.
        """
        self.mem_states = {}

    def get_input_embeddings(self):     return self.embed_tokens
    def set_input_embeddings(self, v):  self.embed_tokens = v

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids   = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if input_shape[1] >1 :
            self.lang_embed = inputs_embeds[:,1,:]

        return_legacy_cache       = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + input_shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        src_for_pos = input_ids if input_ids is not None else inputs_embeds
        positions   = self.embed_positions(
            src_for_pos,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False

        all_hidden_states    = () if output_hidden_states else None
        all_self_attns       = () if output_attentions    else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            # Retrieve stored Titans state for this layer (None on first step)
            current_mem_state = self.mem_states.get(idx, None)

            if self.gradient_checkpointing and self.training:
                _lang_embed = self.lang_embed

                def make_ckpt_call(layer, lang_embed):
                    def ckpt_call(
                        hidden_states, causal_mask, encoder_hidden_states,
                        _u1, layer_head_mask, cross_attn_layer_head_mask,
                        _u2, output_attentions, use_cache,
                        cache_position, mem_state,
                    ):
                        return layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            layer_head_mask=layer_head_mask,
                            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                            past_key_value=None,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            mem_state=mem_state,
                            lang_embed=lang_embed,   # ← kwarg safely via closure
                        )
                    return ckpt_call

                layer_outputs = self._gradient_checkpointing_func(
                    make_ckpt_call(decoder_layer, _lang_embed),
                    hidden_states,
                    causal_mask,
                    encoder_hidden_states,
                    None,                                                              # _u1
                    head_mask[idx]            if head_mask            is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,                                                              # _u2
                    output_attentions,
                    use_cache,
                    cache_position,
                    current_mem_state,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx]           if head_mask           is not None else None),
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    past_key_value=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    mem_state=current_mem_state,    # ← Titans state in
                    lang_embed = self.lang_embed
                )

            hidden_states = layer_outputs[0]
            hidden_states_0 = layer_outputs[-2]
            # new_mem_state is always the last element — store for next step
            new_mem_state: Optional[MemoryState] = layer_outputs[-1]
            if new_mem_state is not None:
                self.mem_states[idx] = new_mem_state   # ← Titans state persisted

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return DecoderOutput(
            last_hidden_state=hidden_states,
            last_hidden_state_0=hidden_states_0,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position,
                            past_key_values, output_attentions):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens  = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens, is_training=self.training,
            ):
                return None

        dtype, device    = input_tensor.dtype, input_tensor.device
        min_dtype        = torch.finfo(dtype).min
        sequence_length  = input_tensor.shape[1]

        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask  = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask  = causal_mask.clone()
                mask_length  = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


@dataclass
class ModelOutput(Seq2SeqModelOutput):
    last_hidden_state_0 : Optional[torch.Tensor] = None

# WhisperModel — unchanged
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
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length), mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length, attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0
        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size), mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length, min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0
        return input_features

    def forward(
        self,
        input_features=None, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, head_mask=None, decoder_head_mask=None,
        cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None,
        decoder_inputs_embeds=None, decoder_position_ids=None,
        use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, cache_position=None,
    ):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.encoder(
                input_features, head_mask=head_mask,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_hidden_state_0=decoder_outputs.last_hidden_state_0,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@dataclass
class Seq2SeqLMOutput(Seq2SeqLMOutput):
    bce_logits: Optional[torch.Tensor] = None

# WhisperForConditionalGeneration
class WhisperForConditionalGeneration(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix  = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model    = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):             return self.model.get_encoder()
    def get_decoder(self):             return self.model.get_decoder()
    def get_output_embeddings(self):   return self.proj_out
    def set_output_embeddings(self, e): self.proj_out = e
    def get_input_embeddings(self):    return self.model.get_input_embeddings()
    def freeze_encoder(self):          self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features=None, attention_mask=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
        encoder_outputs=None, past_key_values=None,
        decoder_inputs_embeds=None, decoder_position_ids=None,
        labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None, cache_position=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Reset Titans memory at the start of every new utterance.
        # past_key_values is None iff this is the first forward step (training
        # or the very first generate() step).  Memory accumulates within one
        # utterance and resets between utterances.
        if past_key_values is None:
            self.model.decoder.reset_mem_states()

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

        lm_logits = self.proj_out(outputs[0])
        lz_logits = self.proj_out(outputs[1])
        loss      = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels   = labels.to(lm_logits.device)
            loss     = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss, logits=lm_logits,
            bce_logits = lz_logits,
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
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1],
                device=decoder_input_ids.device,
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1]:]

        return {
            "encoder_outputs":         encoder_outputs,
            "past_key_values":         past_key_values,
            "decoder_input_ids":       decoder_input_ids,
            "use_cache":               use_cache,
            "decoder_attention_mask":  decoder_attention_mask,
            "decoder_position_ids":    decoder_position_ids,
            "cache_position":          cache_position,
        }



# Remaining unchanged model classes
class WhisperDecoderWrapper(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.decoder = WhisperDecoder(config)

    def get_input_embeddings(self):    return self.decoder.embed_tokens
    def set_input_embeddings(self, v): self.decoder.embed_tokens = v
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

    def forward(
        self, input_ids=None, attention_mask=None, encoder_outputs=None,
        head_mask=None, cross_attn_head_mask=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None, cache_position=None,
    ):
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if isinstance(encoder_outputs, (BaseModelOutput, tuple, list)):
            encoder_outputs = encoder_outputs[0]

        outputs = self.model.decoder(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs,
            head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            cache_position=cache_position,
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, use_cache=None,
        encoder_outputs=None, attention_mask=None, cache_position=None, **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, (Cache, EncoderDecoderCache)):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_ids.shape[1]:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "input_ids":       input_ids,
            "use_cache":       use_cache,
            "attention_mask":  attention_mask,
            "cache_position":  cache_position,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


