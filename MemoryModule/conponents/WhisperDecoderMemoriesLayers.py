import torch
import torch.nn as nn
import transformers
from typing import Any, Optional
from transformers.models.whisper.modeling_whisper import WhisperDecoder
from MemoryModule.conponents.LinearAttention import LinearAttentionMem
from MemoryModule.utils.common import bias_term_adjust
from MemoryModule.conponents.WhisperDecoderMemoryLinerLayers import WhisperDecoderMemoryLayer


class WhisperDecoderWithSharedMemory(WhisperDecoder):
    def __init__(
        self,
        config,
        num_memories: int = 8,
        bias_term_adjust = bias_term_adjust,
        LinearAttentionMem = LinearAttentionMem,
    ):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                WhisperDecoderMemoryLayer(
                    config,
                    num_memories=num_memories,
                    bias_term_adjust=bias_term_adjust,
                    LinearAttentionMem=LinearAttentionMem,
                    is_memory_layer=(i == 2),
                )
                for i in range(config.decoder_layers)
            ]
        )

        # Persistent flag for first forward pass
        self.register_buffer("first_pass_flag", torch.tensor(True))

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[transformers.cache_utils.EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Any:
        hidden_states = inputs_embeds
        is_first = self.first_pass_flag.item()

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None

        shared_M = None
        shared_bias = None

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                M=shared_M,
                bias=shared_bias,
                is_first_pass=is_first,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]


            if idx == 2:
                shared_M = layer_outputs[-2]
                shared_bias = layer_outputs[-1]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)
            if use_cache:
                next_cache = next_cache + (layer_outputs[3],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if is_first:
            self.first_pass_flag.fill_(False)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attentions,
                "cross_attentions": all_cross_attentions,
                "memory_states": (shared_M, shared_bias),
            }
        else:
            return hidden_states, next_cache, all_hidden_states, all_self_attentions, all_cross_attentions, shared_M, shared_bias