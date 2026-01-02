import torch
import torch.nn as nn
import transformers
from typing import Any, Optional
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoder,
    BaseModelOutputWithPastAndCrossAttentions,
)

from MemoryModule.conponents.LinearAttention import LinearAttentionMem
from MemoryModule.utils.common import bias_term_adjust
from MemoryModule.conponents.LinearMemoryLayer import WhisperMemoryLayer
from transformers.modeling_outputs import Seq2SeqLMOutput




class WhisperDecoderLayers(WhisperDecoder):
    def __init__(
        self,
        config,
        num_memories: int = 8,
        bias_term_adjust=bias_term_adjust,
        LinearAttentionMem=LinearAttentionMem,
    ):
        super().__init__(config)

        # Replace standard Whisper layers with memory layers
        self.layers = nn.ModuleList(
            [
                WhisperMemoryLayer(
                    config,
                    num_memories=num_memories,
                    bias_term_adjust=bias_term_adjust,
                    LinearAttentionMem=LinearAttentionMem,
                    is_memory_layer=(i == 2),
                )
                for i in range(config.decoder_layers)
            ]
        )

        # Track first forward pass (persistent across steps)
        self.register_buffer("first_pass_flag", torch.tensor(True))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[transformers.cache_utils.EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Any:

        if input_ids is None:
            raise ValueError("WhisperDecoderLayers requires `input_ids`")

        # ------------------------------------------------------------------
        # Embeddings + positional embeddings (Whisper-correct)
        # ------------------------------------------------------------------
        print("Decoder layer")
        hidden_states = input_ids
        batch_size, seq_len  = hidden_states.shape

        hidden_states = self.embed_tokens(hidden_states)
        print("Embeding")

        if cache_position is not None:
            position_ids = cache_position
        else:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)

        pos_embeds = self.embed_positions(position_ids)
        hidden_states = hidden_states + pos_embeds

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        
        # ------------------------------------------------------------------
        # Setup outputs
        # ------------------------------------------------------------------
        is_first = bool(self.first_pass_flag.item())

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        # ------------------------------------------------------------------
        # Decoder layers
        # ------------------------------------------------------------------
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                M=M,
                bias=bias,
                is_first_pass=is_first,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache = use_cache,
                cache_position=cache_position,
            )

            # ---- unpack ----

            hidden_states = layer_outputs[0][0]
            

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attns += (layer_outputs[2],)

            # Memory update (only from memory layer)
            if decoder_layer.is_memory_layer:
                M = layer_outputs[-2]
                bias = layer_outputs[-1]

        # ------------------------------------------------------------------
        # Final layer norm
        # ------------------------------------------------------------------
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if is_first:
            self.first_pass_flag.fill_(False)

        # ------------------------------------------------------------------
        # Return
        # ------------------------------------------------------------------
        if not return_dict:
            return (
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_cross_attns,
                M,
                bias,)
            
        
        # return BaseModelOutputWithPastAndCrossAttentions(
        #         last_hidden_state=hidden_states,
        #         past_key_values=next_cache,
        #         hidden_states=all_hidden_states,
        #         attentions=all_self_attns,
        #         cross_attentions=all_cross_attns)
