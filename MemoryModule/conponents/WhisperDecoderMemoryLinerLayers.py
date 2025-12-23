from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import transformers
from MemoryModule.conponents.Mom import MOM
from MemoryModule.conponents.MHARouting import MHARouting

class WhisperDecoderMemoryLayer(WhisperDecoderLayer):
    def __init__(
        self,
        config,
        num_memories: int = 8,
        bias_term_adjust: Any = None,
        LinearAttentionMem: Any = None,
        is_memory_layer: bool = False,
    ):
        super().__init__(config)
        self.is_memory_layer = is_memory_layer
        self.num_memories = num_memories
        self.config = config

        self.attn_route = MHARouting(config.d_model)
        self.context_mom = MOM(
            n_text_state=config.d_model,
            n_heads=config.decoder_attention_heads,
            num_memories=num_memories,
            bias_term_adjust=bias_term_adjust,
            LinearAttentionMem=LinearAttentionMem,
        )
        self.memory_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        M: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_pass: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[transformers.cache_utils.EncoderDecoderCache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[Any, ...]:
        hidden_states = hidden_states.unsqueeze(0)
        residual = hidden_states
        B = hidden_states.size(0)

        # Initialize M/bias ONLY on first pass in memory layer (layer 0)
        if self.is_memory_layer and is_first_pass and M is None:
            M = torch.zeros((
                B,
                self.num_memories * self.config.decoder_attention_heads,
                self.config.d_model,self.config.d_model),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            bias = torch.zeros((B,
                self.num_memories),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        
        #print("hidden state ", hidden_states.shape)
        # Self-attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        #print("hidden state ", hidden_states.shape)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        #print("hidden state ", hidden_states.shape)

        # Routing gate
        atten_gate = self.attn_route(hidden_states)  # [B, T, 1]

        # Memory fusion
        if self.is_memory_layer:
            # Update via MOM
            bias, M, mem_out = self.context_mom(hidden_states, M, bias)
            M = self.memory_norm(M)
            hidden_states = hidden_states + atten_gate * mem_out
        else:
            # Read-only
            hidden_states = hidden_states
        
        # Cross-attention
        cross_attn_weights = None
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_values.cross_attention if past_key_values else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        # Feed Forward Network
        residual = hidden_states
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = self.final_layer_norm(residual + hidden_states)

        # Outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)

        return outputs + (M, bias)