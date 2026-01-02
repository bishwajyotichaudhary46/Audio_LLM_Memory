from typing import Optional, Any
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperAttention
from MemoryModule.conponents.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel,WhisperDecoder, WhisperGenerationMixin,_compute_mask_indices, WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperAttention, WhisperModel,WhisperDecoderLayer, shift_tokens_right, WhisperPositionalEmbedding,WhisperEncoder,CausalLMOutputWithCrossAttentions


from torch.nn import CrossEntropyLoss
from MemoryModule.utils import auto_docsting
from typing import Union, Optional
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.cache_utils import Cache
import math

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput, BaseModelOutput
from transformers.utils.doc import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from MemoryModule.utils.mask_utils import create_causal_mask
from MemoryModule.conponents.Mom import MOM
from MemoryModule.conponents.MHARouting import MHARouting
from MemoryModule.conponents.LinearAttention import LinearAttentionMem
from MemoryModule.utils.common import bias_term_adjust

class DecoderLayer(WhisperDecoderLayer):

    def __init__(self, config: WhisperConfig, 
        layer_idx: Optional[int] = None, 
        num_memories: int = 8,
        bias_term_adjust: Any = None,
        LinearAttentionMem: Any = None,
        is_memory_layer: bool = False,):

        super().__init__(config, layer_idx)
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
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,

    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_values (`Cache`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # # Initialize M/bias ONLY on first pass in memory layer
        # if self.is_memory_layer and M is None:
        #     B = hidden_states.shape[0]
        #     M = torch.zeros((
        #         B,
        #         self.num_memories * self.config.decoder_attention_heads,
        #         self.config.d_model,self.config.d_model),
        #         device=hidden_states.device,
        #         dtype=hidden_states.dtype,
        #     )
        #     bias = torch.zeros((B,
        #         self.num_memories),
        #         device=hidden_states.device,
        #         dtype=hidden_states.dtype,
        #     )
            # print("Memory Iniitalized ", M.shape, bias.shape)


        # Self Attention
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # Memory fusion
        if self.is_memory_layer:
            # print("attention route ", hidden_states.shape)

            # Routing gate
            atten_gate = self.attn_route(hidden_states)  # [B, T, 1]
            # print("attn gate:", atten_gate.shape)

            # Update via MOM
            mem_out = self.context_mom(hidden_states)
            mem_out = self.memory_norm(mem_out)
            hidden_states = hidden_states + atten_gate * mem_out
            # print("Memory Updated", M.shape)
        else:
            # Read-only
            hidden_states = hidden_states

        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
    

# class DecoderLayer(WhisperDecoderLayer):

#     def __init__(self, config: WhisperConfig, layer_idx: Optional[int] = None):
#         super().__init__(config, layer_idx)
        
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[EncoderDecoderCache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = True,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`): attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             encoder_hidden_states (`torch.FloatTensor`):
#                 cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
#             encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             past_key_values (`Cache`): cached past key and value projection states
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """
#         residual = hidden_states
#         hidden_states = self.self_attn_layer_norm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             cache_position=cache_position,
#         )
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#         hidden_states = residual + hidden_states

#         # Cross-Attention Block
#         cross_attn_weights = None
#         if encoder_hidden_states is not None:
#             residual = hidden_states
#             hidden_states = self.encoder_attn_layer_norm(hidden_states)
#             hidden_states, cross_attn_weights = self.encoder_attn(
#                 hidden_states=hidden_states,
#                 key_value_states=encoder_hidden_states,
#                 attention_mask=encoder_attention_mask,
#                 past_key_values=past_key_values,
#                 output_attentions=output_attentions,
#             )
#             hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#             hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.final_layer_norm(hidden_states)
#         hidden_states = self.activation_fn(self.fc1(hidden_states))
#         hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
#         hidden_states = self.fc2(hidden_states)
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights, cross_attn_weights)

#         return outputs
    

class Decoder(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig, 
        num_memories: int = 8,
        bias_term_adjust=bias_term_adjust,
        LinearAttentionMem=LinearAttentionMem,):
        super().__init__(config)
       
        self.layers = nn.ModuleList(
            [DecoderLayer(config, 
                          layer_idx, 
                          num_memories=num_memories, 
                          bias_term_adjust= bias_term_adjust, 
                          LinearAttentionMem=LinearAttentionMem,
                          is_memory_layer=(layer_idx == 2),) for layer_idx in range(config.decoder_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
            )
        else:
            positions = self.embed_positions(
                inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
            )

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                # )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values if use_cache else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class Model(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.decoder = Decoder(config)


    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
        # r"""
        # decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        #     Indices of decoder input sequence tokens in the vocabulary.

        #     Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
        #     [`PreTrainedTokenizer.__call__`] for details.

        #     [What are decoder input IDs?](../glossary#decoder-input-ids)

        #     Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
        #     `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
        #     `past_key_values`).
        # decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        #     Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
        #     be used by default.

        #     If you want to change padding behavior, you should read
        #     [`modeling_whisper._prepare_decoder_attention_mask`] and modify to your needs. See diagram 1 in [the BART
        #     paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
        # decoder_position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        #     Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
        #     config.n_positions - 1]`.

        #     [What are position IDs?](../glossary#position-ids)

        # Example:
        #  ```python
        #  >>> import torch
        #  >>> from transformers import AutoFeatureExtractor, WhisperModel
        #  >>> from datasets import load_dataset

        #  >>> model = WhisperModel.from_pretrained("openai/whisper-base")
        #  >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        #  >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        #  >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        #  >>> input_features = inputs.input_features
        #  >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        #  >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        #  >>> list(last_hidden_state.shape)
        #  [1, 2, 512]
        #  ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
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


class ConditionalGeneration(WhisperForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = {"proj_out.weight": "model.decoder.embed_tokens.weight"}

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = Model(config)
        # self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.max_target_positions = config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read
            [`modeling_whisper._prepare_decoder_attention_mask`] and modify to your needs. See diagram 1 in [the BART
            paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
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

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )