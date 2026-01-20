# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
import itertools
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
#from transformers.configuration_utils import PreTrainedConfig
from MemoryModule.utils.configuration_utils import PreTrainedConfig
from transformers.utils import is_torch_xpu_available, logging
#from transformers.utils.generic import GeneralInterface
from transformers.utils.import_utils import is_torch_flex_attn_available, is_torch_greater_or_equal
from MemoryModule.utils.import_utils import is_tracing
from MemoryModule.utils.generic import GeneralInterface

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


logger = logging.get_logger(__name__)

def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx

def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int
) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask

def _can_skip_causal_mask_xpu(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    local_attention_size: Optional[int],
) -> bool:
    """
    XPU-specific logic for determining if we can skip causal mask creation.

    For XPU devices, we have special handling:
    - Single query tokens (query_length == 1) use the same logic as CUDA
    - Multi-query tokens can skip if padding_mask is provided and correctly structured
      The mask must have all True values in the query window and all False after
    """

    if is_tracing(padding_mask):
        return False

    # Check local attention constraint (same as CUDA)
    if local_attention_size is not None and kv_length >= local_attention_size:
        return False

    if padding_mask is None:
        # Without padding mask, can skip if single query token or full causal attention
        return query_length == 1 or kv_length == query_length

    # XPU allows skipping under additional conditions when padding_mask is provided
    if query_length == 1:
        # Single query token: skip only if no padding tokens present
        return padding_mask.all()

    # XPU-specific: check if query window is all True and rest is all False
    # This allows XPU to optimize the 1st token in static cache
    return padding_mask[:, :query_length].all() and not padding_mask[:, query_length:].any()

def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    if _is_torch_xpu_available:
        # XPU devices have special handling for mask skipping:
        # - Single query tokens use the same logic as CUDA
        # - Multi-query tokens can skip if padding_mask is provided and correctly structured
        #   (all True in query window, all False after)
        return _can_skip_causal_mask_xpu(padding_mask, query_length, kv_length, local_attention_size)
    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing(padding_mask)
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or kv_length == query_length)
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (padding_mask is None or padding_mask.all())
    ):
        return True

    return False
def _ignore_bidirectional_mask_sdpa(padding_mask: Optional[torch.Tensor]) -> bool:
    """
    Detects whether the bidirectional mask can be ignored in case PyTorch's SDPA is used, i.e. when there is full
    attention with no padding.
    """
    # When using `torch.export` or `torch.onnx.dynamo_export`, we need to avoid to check the contents of the mask;
    # otherwise, we will encounter dynamic control flows
    if not is_tracing(padding_mask) and (padding_mask is None or padding_mask.all()):
        return True

    return False

def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask

def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask
def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    """
    Used to broadcast our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Allows the usage of any index-based mask function without relying on vmap.

    NOTE: This is limited to index based functions only and is not guaranteed to work otherwise.

    Reference:
        - https://github.com/huggingface/optimum-onnx/blob/c123e8f4fab61b54a8e0e31ce74462bcacca576e/optimum/exporters/onnx/model_patcher.py#L362-L365
    """
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices
def _vmap_expansion_sdpa(mask_function: Callable) -> Callable:
    """
    Used to vmap our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).
    """
    # We vmap the function over all 4 dimensions, broadcasting [b_idx, h_idx, q_idx, kv_idx]
    dimensions = [(None, None, None, 0), (None, None, 0, None), (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function

def sdpa_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
        use_vmap (`bool`, optional):
            Whether to use `vmap` during the mask construction or not. Allows powerful custom patterns that may not be
            index-based (for the cost of speed performance). By default `False`.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5)
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [ True,  True,  True,  True, False],
                  [ True,  True,  True,  True,  True]]]])
    ```

    ## Creating a sliding window mask:

    To create the following sliding window mask (`sliding_window=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=sliding_window_causal_mask_function(3))
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [False,  True,  True,  True, False],
                  [False, False,  True,  True,  True]]]])
    ```

    ## Creating a chunked attention mask

    To create the following chunked attention mask (`chunk_size=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ⬚ ⬚ ■ ⬚
        4 ⬚ ⬚ ⬚ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=chunked_causal_mask_function(3, torch.zeros(1, dtype=int)))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    q_length = cache_position.shape[0]

    # Potentially pad the 2D mask
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask
    #   1. Causal masks can rely on the `is_causal` argument
    #   2. Bidirectional do not need any further processing (no bias)
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset

    # Actual mask creation
    # Option 1: Fast non-vmap mask creation (default)
    if not use_vmap:
        # Apply mask function element-wise through broadcasting
        attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, cache_position, kv_arange))
        # Expand the mask to match batch size and query length if they weren't used in the mask function
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)

    # Option 2: Vmap mask creation (torch>=2.6 and custom patterns)
    elif _is_torch_greater_or_equal_than_2_6:
        # This creates the 4D mask easily. Note that we need this context manager as vmap cannot handle slicing a tensor from
        # scalar tensor (it internally calls `.item()` which vmap does not allow, but this context works around it
        # We don't need to add an offset to the mask_function either, as we vmap directly the correct indices for k and kv indices
        with TransformGetItemToIndex():
            attention_mask = _vmap_expansion_sdpa(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    # Option 3: Error out since it indicates that the user did something custom, which they shouldn't have (torch<2.6)
    else:
        raise ValueError(
            "The vmap functionality for mask creation is only supported from torch>=2.6. "
            "Please update your torch version or use `use_vmap=False` with index-based masks."
        )

    # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        attention_mask = attention_mask | torch.all(~attention_mask, dim=-1, keepdim=True)

    return attention_mask

def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    allow_is_bidirectional_skip: bool = False,
    use_vmap: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
        use_vmap (`bool`, optional):
            Whether to use `vmap` during the mask construction or not. Allows powerful custom patterns that may not be
            index-based (for the cost of speed performance). By default `False`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    _ = kwargs.pop("allow_torch_fix", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        allow_torch_fix=False,
        use_vmap=use_vmap,
        **kwargs,
    )
    # only bidirectional masks can be skipped, otherwise we convert bool -> float
    if mask is not None:
        min_dtype = torch.finfo(dtype).min
        # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
        mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask

def flash_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Create the attention mask necessary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    We just slice it in case of sliding window.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        # We only return an actual mask if there is at least 1 padding token, otherwise we return `None` and use `is_causal` in FA2
        # (note that the attention_mask is a boolean dtype here)
        if attention_mask.all():
            attention_mask = None

    return attention_mask


class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flex_attention": flash_attention_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()
def find_packed_sequence_indices(position_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].

        If the there is only one sequence in each batch item (and we don't compile), then we return `None` indicating
        no packed sequences. This is the same as [[0, 0, 0, 0, 0, 0]] for the example above.
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    # Sadly this is a dynamic control flow, so we cannot enable this check on anything compile related
    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None

    return packed_sequence_mask

def _preprocess_mask_arguments(
    config: PreTrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, BlockMask]], # type: ignore
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, BlockMask]], int, int]: # type: ignore
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = input_embeds.shape[1], 0
        # if hasattr(past_key_values, "get_mask_sizes"):
        #     kv_length, kv_offset = past_key_values.get_mask_sizes(
        #         cache_position, layer_idx
        #     )
        # else:
        #     # EncoderDecoderCache (Whisper)
        #     kv_length = past_key_values.get_seq_length()
        #     print("kv length", kv_length)
        #     kv_offset = 0
    #print("past_key_values", past_key_values)
    # if past_key_values is not None:
    #     kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # # Otherwise, we infer based on our input
    else:
        # 1. Rely on input directly
        if attention_mask is None:
            kv_length, kv_offset = input_embeds.shape[1], 0
        # 2. Rely on the mask instead - needed for special cases like prefix tuning in PEFT
        #
        # This is a very unique and special case where an encoder utilizes a cache and expects its length
        # to be accounted for (usually, they should never use a cache). In general, the mask should always
        # match with the input sizes nonetheless (i.e. it does not affect others).
        # Conclusion: "prefix tuning is evil"
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset
def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask
def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask

def create_causal_mask(
    config: PreTrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Defaulting to using non-vmap based mask creations except when detecting
    # users passing custom mask functions (as we cannot guarantee that they
    # are properly index-based as required by our implementation).
    use_vmap = False

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    if _is_torch_xpu_available:
        # Do not allow skip if we are compiling for decoding, but for prefill, we still allow skip to optimization the perf of 1st token generation
        allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and cache_position.shape[0] == 1)
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    # If we detected packing format
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
    )
    return causal_mask
