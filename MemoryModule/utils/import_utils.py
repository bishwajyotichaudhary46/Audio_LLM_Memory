
import importlib.metadata

from transformers.utils.import_utils import is_torchdynamo_compiling,  is_torch_fx_proxy




PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

def is_jax_jitting(x):
    """returns True if we are inside of `jax.jit` context, False otherwise.

    When a torch model is being compiled with `jax.jit` using torchax,
    the tensor that goes through the model would be an instance of
    `torchax.tensor.Tensor`, which is a tensor subclass. This tensor has
    a `jax` method to return the inner Jax array
    (https://github.com/google/torchax/blob/13ce870a1d9adb2430333c27bb623469e3aea34e/torchax/tensor.py#L134).
    Here we use ducktyping to detect if the inner jax array is a jax Tracer
    then we are in tracing context. (See more at: https://github.com/jax-ml/jax/discussions/9241)

    Args:
      x: torch.Tensor

    Returns:
      bool: whether we are inside of jax jit tracing.
    """

    if not hasattr(x, "jax"):
        return False
    try:
        import jax

        return isinstance(x.jax(), jax.core.Tracer)
    except Exception:
        return False

def is_jit_tracing() -> bool:
    try:
        import torch

        return torch.jit.is_tracing()
    except Exception:
        return False


def is_cuda_stream_capturing() -> bool:
    try:
        import torch

        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False

def is_tracing(tensor=None) -> bool:
    """Checks whether we are tracing a graph with dynamo (compile or export), torch.jit, torch.fx, jax.jit (with torchax) or
    CUDA stream capturing"""
    # Note that `is_torchdynamo_compiling` checks both compiling and exporting (the export check is stricter and
    # only checks export)
    _is_tracing = is_torchdynamo_compiling() or is_jit_tracing() or is_cuda_stream_capturing()
    if tensor is not None:
        _is_tracing |= is_torch_fx_proxy(tensor)
        _is_tracing |= is_jax_jitting(tensor)
    return _is_tracing