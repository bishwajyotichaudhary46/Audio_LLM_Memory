# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Generic utilities
"""
import json
from collections import OrderedDict, UserDict
from collections.abc import Iterable,  MutableMapping, Callable
from dataclasses import  fields, is_dataclass
from functools import partial
from functools import lru_cache
import torch
from typing import Any
import importlib.metadata
import importlib.util
PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()

import numpy as np

#from ..utils import logging

_CAN_RECORD_REGISTRY = {}


#logger = logging.get_logger(__name__)


_is_torch_available = False


def to_numpy(obj):
    """
    Convert a PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


def safe_load_json_file(json_file: str):
    "A helper to load safe config files and raise a proper error message if it wasn't serialized correctly"
    try:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
    except json.JSONDecodeError:
        raise OSError(f"It looks like the config file at '{json_file}' is not a valid JSON file.")
    return config_dict

def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'numpy."):
        return "np"
    elif representation.startswith("<class 'mlx."):
        return "mlx"
def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return isinstance(x, np.ndarray)
def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return _is_torch_available and isinstance(x, torch.Tensor)
def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}

def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)
def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
        #logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

@lru_cache
def is_mlx_available() -> bool:
    return _is_package_available("mlx")

_is_mlx_available = False
if is_mlx_available():
    _is_mlx_available = True


def is_mlx_array(x):
    """
    Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed.
    """
    return False if not _is_mlx_available else _is_mlx(x)

def is_torch_fx_proxy(x):
    try:
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    except Exception:
        return False
    
def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `np.ndarray` or `mlx.array` in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    if is_torch_fx_proxy(x):
        return True

    return False

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if _is_torch_available:
            from torch.utils._pytree import register_pytree_node

            register_pytree_node(
                cls,
                _model_output_flatten,
                partial(_model_output_unflatten, output_type=cls),
                serialized_type_name=f"{cls.__module__}.{cls.__name__}",
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                # reset first field to None
                setattr(self, class_fields[0].name, None)
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


if _is_torch_available:
    import torch.utils._pytree as _torch_pytree

    def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], "_torch_pytree.Context"]:
        return list(output.values()), list(output.keys())

    def _model_output_unflatten(
        values: Iterable[Any],
        context: "_torch_pytree.Context",
        output_type=None,
    ) -> ModelOutput:
        return output_type(**dict(zip(context, values)))

    _torch_pytree.register_pytree_node(
        ModelOutput,
        _model_output_flatten,
        partial(_model_output_unflatten, output_type=ModelOutput),
        serialized_type_name=f"{ModelOutput.__module__}.{ModelOutput.__name__}",
    )

def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict
class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())
