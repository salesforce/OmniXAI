#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import inspect
import importlib.util
import numpy as np
from abc import ABCMeta
from functools import wraps
from packaging import version
from sklearn.base import BaseEstimator

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

get_pkg_version = importlib_metadata.version


class AutodocABCMeta(ABCMeta):
    """
    Metaclass used to ensure that inherited members of an abstract base class
    also inherit docstrings for inherited methods.
    """

    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if member.__doc__ is None:
                for base in bases[::-1]:
                    attr = getattr(base, name, None)
                    if attr is not None:
                        member.__doc__ = attr.__doc__
                        break
        return cls


def initializer(func):
    """
    Decorator for the __init__ method.
    Automatically assigns the parameters.
    """
    argspec = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(argspec.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        for name, default in zip(reversed(argspec.args), reversed(argspec.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)
        func(self, *args, **kargs)

    return wrapper


class ProgressBar:
    """
    The progress bar for displaying training or testing progress.
    """

    def __init__(self, total: int, length: int = 40, decimals: int = 1, fill: str = "█"):
        """
        :param total: total iterations
        :param length: character length of bar
        :param decimals: positive number of decimals in percent complete
        :param fill: bar fill character
        """
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill

    def print(self, iteration, prefix, suffix, end=""):
        """
        :param iteration: current iteration
        :param prefix: prefix string
        :param suffix: suffix string
        :param end: end character (e.g. ``"\\r"``, ``"\\r\\n"``)
        """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        fill_len = self.length * iteration // self.total
        bar = self.fill * fill_len + "-" * (self.length - fill_len)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=end)
        if iteration >= self.total:
            print()


def is_torch_available():
    """
    Checks if pytorch is available.
    """
    if importlib.util.find_spec("torch") is not None:
        _torch_version = importlib_metadata.version("torch")
        if version.parse(_torch_version) < version.parse("1.3"):
            raise EnvironmentError(f"Torch found but with version {_torch_version}. " f"The minimum version is 1.3")
        return True
    else:
        return False


def is_tf_available():
    """
    Checks if tensorflow 2.0 is available.
    """
    candidates = (
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
        "tensorflow-rocm",
        "tensorflow-macos",
    )
    _tf_version = None
    for pkg in candidates:
        try:
            _tf_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    if _tf_version is not None:
        if version.parse(_tf_version) < version.parse("2"):
            raise EnvironmentError(f"Tensorflow found but with version {_tf_version}. " f"The minimum version is 2.0")
        return True
    else:
        return False


def is_transformers_available():
    """
    Checks if the `transformers` library is installed.
    """
    if importlib.util.find_spec("transformers") is not None:
        _version = importlib_metadata.version("transformers")
        if version.parse(_version) < version.parse("4.0"):
            raise EnvironmentError(f"Transformers found but with version {_version}. " f"The minimum version is 4.0")
        return True
    else:
        return False


def is_nltk_available():
    """
        Checks if the `nltk` library is installed.
        """
    if importlib.util.find_spec("nltk") is not None:
        return True
    else:
        return False


def tensor_to_numpy(x):
    """
    Converts a tensor in pytorch or tensorflow into a numpy array.
    """
    if isinstance(x, (int, np.int32, np.int64, float, np.float32, np.float64)):
        return np.array([x])
    if not isinstance(x, np.ndarray):
        try:
            x = x.detach().cpu().numpy()
        except:
            x = x.numpy()
    return x


def set_random_seed(seed=0):
    """
    Set random seeds.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        import torch

        torch.manual_seed(seed)
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def build_predict_function(model, preprocess, postprocess, mode):
    # A scikit-learn model
    if isinstance(model, BaseEstimator):
        # A model derived from sklearn.base.BaseEstimator
        predict_func = model.predict_proba if mode == "classification" else model.predict
    else:
        # A torch model, tensorflow model or general function
        if is_torch_available():
            import torch.nn as nn
            if isinstance(model, nn.Module):
                model.eval()
        predict_func = model
    # Pre-processing and post-processing
    preprocess = preprocess if preprocess is not None else lambda x: x
    postprocess = postprocess if postprocess is not None else lambda x: x

    # The predict function
    def _predict(x):
        inputs = preprocess(x)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return tensor_to_numpy(postprocess(predict_func(*inputs)))

    return _predict
