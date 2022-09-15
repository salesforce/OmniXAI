#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Grad-CAM methods for vision tasks.
"""
from typing import Callable
from omnixai.utils.misc import is_tf_available, is_torch_available
from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image


class GradCAM(ExplainerBase):
    """
    The Grad-CAM method for generating visual explanations.
    If using this explainer, please cite `Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization, Selvaraju et al., https://arxiv.org/abs/1610.02391`.
    """

    explanation_type = "local"
    alias = ["gradcam", "grad-cam"]

    def __init__(self, model, target_layer, preprocess_function: Callable, mode: str = "classification", **kwargs):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param target_layer: The target layer for explanation, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        _class = None
        if is_torch_available():
            import torch.nn as nn
            from .pytorch.gradcam import GradCAM

            if isinstance(model, nn.Module):
                _class = GradCAM

        if _class is None and is_tf_available():
            import tensorflow as tf
            from .tf.gradcam import GradCAM

            if isinstance(model, tf.keras.Model):
                _class = GradCAM

        if _class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.explainer = _class(
            model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode
        )

    def explain(self, X: Image, y=None, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        return self.explainer.explain(X=X, y=y, **kwargs)


class GradCAMPlus(ExplainerBase):
    """
    The Grad-CAM++ method for generating visual explanations.
    If using this explainer, please cite `Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks, Chattopadhyay et al., https://arxiv.org/pdf/1710.11063`.
    """

    explanation_type = "local"
    alias = ["gradcam++", "grad-cam++"]

    def __init__(self, model, target_layer, preprocess_function: Callable, mode: str = "classification", **kwargs):
        """
        :param model: The model whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param target_layer: The target layer for explanation, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        _class = None
        if is_torch_available():
            import torch.nn as nn
            from .pytorch.gradcam import GradCAMPlus

            if isinstance(model, nn.Module):
                _class = GradCAMPlus

        if _class is None and is_tf_available():
            import tensorflow as tf
            from .tf.gradcam import GradCAMPlus

            if isinstance(model, tf.keras.Model):
                _class = GradCAMPlus

        if _class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.explainer = _class(
            model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode
        )

    def explain(self, X: Image, y=None, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        return self.explainer.explain(X=X, y=y, **kwargs)


class LayerCAM(ExplainerBase):
    """
    The Layer-CAM method for generating visual explanations.
    If using this explainer, please cite `LayerCAM: Exploring Hierarchical Class Activation Maps for Localization,
    Jiang et al., http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf`.
    """

    explanation_type = "local"
    alias = ["layercam", "layer-cam"]

    def __init__(self, model, target_layer, preprocess_function: Callable, mode: str = "classification", **kwargs):
        """
        :param model: The model whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param target_layer: The target layer for explanation, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        _class = None
        if is_torch_available():
            import torch.nn as nn
            from .pytorch.gradcam import LayerCAM

            if isinstance(model, nn.Module):
                _class = LayerCAM

        if _class is None and is_tf_available():
            import tensorflow as tf
            from .tf.gradcam import LayerCAM

            if isinstance(model, tf.keras.Model):
                _class = LayerCAM

        if _class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.explainer = _class(
            model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode
        )

    def explain(self, X: Image, y=None, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        return self.explainer.explain(X=X, y=y, **kwargs)
