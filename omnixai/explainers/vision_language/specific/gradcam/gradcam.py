#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Grad-CAM methods for vision-language models.
"""
from typing import Callable
from omnixai.utils.misc import is_tf_available, is_torch_available
from omnixai.explainers.base import ExplainerBase
from omnixai.data.multi_inputs import MultiInputs


class GradCAM(ExplainerBase):
    """
    The Grad-CAM method for vision-language models.
    If using this explainer, please cite `Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization, Selvaraju et al., https://arxiv.org/abs/1610.02391`.
    """

    explanation_type = "local"
    alias = ["gradcam", "grad-cam"]

    def __init__(
            self,
            model,
            target_layer,
            preprocess_function: Callable,
            tokenizer: Callable,
            loss_function: Callable,
            patch_shape: tuple = (24, 24),
            **kwargs
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param target_layer: The target layer for explanation, which can be
            `tf.keras.layers.Layer` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param tokenizer: The tokenizer for processing text inputs.
        :param loss_function: The loss function used to compute gradients w.r.t the target layer.
        :param patch_shape: The patch shape, e.g., (24, 24), in the vision transformer model.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")
        assert target_layer is not None, "The target layer cannot be None."
        assert preprocess_function is not None, "`preprocess_function` cannot be None."
        assert tokenizer is not None, "The tokenizer cannot be None."
        assert loss_function is not None, "The loss_function cannot be None."

        _class = None
        if is_torch_available():
            import torch.nn as nn
            from .pytorch.gradcam import GradCAM

            if isinstance(model, nn.Module):
                _class = GradCAM

        if _class is None and is_tf_available():
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                raise ValueError("The tensorflow model is not supported yet.")

        if _class is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.explainer = _class(
            model=model,
            target_layer=target_layer,
            preprocess_function=preprocess_function,
            tokenizer=tokenizer,
            loss_function=loss_function,
            patch_shape=patch_shape,
            **kwargs
        )

    def explain(self, X: MultiInputs, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances, e.g., `X.image` contains the input images
            and `X.text` contains the input texts.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        return self.explainer.explain(X=X, **kwargs)
