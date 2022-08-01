#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from abc import abstractmethod

from omnixai.utils.misc import AutodocABCMeta
from omnixai.data.multi_inputs import MultiInputs
from omnixai.utils.misc import is_torch_available

if not is_torch_available():
    raise EnvironmentError("Torch cannot be found.")
else:
    import torch
    import torch.nn as nn


class Base(metaclass=AutodocABCMeta):

    def __init__(
            self,
            model: nn.Module,
            target_layer: nn.Module,
            preprocess_function: Callable,
            tokenizer: Callable,
            loss_function: Callable,
            patch_shape: tuple,
            **kwargs
    ):
        super().__init__()
        assert isinstance(
            model, nn.Module
        ), f"`model` should be an instance of torch.nn.Module instead of {type(model)}"
        assert isinstance(
            target_layer, nn.Module
        ), f"`target_layer` should be an instance of torch.nn.Module instead of {type(target_layer)}"

        self.model = model.eval()
        self.target_layers = [target_layer]
        self.preprocess = preprocess_function
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.patch_shape = tuple(patch_shape)
        self.kwargs = kwargs

        # Hooks for storing layer activations and gradients
        self.hooks = []
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(self._activation_hook))
            self.hooks.append(layer.register_backward_hook(self._gradient_hook))

    def _unregister_hooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def __del__(self):
        self._unregister_hooks()

    def _activation_hook(self, module, inputs, outputs):
        self.activations.append(outputs.detach().cpu())

    def _gradient_hook(self, module, inputs, outputs):
        self.gradients = [outputs[0].detach().cpu()] + self.gradients

    def _backward(self, outputs, **kwargs):
        if self.loss_function is None:
            loss = outputs[:, 1].sum()
        else:
            loss = self.loss_function(outputs, **kwargs)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _compute_scores(self, masks):
        scores = []
        activations = [activation.numpy() for activation in self.activations]
        gradients = [gradient.numpy() for gradient in self.gradients]

        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            # Compute GradCAM scores
            weights = self._compute_weights(activation, gradient, masks)
            gradcam = weights * activation
            # Reshape the scores according to the patch size
            shape = gradcam.shape
            assert shape[-1] >= np.prod(self.patch_shape), \
                f"The patch shape {self.patch_shape} is too large, i.e., {self.patch_shape} vs {shape[-1]}."
            n = self.patch_shape[0] * self.patch_shape[1]
            gradcam = gradcam[..., -n:].reshape(shape[:-1] + self.patch_shape)
            gradcam = np.mean(gradcam, axis=1)
            scores.append(gradcam)
        return scores

    @abstractmethod
    def _compute_weights(self, activations, gradients, masks):
        pass

    def explain(self, X: MultiInputs, **kwargs):
        assert "image" in X, "The input doesn't have attribute `image`."
        assert "text" in X, "The input doesn't have attribute `text`."

        tokenized_texts = self.tokenizer(X.text.to_str())
        try:
            masks = tokenized_texts.attention_mask
        except:
            masks = tokenized_texts["attention_mask"]
        masks = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else \
            np.array(masks, dtype=np.float32)
        if masks.ndim == 1:
            masks = np.expand_dims(masks, axis=0)

        self.activations, self.gradients = [], []
        outputs = self.model(*self.preprocess(X))
        self._backward(outputs, **kwargs)
        scores = self._compute_scores(masks)


class GradCAM(Base):

    def __init__(
            self,
            model: nn.Module,
            target_layer: nn.Module,
            preprocess_function: Callable,
            tokenizer: Callable,
            loss_function: Callable = None,
            patch_shape: tuple = (24, 24),
            **kwargs
    ):
        super().__init__(
            model=model,
            target_layer=target_layer,
            preprocess_function=preprocess_function,
            tokenizer=tokenizer,
            loss_function=loss_function,
            patch_shape=patch_shape,
            **kwargs
        )

    def _compute_weights(self, activations, gradients, masks):
        return gradients * masks[:, None, :, None]
