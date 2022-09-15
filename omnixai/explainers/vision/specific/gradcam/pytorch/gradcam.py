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
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.utils.misc import is_torch_available
from omnixai.explanations.image.pixel_importance import PixelImportance

if not is_torch_available():
    raise EnvironmentError("Torch cannot be found.")
else:
    import torch
    import torch.nn as nn


class Base(metaclass=AutodocABCMeta):
    def __init__(
        self, model: nn.Module, target_layer: nn.Module, preprocess_function: Callable, mode: str = "classification"
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
        self.mode = mode

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

    def _backward(self, outputs, indices):
        if indices is not None:
            loss = 0
            for i, label in enumerate(indices):
                loss = loss + outputs[i, label]
        else:
            loss = torch.sum(outputs)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _compute_scores(self):
        scores = []
        activations = [activation.numpy() for activation in self.activations]
        gradients = [gradient.numpy() for gradient in self.gradients]
        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            weights = self._compute_weights(activation, gradient)
            combination = weights[:, :, None, None] * activation
            score = combination.sum(axis=1)
            score[score < 0] = 0
            scores.append(score)
        return scores

    @abstractmethod
    def _compute_weights(self, activations, gradients):
        pass

    @staticmethod
    def _resize_scores(inputs, scores):
        size = inputs.shape[2:] if inputs.shape[1] == 3 else inputs.shape[1:3]
        resized_scores = []
        for score in scores:
            for i in range(score.shape[0]):
                min_val, max_val = np.min(score[i]), np.max(score[i])
                score[i] = (score[i] - min_val) / (max_val - min_val + 1e-8) * 255
            im = Resize(size).transform(Image(data=score, batched=True))
            resized_scores.append(im.to_numpy() / 255.0)
        return resized_scores

    @staticmethod
    def _resize_image(image, inputs):
        assert image.shape[0] == 1, "`image` can contain one instance only."
        y = image.to_numpy()
        x = inputs
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)

    def explain(self, X: Image, y=None, **kwargs):
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode, use_heatmap=True)

        self.activations, self.gradients = [], []
        device = next(self.model.parameters()).device
        inputs = self.preprocess(X) if self.preprocess is not None else X.to_numpy()
        inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.get_default_dtype())

        # Forward pass
        outputs = self.model(inputs.to(device))
        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(X))]
                else:
                    assert len(X) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of images in X."
                    )
            else:
                scores = outputs.detach().cpu().numpy()
                y = np.argmax(scores, axis=1).astype(int)
        else:
            y = None

        # Backward pass
        self._backward(outputs, y)
        # Compute importance scores
        scores = self._compute_scores()
        # Resize the importance scores
        scores = self._resize_scores(inputs, scores)
        # By default, there is only one target_layer
        scores = scores[0]

        for i, instance in enumerate(inputs):
            image = self._resize_image(X[i], instance).to_numpy()[0]
            label = y[i] if y is not None else None
            explanations.add(image=image, target_label=label, importance_scores=scores[i])
        return explanations


class GradCAM(Base):
    def __init__(
        self, model: nn.Module, target_layer: nn.Module, preprocess_function: Callable, mode: str = "classification"
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        return np.mean(gradients, axis=(2, 3))


class GradCAMPlus(Base):
    def __init__(
        self, model: nn.Module, target_layer: nn.Module, preprocess_function: Callable, mode: str = "classification"
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        a = gradients ** 2
        b = a * gradients
        sum_activations = np.sum(activations, axis=(2, 3))
        mat = a / (2 * a + sum_activations[:, :, None, None] * b + 1e-8)
        mat = np.where(gradients != 0, mat, 0)
        return np.sum(np.maximum(gradients, 0) * mat, axis=(2, 3))


class LayerCAM(Base):
    def __init__(
        self, model: nn.Module, target_layer: nn.Module, preprocess_function: Callable, mode: str = "classification"
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        gradients[gradients < 0] = 0
        return gradients

    def _compute_scores(self):
        scores = []
        activations = [activation.numpy() for activation in self.activations]
        gradients = [gradient.numpy() for gradient in self.gradients]
        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            weights = self._compute_weights(activation, gradient)
            combination = weights * activation
            score = combination.sum(axis=1)
            score[score < 0] = 0
            scores.append(score)
        return scores
