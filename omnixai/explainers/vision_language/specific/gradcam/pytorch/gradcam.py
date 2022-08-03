#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import inspect
import numpy as np
from typing import Callable
from abc import abstractmethod

from omnixai.utils.misc import AutodocABCMeta
from omnixai.data.image import Image
from omnixai.data.multi_inputs import MultiInputs
from omnixai.utils.misc import is_torch_available
from omnixai.preprocessing.image import Resize
from omnixai.explanations.image.pixel_importance import PixelImportance

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
        loss = self.loss_function(outputs, **kwargs)
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _compute_scores(self, masks):
        scores = []
        activations = [activation.numpy() for activation in self.activations]
        gradients = [gradient.numpy() for gradient in self.gradients]

        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            # Compute GradCAM scores
            gradcam = self._compute_gradcam(activation, gradient, masks)
            # Reshape the scores according to the patch size
            shape = gradcam.shape
            assert shape[-1] >= np.prod(self.patch_shape), \
                f"The patch shape {self.patch_shape} is too large, i.e., {self.patch_shape} vs {shape[-1]}."
            n = self.patch_shape[0] * self.patch_shape[1]
            gradcam = gradcam[..., -n:].reshape(shape[:-1] + self.patch_shape)

            # The GradCAM scores corresponding to the tokens
            gradcam = np.mean(gradcam, axis=1)
            # The average GradCAM scores
            lengths, avg = np.sum(masks, axis=1), np.sum(gradcam, axis=1)
            avg = avg / lengths.reshape((-1,) + (1,) * (avg.ndim - 1))
            scores.append((gradcam, avg))
        return scores

    @abstractmethod
    def _compute_gradcam(self, activations, gradients, masks):
        pass

    @staticmethod
    def _padding(masks):
        lengths = [len(mask) for mask in masks]
        if min(lengths) == max(lengths):
            return np.array(masks, dtype=np.float32)
        else:
            m = np.zeros((len(masks), max(lengths)))
            for i, (length, mask) in enumerate(zip(lengths, masks)):
                m[i, :length] = mask
            return m

    @staticmethod
    def _resize_scores(scores, shape):
        resized_scores = []
        for score in scores:
            min_val, max_val = np.min(score), np.max(score)
            score = (score - min_val) / (max_val - min_val + 1e-8) * 255
            im = Resize(shape).transform(Image(data=score, batched=False))
            resized_scores.append(im.to_numpy() / 255.0)
        return np.concatenate(resized_scores, axis=0)

    def explain(self, X: MultiInputs, **kwargs):
        assert "image" in X, "The input doesn't have attribute `image`."
        assert "text" in X, "The input doesn't have attribute `text`."
        explanations = PixelImportance("vlm", use_heatmap=True)

        tokenizer_params = {}
        signature = inspect.signature(self.tokenizer).parameters
        if "padding" in signature:
            tokenizer_params["padding"] = True
        tokenized_texts = self.tokenizer(X.text.values, **tokenizer_params)
        masks = self._padding(tokenized_texts["attention_mask"])

        self.activations, self.gradients = [], []
        inputs = self.preprocess(X)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        outputs = self.model(*inputs)
        self._backward(outputs, **kwargs)
        scores = self._compute_scores(masks)
        # By default, there is only one target_layer
        gradcams, avg = scores[0]

        shape = X.image.image_shape[:2]
        avg = self._resize_scores(avg, shape)
        gradcams = np.stack([self._resize_scores(g, shape) for g in gradcams])

        for i in range(X.num_samples()):
            tokens = [t for t, m in zip(tokenized_texts["input_ids"][i],
                                        tokenized_texts["attention_mask"][i])
                      if m > 0]
            labels = ["Avg GradCAM"] + [self.tokenizer.decode([t]) for t in tokens]
            scores = [avg[i]] + [g for g in gradcams[i][:len(tokens)]]
            explanations.add(
                image=X.image[i].to_numpy()[0],
                importance_scores=scores,
                labels=labels,
                target_label=None
            )
        return explanations


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

    def _compute_gradcam(self, activations, gradients, masks):
        scores = gradients * activations * masks[:, None, :, None]
        scores[scores < 0] = 0
        return scores
