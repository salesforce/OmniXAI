#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import torchvision
import torch.nn as nn
import numpy as np
from typing import Union, List
from collections import defaultdict
from ..utils import Objective, FeatureOptimizerMixin
from ..utils import fft_inputs, fft_scale
from .preprocess import fft_images


class FeatureOptimizer(FeatureOptimizerMixin):
    """
    The optimizer for feature visualization.
    """

    def __init__(
            self,
            model: nn.Module,
            objectives: Union[Objective, List[Objective]],
            **kwargs
    ):
        self.model = model.eval()
        self.objectives = objectives if isinstance(objectives, (list, tuple)) \
            else [objectives]
        self.formatted_objectives, self.num_combinations, self.names = \
            self._process_objectives(self.objectives)

        self.hooks = []
        self.layer_outputs = {}
        self._register_hooks()

    def _get_hook(self, index):
        if isinstance(index, int):
            index = [index]

        def _activation_hook(module, inputs, outputs):
            for i in index:
                self.layer_outputs[i] = outputs

        return _activation_hook

    def _register_hooks(self):
        indices = defaultdict(list)
        for i, obj in enumerate(self.objectives):
            indices[obj.layer].append(i)
        for layer, index in indices.items():
            self.hooks.append(layer.register_forward_hook(self._get_hook(index)))

    def _unregister_hooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def __del__(self):
        self._unregister_hooks()

    def _loss(self):
        loss = 0
        for i, obj in enumerate(self.formatted_objectives):
            outputs = self.layer_outputs[i]
            # Layer loss
            if obj["type"] == "layer":
                loss += -torch.mean(
                    outputs, dim=list(range(1, len(outputs.shape)))
                ) * obj["weight"]
            # Channel loss
            elif obj["type"] == "channel":
                idx = torch.arange(outputs.shape[0])
                outputs = outputs[idx, obj["batch_indices"]]
                loss += -torch.mean(
                    outputs, dim=list(range(1, len(outputs.shape)))
                ) * obj["weight"]
            # Neuron loss
            elif obj["type"] == "neuron":
                idx = torch.arange(outputs.shape[0])
                y = outputs.reshape((outputs.shape[0], -1))
                loss += -y[idx, obj["batch_indices"]] * obj["weight"]
            # Direction loss
            elif obj["type"] == "direction":
                if isinstance(obj["vector"], np.ndarray):
                    obj["vector"] = torch.tensor(obj["vector"], device=outputs.device)
                loss += -self._dot_cos(outputs, obj["vector"])
        return loss

    @staticmethod
    def _dot_cos(x, y):
        x = x.view((x.shape[0], -1))
        y = y.view((y.shape[0], -1))
        a = x / torch.norm(x, dim=1, keepdim=True)
        b = y / torch.norm(y, dim=1, keepdim=True)
        cos = torch.clamp(torch.sum(a * b, dim=1), min=1e-1) ** 2
        dot = torch.sum(x * y)
        return dot * cos

    @staticmethod
    def _default_transform(size):
        from omnixai.preprocessing.pipeline import Pipeline
        from .preprocess import RandomBlur, RandomCrop, \
            RandomResize, RandomFlip, Padding

        unit = max(int(size / 32), 2)
        pipeline = Pipeline() \
            .step(Padding(size=unit * 4)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomResize((0.8, 1.2))) \
            .step(RandomBlur(kernel_size=9)) \
            .step(RandomCrop(unit)) \
            .step(RandomCrop(unit)) \
            .step(RandomFlip())
        return pipeline

    @staticmethod
    def _normal_color(x):
        mat = torch.tensor(
            [[0.56282854, 0.58447580, 0.58447580],
             [0.19482528, 0.00000000, -0.19482528],
             [0.04329450, -0.10823626, 0.06494176]],
            dtype=x.dtype,
            device=x.device
        )
        y = torch.transpose(torch.transpose(x, 1, 2), 2, 3)
        y = torch.matmul(y.reshape((-1, 3)), mat).reshape(y.shape)
        return torch.transpose(torch.transpose(y, 2, 3), 1, 2)

    @staticmethod
    def _normalize(x, normalizer, value_range, normal_color=True):
        if normal_color:
            x = FeatureOptimizer._normal_color(x)
        min_value, max_value = value_range
        x = torch.sigmoid(x) if normalizer == "sigmoid" \
            else torch.clip(x, min_value, max_value)
        y = x.reshape((x.shape[0], -1))
        y = y - torch.min(y, dim=1, keepdim=True)[0]
        y = y / (torch.max(y, dim=1, keepdim=True)[0] + 1e-8)
        y = y * (max_value - min_value) + min_value
        return y.reshape(x.shape)

    @staticmethod
    def total_variation(x):
        b, c, h, w = x.shape
        tv_h = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=(1, 2, 3))
        tv_w = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=(1, 2, 3))
        return (tv_h + tv_w) / (c * h * w)

    @staticmethod
    def _regularize(reg_type, weight):
        if reg_type is None or reg_type == "":
            return lambda x: 0
        elif reg_type == "l1":
            return lambda x: torch.mean(torch.abs(x), dim=(1, 2, 3)) * weight
        elif reg_type == "l2":
            return lambda x: torch.sqrt(torch.mean(x ** 2, dim=(1, 2, 3))) * weight
        elif reg_type == "tv":
            return lambda x: FeatureOptimizer.total_variation(x) * weight
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

    def optimize(
            self,
            *,
            num_iterations=300,
            learning_rate=0.05,
            transformers=None,
            regularizers=None,
            image_shape=None,
            value_normalizer="sigmoid",
            value_range=(0.05, 0.95),
            init_std=0.01,
            use_fft=False,
            fft_decay=1.0,
            normal_color=False,
            save_all_images=False,
            verbose=True,
    ):
        from omnixai.utils.misc import ProgressBar
        bar = ProgressBar(num_iterations) if verbose else None

        if image_shape is None:
            image_shape = (224, 224)
        if transformers is None:
            transformers = self._default_transform(min(image_shape[0], image_shape[1]))
        if regularizers is not None:
            if not isinstance(regularizers, list):
                regularizers = [regularizers]
            regularizers = [self._regularize(reg, w) for reg, w in regularizers]
        if use_fft:
            # Using "normal color" for FFT preconditioning
            normal_color = True

        device = next(self.model.parameters()).device
        shape = (self.num_combinations, 3, *image_shape)
        if not use_fft:
            inputs = torch.tensor(
                np.random.randn(*shape) * init_std,
                dtype=torch.float32,
                requires_grad=True,
                device=device
            )
            normalize = lambda x: self._normalize(
                x, value_normalizer, value_range, normal_color)
        else:
            inputs = torch.tensor(
                fft_inputs(*shape, mode="torch", std=init_std),
                dtype=torch.float32,
                requires_grad=True,
                device=device
            )
            scales = fft_scale(
                image_shape[0], image_shape[1], mode="torch", decay_power=fft_decay)
            scales = torch.tensor(scales, dtype=torch.complex64, device=device)
            normalize = lambda x: self._normalize(
                fft_images(image_shape[0], image_shape[1], inputs, scales),
                value_normalizer, value_range, normal_color
            )
        optimizer = torch.optim.Adam([inputs], lr=learning_rate)

        results = []
        for i in range(num_iterations):
            images = transformers.transform(normalize(inputs))
            images = torchvision.transforms.Resize((image_shape[0], image_shape[1]))(images)
            self.model(images)
            loss = self._loss()
            if regularizers is not None:
                for func in regularizers:
                    loss += func(images)

            optimizer.zero_grad()
            grad = torch.autograd.grad(torch.unbind(loss), inputs)[0]
            inputs.grad = grad
            optimizer.step()

            if save_all_images or i == num_iterations - 1:
                results.append(normalize(inputs).detach().cpu().numpy())
            if verbose:
                bar.print(i + 1, prefix=f"Step: {i + 1}", suffix="")
        return results, self.names
