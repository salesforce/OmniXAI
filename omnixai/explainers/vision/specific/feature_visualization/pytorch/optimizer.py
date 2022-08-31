#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from dataclasses import dataclass


@dataclass
class Objective:
    layer: nn.Module
    weight: float = 1.0
    channel_indices: Union[int, List[int]] = None
    neuron_indices: Union[int, List[int]] = None
    direction_vectors: Union[np.ndarray, List[np.ndarray]] = None


class FeatureOptimizer:
    """
    The optimizer for feature visualization.
    """

    def __init__(
            self,
            model: nn.Module,
            objectives: Union[Objective, List[Objective]],
            **kwargs
    ):
        self.model = model
        self.objectives = objectives if isinstance(objectives, (list, tuple)) \
            else [objectives]

        self.hooks = []
        self.layer_outputs = {}
        self._register_hooks()

    def _get_hook(self, index):
        def _activation_hook(module, inputs, outputs):
            self.layer_outputs[index] = outputs
        return _activation_hook

    def _register_hooks(self):
        for i, obj in enumerate(self.objectives):
            self.hooks.append(obj.layer.register_forward_hook(self._get_hook(i)))

    def _unregister_hooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def __del__(self):
        self._unregister_hooks()

    def optimize(
            self,
            image_shape,
            num_channels=None,
    ):
        if num_channels is None:
            num_channels = 3
        device = next(self.model.parameters()).device
        values = np.zeros((1, num_channels, *image_shape))
        inputs = torch.tensor(
            values, dtype=torch.float32, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([inputs], lr=0.1)

        outputs = self.model(inputs)
        print(outputs.shape)
        print(self.layer_outputs[0].shape)

        loss = torch.mean(self.layer_outputs[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
