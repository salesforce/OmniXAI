#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch.nn as nn
import numpy as np


class FeatureMapExtractor:

    def __init__(
            self,
            model: nn.Module,
            layer: nn.Module
    ):
        self.model = model.eval()
        self.layer = layer

        self.hooks = []
        self.layer_outputs = None
        self._register_hooks()

    def _get_hook(self):
        def _activation_hook(module, inputs, outputs):
            self.layer_outputs = outputs

        return _activation_hook

    def _register_hooks(self):
        self.hooks.append(self.layer.register_forward_hook(self._get_hook()))

    def _unregister_hooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def __del__(self):
        self._unregister_hooks()

    def extract(self, x):
        device = next(self.model.parameters()).device
        self.model(x.to(device))
        outputs = self.layer_outputs.detach().cpu().numpy()
        outputs = np.swapaxes(np.swapaxes(outputs, 1, 2), 2, 3)
        return outputs
