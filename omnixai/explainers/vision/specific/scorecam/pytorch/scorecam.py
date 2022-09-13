#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from scipy.special import softmax
from typing import Callable
from tqdm import trange

from omnixai.data.image import Image
from omnixai.utils.misc import is_torch_available
from omnixai.explanations.image.pixel_importance import PixelImportance
from ..utils import ScoreCAMMixin

if not is_torch_available():
    raise EnvironmentError("Torch cannot be found.")
else:
    import torch
    import torch.nn as nn


class ScoreCAM(ScoreCAMMixin):

    def __init__(
            self,
            model: nn.Module,
            target_layer: nn.Module,
            preprocess_function: Callable,
            mode: str = "classification"
    ):
        assert isinstance(
            model, nn.Module
        ), f"`model` should be an instance of torch.nn.Module instead of {type(model)}"
        assert isinstance(
            target_layer, nn.Module
        ), f"`target_layer` should be an instance of torch.nn.Module instead of {type(target_layer)}"

        self.model = model.eval()
        self.target_layer = target_layer
        self.preprocess = preprocess_function
        self.mode = mode

        self.hooks = []
        self.layer_output = None
        self._register_hooks()

    def _register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self._activation_hook))

    def _unregister_hooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def __del__(self):
        self._unregister_hooks()

    def _activation_hook(self, module, inputs, outputs):
        self.layer_output = outputs.detach()

    @staticmethod
    def _normalize(x):
        if len(x.shape) == 4:
            assert x.shape[1] == 1
            x = x.squeeze(dim=1)
        assert len(x.shape) == 3
        y = x.view((x.shape[0], -1))
        min_value, _ = torch.min(y, dim=1, keepdim=True)
        max_value, _ = torch.max(y, dim=1, keepdim=True)
        y = (y - min_value) / (max_value - min_value + 1e-6)
        return y.view(*x.shape)

    def explain(self, X: Image, y=None, **kwargs):
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode, use_heatmap=True)

        device = next(self.model.parameters()).device
        inputs = self.preprocess(X) if self.preprocess is not None else X.to_numpy()
        inputs = inputs if isinstance(inputs, torch.Tensor) else \
            torch.tensor(inputs, dtype=torch.get_default_dtype())
        inputs = inputs.to(device)

        outputs = self.model(inputs)
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
        layer_outputs = self.layer_output

        weights = []
        for i in trange(layer_outputs.shape[1]):
            saliency = layer_outputs[:, i:i + 1, ...]
            saliency = nn.functional.interpolate(
                saliency, size=(inputs.shape[-2], inputs.shape[-1]),
                mode="bilinear", align_corners=False
            )
            norm_saliency = self._normalize(saliency).unsqueeze(dim=1)
            w = self.model((inputs * norm_saliency).to(device)).detach().cpu().numpy()
            w = np.array([w[i, label] for i, label in enumerate(y)]) \
                if self.mode == "classification" else w.flatten()
            weights.append(np.expand_dims(w, axis=-1))

        weights = np.concatenate(weights, axis=1)
        if not (np.max(weights) <= 1.0 and np.min(weights) >= 0.0):
            weights = softmax(weights, axis=1)
        targets = layer_outputs.detach().cpu().numpy()
        assert targets.shape[1] == weights.shape[1]

        score_cams = np.zeros((targets.shape[0], targets.shape[2], targets.shape[3]))
        for i in range(targets.shape[1]):
            score_cams += targets[:, i, ...] * np.expand_dims(weights[:, i], axis=(1, 2))
        score_cams = np.maximum(score_cams, 0)
        score_cams = self._resize_scores(inputs, score_cams)

        for i, instance in enumerate(inputs):
            image = self._resize_image(X[i], instance).to_numpy()[0]
            label = y[i] if y is not None else None
            explanations.add(image=image, target_label=label, importance_scores=score_cams[i])
        return explanations
