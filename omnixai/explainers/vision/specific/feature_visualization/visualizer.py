#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The feature visualizer for vision models.
"""
import warnings
import numpy as np

from ....base import ExplainerBase
from .....data.image import Image


class FeatureVisualizer(ExplainerBase):
    """
    Feature visualization for vision models. The input of the model has shape (B, C, H, W)
    for PyTorch and (B, H, W, C) for TensorFlow. This class applies the optimized based method
    for visualizing layer, channel, neuron features. For more details, please visit
    `https://distill.pub/2017/feature-visualization/`.
    """
    explanation_type = "global"
    alias = ["fv", "feature_visualization"]

    def __init__(
            self,
            model,
            **kwargs,
    ):
        super().__init__()

    def explain(self, **kwargs):
        pass
