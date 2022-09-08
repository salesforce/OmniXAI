#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import torch.nn as nn
from omnixai.explainers.vision.specific.feature_visualization.pytorch.optimizer import \
    Objective, FeatureOptimizer


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)


class TestFeatureOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Model()

    def test_layer(self):
        objectives = [
            Objective(
                layer=self.model.layers[4]
            ),
            Objective(
                layer=self.model.layers[4],
                channel_indices=0
            ),
            Objective(
                layer=self.model.layers[4],
                neuron_indices=[0, 1]
            ),
            Objective(
                layer=self.model.layers[4],
                direction_vectors=np.random.random((16, 26, 26))
            ),
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        optimizer.optimize(
            num_iterations=2,
            image_shape=(64, 64)
        )


if __name__ == "__main__":
    unittest.main()
