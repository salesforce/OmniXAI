#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import unittest
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model().to(device)

    def test_layer(self):
        objective = Objective(
            layer=self.model.layers[4]
        )
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objective
        )
        optimizer.optimize(
            num_iterations=10,
            image_shape=(64, 64)
        )


if __name__ == "__main__":
    unittest.main()