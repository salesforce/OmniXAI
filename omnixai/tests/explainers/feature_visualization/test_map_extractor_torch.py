#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import torch
import numpy as np
import torch.nn as nn
from omnixai.explainers.vision.specific.feature_visualization.pytorch.feature_maps import \
    FeatureMapExtractor


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

    def test(self):
        extractor = FeatureMapExtractor(
            model=self.model,
            layer=self.model.layers[4]
        )
        x = torch.tensor(np.random.rand(1, 3, 128, 128), dtype=torch.float32)
        outputs = extractor.extract(x)
        print(outputs.shape)


if __name__ == "__main__":
    unittest.main()
