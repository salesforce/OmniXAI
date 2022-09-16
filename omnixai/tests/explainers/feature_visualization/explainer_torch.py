#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import unittest
import numpy as np
from torchvision import models
from omnixai.explainers.vision.specific.feature_visualization.pytorch.optimizer import \
    Objective, FeatureOptimizer


class TestExplainer(unittest.TestCase):

    def setUp(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.vgg16(pretrained=True).to(device)
        print(self.model.features)

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
        plt.imshow(x)
        plt.show()

    def test_layer(self):
        objectives = [
            Objective(
                layer=self.model.features[-6],
                channel_indices=list(range(5))
            )
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        results, names = optimizer.optimize(
            num_iterations=300,
            image_shape=(224, 224),
            use_fft=True
        )
        for res, name in zip(results[-1], names):
            print(name)
            self._plot(res)


if __name__ == "__main__":
    unittest.main()
