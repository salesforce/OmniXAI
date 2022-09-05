#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
import unittest
from torchvision import models
from tensorflow.keras.applications import vgg16
from omnixai.explainers.vision.specific.feature_visualization.visualizer import FeatureVisualizer


class TestExplainer(unittest.TestCase):

    def setUp(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.vgg16(pretrained=True).to(device)
        self.target_layer = self.model.features[20]
        # self.model = vgg16.VGG16()
        # self.target_layer = self.model.layers[15]

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        plt.imshow(x)
        plt.show()

    def test_layer(self):
        optimizer = FeatureVisualizer(
            model=self.model,
            objectives=[{"layer": self.target_layer, "type": "channel", "index": list(range(5))}]
        )
        results = optimizer.explain(
            num_iterations=300,
            image_shape=(224, 224)
        )
        for res in results.to_pil():
            self._plot(res)


if __name__ == "__main__":
    unittest.main()
