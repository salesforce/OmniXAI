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
from omnixai.explanations.base import ExplanationBase


class TestExplainer(unittest.TestCase):

    def setUp(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.vgg16(pretrained=True).to(device)
        self.target_layer = self.model.features[-6]
        # self.model = vgg16.VGG16()
        # self.target_layer = self.model.layers[15]

    def test(self):
        optimizer = FeatureVisualizer(
            model=self.model,
            objectives=[{"layer": self.target_layer, "type": "channel", "index": list(range(5))}]
        )
        explanations = optimizer.explain(
            num_iterations=300,
            image_shape=(224, 224),
            use_fft=True
        )

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.ipython_plot()


if __name__ == "__main__":
    unittest.main()
