#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from tensorflow.keras.applications import mobilenet_v2
from omnixai.explainers.vision.specific.feature_visualization.tf.optimizer import \
    Objective, FeatureOptimizer


class TestExplainer(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v2.MobileNetV2(
            include_top=True, weights="imagenet", classes=1000)

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        plt.imshow(x)
        plt.show()

    def test(self):
        objectives = [
            Objective(
                layer=self.model.layers[-5]
            )
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        results = optimizer.optimize(
            num_iterations=256,
            verbose=True
        )
        self._plot(results[-1][0])


if __name__ == "__main__":
    unittest.main()
