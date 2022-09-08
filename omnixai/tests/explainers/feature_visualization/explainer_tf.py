#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
from tensorflow.keras.applications import vgg16
from omnixai.explainers.vision.specific.feature_visualization.tf.optimizer import \
    Objective, FeatureOptimizer


class TestExplainer(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vgg16.VGG16()

    @staticmethod
    def _plot(x):
        import matplotlib.pyplot as plt
        plt.imshow(x)
        plt.show()

    def test(self):
        objectives = [
            Objective(
                layer=self.model.layers[15],
                channel_indices=list(range(5))
            )
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        results, names = optimizer.optimize(
            num_iterations=300
        )
        for res, name in zip(results[-1], names):
            print(name)
            self._plot(res)


if __name__ == "__main__":
    unittest.main()
