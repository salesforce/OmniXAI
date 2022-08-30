#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import unittest
import numpy as np
import tensorflow as tf
from omnixai.explainers.vision.specific.feature_visualization.tf.optimizer import \
    Objective, FeatureOptimizer


class TestFeatureOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 3)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), name="early"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), name="features"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(name="pre-logits"),
            tf.keras.layers.Dense(10, name="logits")
        ])
        self.model.compile()

    def test_build_model(self):
        print(self.model.layers[-1])

    def test_layer(self):
        objective = Objective(
            layer=self.model.layers[-1]
        )
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objective
        )
        loss, layer_masks = optimizer._layer_loss(objective)

        mask = np.ones(self.model.layers[-1].output.shape[1:])
        self.assertEqual(layer_masks[0].shape, mask.shape)
        self.assertListEqual(layer_masks[0].tolist(), mask.tolist())

        model, objective_func, input_shape = optimizer._build_model()
        self.assertEqual(model.outputs[0].name, model.layers[-1].output.name)
        self.assertEqual(input_shape, (1, 28, 28, 3))


if __name__ == "__main__":
    unittest.main()
