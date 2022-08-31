#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
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
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])
        self.model.compile()

    def test_build_model(self):
        objectives = [
            Objective(
                layer=self.model.layers[-1]
            ),
            Objective(
                layer=self.model.layers[-2],
                direction_vectors=np.random.random(
                    self.model.layers[-2].output.shape[1:])
            ),
            Objective(
                layer=self.model.layers[1],
                channel_indices=list(range(5))
            ),
            Objective(
                layer=self.model.layers[4],
                neuron_indices=list(range(3))
            )
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        model, objective_func, input_shape = optimizer._build_model()

        self.assertEqual(model.outputs[0].name, self.model.layers[-1].output.name)
        self.assertEqual(model.outputs[1].name, self.model.layers[-2].output.name)
        self.assertEqual(model.outputs[2].name, self.model.layers[1].output.name)
        self.assertEqual(model.outputs[3].name, self.model.layers[4].output.name)
        self.assertEqual(input_shape, (15, 28, 28, 3))

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
        self.assertEqual(model.outputs[0].name, self.model.layers[-1].output.name)
        self.assertEqual(input_shape, (1, 28, 28, 3))

    def test_channel(self):
        objective = Objective(
            layer=self.model.layers[1],
            channel_indices=list(range(10))
        )
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objective
        )
        loss, channel_masks = optimizer._channel_loss(objective)

        mask = np.zeros(self.model.layers[1].output.shape[1:])
        mask[:, :, 0] = 1
        self.assertEqual(channel_masks[0].shape, mask.shape)
        self.assertAlmostEqual(np.sum(np.abs(channel_masks[0] - mask)), 0, delta=1e-6)

        model, objective_func, input_shape = optimizer._build_model()
        self.assertEqual(model.outputs[0].name, self.model.layers[1].output.name)
        self.assertEqual(input_shape, (10, 28, 28, 3))

    def test_neuron(self):
        objective = Objective(
            layer=self.model.layers[1],
            neuron_indices=list(range(10))
        )
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objective
        )
        loss, neuron_masks = optimizer._neuron_loss(objective)

        mask = np.zeros((10,) + self.model.layers[1].output.shape[1:])
        for i in range(10):
            mask[i, 0, i // mask.shape[3], i % mask.shape[3]] = 1.0
        self.assertEqual(neuron_masks.shape, mask.shape)
        self.assertAlmostEqual(np.sum(np.abs(neuron_masks - mask)), 0, delta=1e-6)

        model, objective_func, input_shape = optimizer._build_model()
        self.assertEqual(model.outputs[0].name, self.model.layers[1].output.name)
        self.assertEqual(input_shape, (10, 28, 28, 3))

    def test_direction(self):
        vector = np.random.random(self.model.layers[1].output.shape[1:])
        objective = Objective(
            layer=self.model.layers[1],
            direction_vectors=vector
        )
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objective
        )
        loss, direction_masks = optimizer._direction_loss(objective)
        self.assertAlmostEqual(np.sum(np.abs(direction_masks[0] - vector)), 0, delta=1e-6)

        model, objective_func, input_shape = optimizer._build_model()
        self.assertEqual(model.outputs[0].name, self.model.layers[1].output.name)
        self.assertEqual(input_shape, (1, 28, 28, 3))

    def test_optimize(self):
        objectives = [
            Objective(
                layer=self.model.layers[-1]
            ),
            Objective(
                layer=self.model.layers[-2],
                direction_vectors=np.random.random(
                    self.model.layers[-2].output.shape[1:])
            ),
            Objective(
                layer=self.model.layers[1],
                channel_indices=list(range(5))
            ),
            Objective(
                layer=self.model.layers[4],
                neuron_indices=list(range(3))
            )
        ]
        optimizer = FeatureOptimizer(
            model=self.model,
            objectives=objectives
        )
        results = optimizer.optimize(
            num_iterations=2,
            regularizers=("l1", 0.001),
            verbose=True
        )


if __name__ == "__main__":
    unittest.main()
