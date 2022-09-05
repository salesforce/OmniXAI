#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import numpy as np
import tensorflow as tf
from omnixai.explainers.vision.specific.feature_visualization.tf.feature_maps import \
    FeatureMapExtractor


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

    def test(self):
        extractor = FeatureMapExtractor(
            model=self.model,
            layer=self.model.layers[-4]
        )
        x = tf.convert_to_tensor(np.random.rand(1, 28, 28, 3), dtype=tf.float32)
        outputs = extractor.extract(x)
        print(outputs.shape)


if __name__ == "__main__":
    unittest.main()
