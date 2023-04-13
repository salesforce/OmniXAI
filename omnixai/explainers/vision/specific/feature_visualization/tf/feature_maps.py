#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import tensorflow as tf


class FeatureMapExtractor:

    def __init__(
            self,
            model: tf.keras.Model,
            layer: tf.keras.layers.Layer,
            **kwargs
    ):
        self.model = tf.keras.Model(model.input, layer.output)

    def extract(self, x):
        outputs = self.model(x)
        return outputs.numpy()
