#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable

from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.utils.misc import is_tf_available
from omnixai.explanations.image.pixel_importance import PixelImportance

if not is_tf_available():
    raise EnvironmentError("Tensorflow cannot be found.")
else:
    import tensorflow as tf


class ScoreCAM:

    def __init__(
            self,
            model: tf.keras.Model,
            target_layer: tf.keras.layers.Layer,
            preprocess_function: Callable,
            mode: str = "classification",
    ):
        assert isinstance(
            model, tf.keras.Model
        ), f"`model` should be an instance of tf.keras.Model instead of {type(model)}"
        assert isinstance(
            target_layer, tf.keras.layers.Layer
        ), f"`target_layer` should be an instance of tf.keras.layers.Layer instead of {type(target_layer)}"

        self.model = model
        self.target_layer = target_layer
        self.preprocess = preprocess_function
        self.mode = mode

    def explain(self, X: Image, y=None, **kwargs):
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode, use_heatmap=True)
