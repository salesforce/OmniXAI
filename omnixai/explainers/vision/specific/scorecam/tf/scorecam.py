#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from tqdm import trange

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
        inputs = self.preprocess(X) if self.preprocess is not None else X.to_numpy()
        model = tf.keras.Model(self.model.input, [self.target_layer.output, self.model.output])

        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(X))]
                else:
                    assert len(X) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of images in X."
                    )
        else:
            y = None

        inputs = tf.convert_to_tensor(inputs)
        activations, predictions = model(inputs)
        if self.mode == "classification" and y is None:
            y = tf.argmax(predictions, axis=-1).numpy()

        print(activations.shape)
        print(predictions.shape)

        weights = []
        for i in trange(activations.shape[-1]):
            saliency = activations[..., i:i + 1]
            pass
