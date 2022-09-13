#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from scipy.special import softmax
from typing import Callable
from tqdm import trange

from omnixai.data.image import Image
from omnixai.utils.misc import is_tf_available
from omnixai.explanations.image.pixel_importance import PixelImportance
from ..utils import ScoreCAMMixin

if not is_tf_available():
    raise EnvironmentError("Tensorflow cannot be found.")
else:
    import tensorflow as tf


class ScoreCAM(ScoreCAMMixin):

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

    @staticmethod
    def _normalize(x):
        assert len(x.shape) == 4
        min_value = tf.reduce_min(x, (1, 2, 3), keepdims=True)
        max_value = tf.reduce_max(x, (1, 2, 3), keepdims=True)
        x = (x - min_value) / (max_value - min_value + 1e-6)
        return x

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
            y = tf.argmax(predictions, axis=-1).numpy().astype(int)

        weights = []
        for i in trange(activations.shape[-1]):
            saliency = activations[..., i:i + 1]
            saliency = tf.image.resize(
                saliency, size=(inputs.shape[1], inputs.shape[2]))
            norm_saliency = self._normalize(saliency)
            w = self.model(inputs * norm_saliency).numpy()
            w = np.array([w[i, label] for i, label in enumerate(y)]) \
                if self.mode == "classification" else w.flatten()
            weights.append(np.expand_dims(w, axis=-1))

        weights = np.concatenate(weights, axis=1)
        if not (np.max(weights) <= 1.0 and np.min(weights) >= 0.0):
            weights = softmax(weights, axis=1)
        targets = activations.numpy()
        assert targets.shape[-1] == weights.shape[1]

        score_cams = np.zeros((targets.shape[0], targets.shape[1], targets.shape[2]))
        for i in range(targets.shape[-1]):
            score_cams += targets[..., i] * np.expand_dims(weights[:, i], axis=(1, 2))
        score_cams = np.maximum(score_cams, 0)
        score_cams = self._resize_scores(inputs, score_cams, channel_last=True)

        for i, instance in enumerate(inputs):
            image = self._resize_image(X[i], instance).to_numpy()[0]
            label = y[i] if y is not None else None
            explanations.add(image=image, target_label=label, importance_scores=score_cams[i])
        return explanations
