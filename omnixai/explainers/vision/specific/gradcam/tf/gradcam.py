#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from abc import abstractmethod

from omnixai.utils.misc import AutodocABCMeta
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.utils.misc import is_tf_available
from omnixai.explanations.image.pixel_importance import PixelImportance

if not is_tf_available():
    raise EnvironmentError("Tensorflow cannot be found.")
else:
    import tensorflow as tf


class Base(metaclass=AutodocABCMeta):
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: tf.keras.layers.Layer,
        preprocess_function: Callable,
        mode: str = "classification",
    ):
        super().__init__()
        assert isinstance(
            model, tf.keras.Model
        ), f"`model` should be an instance of tf.keras.Model instead of {type(model)}"
        assert isinstance(
            target_layer, tf.keras.layers.Layer
        ), f"`target_layer` should be an instance of tf.keras.layers.Layer instead of {type(target_layer)}"

        self.model = model
        self.target_layers = [target_layer]
        self.preprocess = preprocess_function
        self.mode = mode

    def _activations_and_gradients(self, inputs, targets):
        model = tf.keras.Model(
            inputs=[self.model.inputs], outputs=[[layer.output for layer in self.target_layers], self.model.output]
        )
        inputs = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape:
            activations, predictions = model(inputs)
            if self.mode == "classification":
                y = tf.argmax(predictions, axis=-1) if targets is None else tf.convert_to_tensor(targets)
                predictions = tf.reshape(tf.gather(predictions, y, axis=1), shape=(-1,))
                y = y.numpy()
            else:
                y = targets
            gradients = tape.gradient(predictions, activations)

        activation_values = [np.transpose(act.numpy(), (0, 3, 1, 2)) for act in activations]
        gradient_values = [np.transpose(grad.numpy(), (0, 3, 1, 2)) for grad in gradients]
        return activation_values, gradient_values, y

    def _compute_scores(self, activations, gradients):
        scores = []
        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            weights = self._compute_weights(activation, gradient)
            combination = weights[:, :, None, None] * activation
            score = combination.sum(axis=1)
            score[score < 0] = 0
            scores.append(score)
        return scores

    @abstractmethod
    def _compute_weights(self, activations, gradients):
        pass

    @staticmethod
    def _resize_scores(inputs, scores):
        size = inputs.shape[2:] if inputs.shape[1] == 3 else inputs.shape[1:3]
        resized_scores = []
        for score in scores:
            for i in range(score.shape[0]):
                min_val, max_val = np.min(score[i]), np.max(score[i])
                score[i] = (score[i] - min_val) / (max_val - min_val + 1e-8) * 255
            im = Resize(size).transform(Image(data=score, batched=True))
            resized_scores.append(im.to_numpy() / 255.0)
        return resized_scores

    @staticmethod
    def _resize_image(image, inputs):
        assert image.shape[0] == 1, "`image` can contain one instance only."
        y = image.to_numpy()
        x = inputs
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)

    def explain(self, X: Image, y=None, **kwargs):
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode, use_heatmap=True)
        inputs = self.preprocess(X) if self.preprocess is not None else X.to_numpy()

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

        # Compute activations and gradients
        activations, gradients, y = self._activations_and_gradients(inputs, targets=y)
        # Compute importance scores
        scores = self._compute_scores(activations, gradients)
        # Resize the importance scores
        scores = self._resize_scores(inputs, scores)
        # By default, there is only one target_layer
        scores = scores[0]

        for i, instance in enumerate(inputs):
            image = self._resize_image(X[i], instance).to_numpy()[0]
            label = y[i] if y is not None else None
            explanations.add(image=image, target_label=label, importance_scores=scores[i])
        return explanations


class GradCAM(Base):
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: tf.keras.layers.Layer,
        preprocess_function: Callable,
        mode: str = "classification",
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        return np.mean(gradients, axis=(2, 3))


class GradCAMPlus(Base):
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: tf.keras.layers.Layer,
        preprocess_function: Callable,
        mode: str = "classification",
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        a = gradients ** 2
        b = a * gradients
        sum_activations = np.sum(activations, axis=(2, 3))
        mat = a / (2 * a + sum_activations[:, :, None, None] * b + 1e-8)
        mat = np.where(gradients != 0, mat, 0)
        return np.sum(np.maximum(gradients, 0) * mat, axis=(2, 3))


class LayerCAM(Base):
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer: tf.keras.layers.Layer,
        preprocess_function: Callable,
        mode: str = "classification",
    ):
        super().__init__(model=model, target_layer=target_layer, preprocess_function=preprocess_function, mode=mode)

    def _compute_weights(self, activations, gradients):
        assert len(gradients.shape) == 4, f"The ndim of `gradients` should be 4 instead of {len(gradients.shape)}"
        gradients[gradients < 0] = 0
        return gradients

    def _compute_scores(self, activations, gradients):
        scores = []
        for layer, activation, gradient in zip(self.target_layers, activations, gradients):
            weights = self._compute_weights(activation, gradient)
            combination = weights * activation
            score = combination.sum(axis=1)
            score[score < 0] = 0
            scores.append(score)
        return scores
