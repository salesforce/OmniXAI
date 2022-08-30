#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import itertools
import numpy as np
import tensorflow as tf
from typing import Union, List
from dataclasses import dataclass


@dataclass
class Objective:
    layer: tf.keras.layers.Layer
    weight: float = 1.0
    channel_indices: Union[int, List[int]] = None
    neuron_indices: Union[int, List[int]] = None
    direction_vectors: Union[np.ndarray, List[np.ndarray]] = None


class FeatureOptimizer:
    """
    The optimizer for feature visualization. The implementation is adapted from:
    https://github.com/deel-ai/xplique/tree/master/xplique/features_visualizations
    """

    def __init__(
            self,
            model: tf.keras.Model,
            objectives: Union[Objective, List[Objective]],
            **kwargs
    ):
        self.model = model
        self.objectives = objectives if isinstance(objectives, (list, tuple)) \
            else [objectives]

    def _build_model(self):
        funcs, masks = [], []
        for obj in self.objectives:
            if obj.direction_vectors is not None:
                loss_func = self._direction_loss
            elif obj.channel_indices is not None:
                loss_func = self._channel_loss
            elif obj.neuron_indices is not None:
                loss_func = self._neuron_loss
            else:
                loss_func = self._layer_loss
            func, mask = loss_func(obj)
            funcs.append(func)
            masks.append(mask)

        masks = np.array([np.array(m) for m in itertools.product(*masks)])
        assert masks.shape[1] == len(self.objectives), \
            f"The shape of `masks` doesn't match the number of objectives, " \
            f"{masks.shape[1]} != {len(self.objectives)}."
        masks = [tf.cast(tf.stack(masks[:, i]), tf.float32) for i in range(len(self.objectives))]
        weights = tf.constant([obj.weight for obj in self.objectives])

        def _objective_func(outputs):
            loss = 0.0
            for i in range(len(self.objectives)):
                loss += funcs[i](outputs[i], masks[i]) * weights[i]
            return loss

        layers = [obj.layer.output for obj in self.objectives]
        model = tf.keras.Model(self.model.input, [*layers])
        input_shape = (masks[0].shape[0], *model.input.shape[1:])
        return model, _objective_func, input_shape

    @staticmethod
    def _layer_loss(objective):
        shape = objective.layer.output.shape
        layer_masks = np.ones((1, *shape[1:]))

        def _loss(output, masks, **kwargs):
            return tf.reduce_mean(output ** 2)

        return _loss, layer_masks

    @staticmethod
    def _channel_loss(objective):
        assert objective.channel_indices is not None, \
            "`channel_indices` cannot be None."
        shape = objective.layer.output.shape
        channels = [objective.channel_indices] if isinstance(objective.channel_indices, int) \
            else objective.channel_indices
        channel_masks = np.zeros((len(channels), *shape[1:]))
        for i, c in enumerate(channels):
            channel_masks[i, ..., c] = 1

        def _loss(output, masks, **kwargs):
            return tf.reduce_mean(
                output * masks, axis=list(range(1, len(shape))))

        return _loss, channel_masks

    @staticmethod
    def _neuron_loss(objective):
        assert objective.neuron_indices is not None, \
            "`neuron_indices` cannot be None."
        shape = objective.layer.output.shape
        neurons = [objective.neuron_indices] if isinstance(objective.neuron_indices, int) \
            else objective.neuron_indices
        neuron_masks = np.zeros((len(neurons), np.prod(shape[1:])))
        for i, k in enumerate(neurons):
            neuron_masks[i, k] = 1
        neuron_masks = neuron_masks.reshape((len(neurons), *shape[1:]))

        def _loss(output, masks, **kwargs):
            return tf.reduce_mean(
                output * masks, axis=list(range(1, len(shape))))

        return _loss, neuron_masks

    @staticmethod
    def _direction_loss(objective):
        assert objective.direction_vectors is not None, \
            "`direction_vectors` cannot be None."
        direction_masks = objective.direction_vectors if isinstance(objective.direction_vectors, list) \
            else [objective.direction_vectors]
        direction_masks = np.array(direction_masks)

        def _loss(output, masks, **kwargs):
            return FeatureOptimizer._dot_cos(output, masks)

        return _loss, direction_masks

    @staticmethod
    def _dot_cos(x, y):
        axis = range(1, len(x.shape))
        a = tf.nn.l2_normalize(x, axis=axis)
        b = tf.nn.l2_normalize(y, axis=axis)
        cos = tf.maximum(tf.reduce_sum(a * b, axis=axis), 1e-1) ** 2
        dot = tf.reduce_sum(x * y)
        return dot * cos

    @staticmethod
    def _default_transform(size):
        from omnixai.preprocessing.pipeline import Pipeline
        from .preprocess import RandomBlur, RandomCrop, \
            RandomResize, RandomFlip, Padding

        unit = max(int(size / 16), 1)
        pipeline = Pipeline() \
            .step(Padding(size=unit * 4)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomResize((0.92, 0.96))) \
            .step(RandomBlur(kernel_size=9, sigma=(1.0, 1.1))) \
            .step(RandomCrop(unit)) \
            .step(RandomCrop(unit)) \
            .step(RandomFlip())
        return pipeline

    @staticmethod
    def _normalize(x, normalizer, min_value, max_value):
        x = tf.nn.sigmoid(x) if normalizer == "sigmoid" \
            else tf.clip_by_value(x, min_value, max_value)
        x = x - tf.reduce_min(x, (1, 2, 3), keepdims=True)
        x = x / (tf.reduce_max(x, (1, 2, 3), keepdims=True) + 1e-6)
        return x * (max_value - min_value) + min_value

    @staticmethod
    def _regularize(reg_type, weight):
        if reg_type is None or reg_type == "":
            return lambda x: 0
        elif reg_type == "l1":
            return lambda x: tf.reduce_mean(tf.abs(x), (1, 2, 3)) * weight
        elif reg_type == "l2":
            return lambda x: tf.sqrt(tf.reduce_mean(x ** 2, (1, 2, 3))) * weight
        elif reg_type == "tv":
            return lambda x: tf.image.total_variation(x) * weight
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

    def optimize(
            self,
            num_iterations=200,
            learning_rate=0.05,
            transformer=None,
            regularizers=None,
            pixel_value_normalizer="sigmoid",
            pixel_value_range=(0, 1),
            image_shape=None,
            save_all_images=False,
            verbose=False
    ):
        from omnixai.utils.misc import ProgressBar
        bar = ProgressBar(num_iterations) if verbose else None

        model, objective_func, input_shape = self._build_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        shape = input_shape if image_shape is None \
            else (input_shape[0], *image_shape, input_shape[-1])
        if transformer is None:
            transformer = self._default_transform(min(shape[1], shape[2]))
        if regularizers is not None:
            if not isinstance(regularizers, list):
                regularizers = [regularizers]
            regularizers = [self._regularize(reg, w) for reg, w in regularizers]

        inputs = tf.Variable(
            tf.random.normal(shape, stddev=0.01, dtype=tf.float32),
            trainable=True
        )
        normalize = lambda x: self._normalize(
            x=x,
            normalizer=pixel_value_normalizer,
            min_value=pixel_value_range[0],
            max_value=pixel_value_range[1]
        )

        @tf.function
        def step(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                images = transformer.transform(normalize(x))
                images = tf.image.resize(images, (input_shape[1], input_shape[2]))
                outputs = model(images)
                if len(model.outputs) == 1:
                    outputs = tf.expand_dims(outputs, 0)
                loss = objective_func(outputs)
                if regularizers is not None:
                    for func in regularizers:
                        loss -= func(images)
                return tape.gradient(loss, x)

        results = []
        for i in range(num_iterations):
            grads = step(inputs)
            optimizer.apply_gradients([(-grads, inputs)])
            if save_all_images or i == num_iterations - 1:
                results.append(normalize(inputs).numpy())
            if verbose:
                bar.print(i + 1, prefix=f"Step: {i + 1}", suffix="")
        return results
