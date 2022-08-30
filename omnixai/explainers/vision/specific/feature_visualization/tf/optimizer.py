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

        def _loss(output, **kwargs):
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
        pipeline = Pipeline()\
            .step(Padding(size=unit * 4))\
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 2)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomCrop(unit * 4)) \
            .step(RandomResize((0.92, 0.96))) \
            .step(RandomBlur(kernel_size=9, sigma=(1.0, 1.1))) \
            .step(RandomCrop(unit)) \
            .step(RandomCrop(unit)) \
            .step(RandomFlip())
        return pipeline

    def optimize(
            self,
            num_iterations=200,
            learning_rate=0.05,
            transforms=None,
            regularizers=None,
            pixel_normalizer="sigmoid",
            pixel_range=(0, 1),
            image_shape=None,
    ):
        model, objective_func, input_shape = self._build_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        image_shape = input_shape if image_shape is None \
            else (input_shape[0], *image_shape, input_shape[-1])
        if transforms is None:
            transforms = self._default_transform(min(image_shape[1], image_shape[2]))
