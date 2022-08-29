#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import tensorflow as tf
from typing import Union, List
from dataclasses import dataclass


@dataclass
class Objective:
    layer: tf.keras.layers.Layer
    loss: str = "square"
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
        pass

    @staticmethod
    def _layer_loss(objective):
        shape = objective.layer.output.shape
        layer_masks = np.ones((1, *shape[1:]))

        def _loss(output, **kwargs):
            if objective.loss == "square":
                return tf.reduce_mean(output ** 2)
            else:
                return tf.reduce_mean(output)

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
