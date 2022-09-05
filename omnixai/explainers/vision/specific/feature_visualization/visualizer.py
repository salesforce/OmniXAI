#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The feature visualizer for vision models.
"""
from typing import Dict, List, Union, Tuple
from ....base import ExplainerBase
from .....data.image import Image
from .....preprocessing.pipeline import Pipeline


class FeatureVisualizer(ExplainerBase):
    """
    Feature visualization for vision models. The input of the model has shape (B, C, H, W)
    for PyTorch and (B, H, W, C) for TensorFlow. This class applies the optimized based method
    for visualizing layer, channel, neuron features. For more details, please visit
    `https://distill.pub/2017/feature-visualization/`.
    """
    explanation_type = "global"
    alias = ["fv", "feature_visualization"]

    def __init__(
            self,
            model,
            objectives: Union[Dict, List],
            **kwargs,
    ):
        super().__init__()
        self.model = model
        self.objectives = self._check_objectives(objectives)

    @staticmethod
    def _check_objectives(objectives):
        pass

    def explain(
            self,
            *,
            num_iterations: int = 300,
            learning_rate: float = 0.05,
            transformers: Pipeline = None,
            regularizers: List = None,
            image_shape: Tuple = None,
            normal_color: bool = False,
            verbose: bool = True,
            **kwargs
    ):
        """
        Generates feature visulizations for the specified model and objectives.

        :param num_iterations: The number of iterations during optimization.
        :param learning_rate: The learning rate during optimization.
        :param transformers: The transformations applied on images during optimization.
            `transformers` is an object of `Pipeline` defined in the `preprocessing` package.
            The available transform functions can be found in `.pytorch.preprocess` and
            `.tf.preprocess`. When `transformers` is None, a default transformation will be applied.
        :param regularizers: A list of regularizers applied on images. Each regularizer is a tupe
            `(regularizer_type, weight)` where `regularizer_type` is "l1", "l2" or "tv".
        :param image_shape: The customized image shape. If None, the default shape is (224, 224).
        :param normal_color: Whether to map uncorrelated colors to normal colors.
        :param verbose: Whether to print the optimization progress.
        :return: The optimized images for the objectives.
        """
        from .....utils.misc import is_torch_available, is_tf_available
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and TensorFlow cannot be found.")

        value_normalizer = kwargs.get("value_normalizer", "sigmoid")
        value_range = kwargs.get("value_range", (0.05, 0.95))
        init_std = kwargs.get("init_std", 0.01)

        optimizer, model_type = None, None
        if is_torch_available():
            import torch.nn as nn

            if isinstance(self.model, nn.Module):
                from .pytorch.optimizer import FeatureOptimizer
                optimizer = FeatureOptimizer(self.model, self.objectives)
                model_type = "torch"

        if optimizer is None and is_tf_available():
            import tensorflow as tf

            if isinstance(self.model, tf.keras.Model):
                from .tf.optimizer import FeatureOptimizer
                optimizer = FeatureOptimizer(self.model, self.objectives)
                model_type = "tf"

        if optimizer is None:
            raise TypeError(
                f"The model ({type(self.model)}) is neither a PyTorch model nor a TensorFlow model.")

        results, names = optimizer.optimize(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            transformers=transformers,
            regularizers=regularizers,
            image_shape=image_shape,
            value_normalizer=value_normalizer,
            value_range=value_range,
            init_std=init_std,
            normal_color=normal_color,
            save_all_images=False,
            verbose=verbose
        )
        results = Image(
            data=results[-1] * 255,
            batched=True,
            channel_last=model_type == "tf"
        )
