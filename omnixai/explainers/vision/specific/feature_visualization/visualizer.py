#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The feature visualizer for vision models.
"""
import numpy as np
from typing import Dict, List, Union, Tuple, Callable
from ....base import ExplainerBase
from .....data.image import Image
from .....preprocessing.pipeline import Pipeline
from .....explanations.image.plain import PlainExplanation
from .....utils.misc import is_torch_available, is_tf_available


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
        """
        :param model: The model to explain.
        :param objectives: A list of objectives for visualization. Each objective has the following format:
            `{"layer": layer, "weight": 1.0, "type": "layer", "channel", "neuron" or "direction", "index": channel_idx,
            neuron_idx or direction_vector}`. For example, `{"layer": layer, "weight": 1.0, "type": channel,
            "index": [0, 1, 2]}`. Here, "layer" indicates the target layer and "type" is the objective type.
            If "type" is "channel" or "neuron", please set the channel indices or neuron indices. If "type" is
            "direction", please set the direction vector who shape is the same as the layer output shape
            (without batch-size dimension).
        """
        super().__init__()
        self.model = model
        self.objectives = self._check_objectives(objectives)

    @staticmethod
    def _check_objectives(objectives):
        from .utils import Objective
        if not isinstance(objectives, (list, tuple)):
            objectives = [objectives]

        objs = []
        for obj in objectives:
            assert "layer" in obj, \
                "Please set the target layer, e.g., 'layer': target_layer"
            assert "type" in obj, \
                "Please set the objective type, e.g., 'layer', 'channel', 'neuron' or 'direction'"
            assert obj["type"] in ["layer", "channel", "neuron", "direction"], \
                "Please choose from 'layer', 'channel', 'neuron' or 'direction'."
            if obj["type"] in ["channel", "neuron", "direction"]:
                assert "index" in obj, \
                    "Please set the index, e.g., 'index': [0, 1, 2]."

            if obj["type"] == "layer":
                objs.append(Objective(
                    layer=obj["layer"],
                    weight=obj.get("weight", 1.0)
                ))
            elif obj["type"] == "channel":
                objs.append(Objective(
                    layer=obj["layer"],
                    channel_indices=obj["index"],
                    weight=obj.get("weight", 1.0)
                ))
            elif obj["type"] == "neuron":
                objs.append(Objective(
                    layer=obj["layer"],
                    neuron_indices=obj["index"],
                    weight=obj.get("weight", 1.0)
                ))
            else:
                objs.append(Objective(
                    layer=obj["layer"],
                    direction_vectors=obj["index"],
                    weight=obj.get("weight", 1.0)
                ))
        return objs

    def explain(
            self,
            *,
            num_iterations: int = 300,
            learning_rate: float = 0.05,
            transformers: Pipeline = None,
            regularizers: List = None,
            image_shape: Tuple = None,
            use_fft=False,
            fft_decay=1.0,
            normal_color: bool = False,
            verbose: bool = True,
            **kwargs
    ):
        """
        Generates feature visualizations for the specified model and objectives.

        :param num_iterations: The number of iterations during optimization.
        :param learning_rate: The learning rate during optimization.
        :param transformers: The transformations applied on images during optimization.
            `transformers` is an object of `Pipeline` defined in the `preprocessing` package.
            The available transform functions can be found in `.pytorch.preprocess` and
            `.tf.preprocess`. When `transformers` is None, a default transformation will be applied.
        :param regularizers: A list of regularizers applied on images. Each regularizer is a tupe
            `(regularizer_type, weight)` where `regularizer_type` is "l1", "l2" or "tv".
        :param image_shape: The customized image shape. If None, the default shape is (224, 224).
        :param use_fft: Whether to use fourier preconditioning.
        :param fft_decay: The value controlling the allowed energy of the high frequency.
        :param normal_color: Whether to map uncorrelated colors to normal colors.
        :param verbose: Whether to print the optimization progress.
        :return: The optimized images for the objectives.
        """
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and TensorFlow cannot be found.")
        explanations = PlainExplanation()

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
            use_fft=use_fft,
            fft_decay=fft_decay,
            normal_color=normal_color,
            save_all_images=False,
            verbose=verbose
        )
        images = Image(
            data=results[-1] * 255,
            batched=True,
            channel_last=model_type == "tf"
        )
        new_names = []
        for labels in names:
            new_names.append("|".join([
                f"{label['layer_name']}_{label['type']}" if label['type'] in ["layer", "direction"]
                else f"{label['layer_name']}_{label['type']}_{label['index']}"
                for label in labels])
            )
        images = images.to_pil()
        if not isinstance(images, list):
            images = [images]
        explanations.add(images, new_names)
        return explanations


class FeatureMapVisualizer(ExplainerBase):
    """
    The class for feature map visualization.
    """
    explanation_type = "local"
    alias = ["fm", "feature_map"]

    def __init__(
            self,
            model,
            target_layer,
            preprocess_function: Callable,
            **kwargs,
    ):
        """
        :param model: The model to explain.
        :param target_layer: The target layer for feature map visualization.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        self.model = model
        self.layer = target_layer
        self.preprocess = preprocess_function

        extractor = None
        if is_torch_available():
            import torch.nn as nn

            if isinstance(self.model, nn.Module):
                from .pytorch.feature_maps import FeatureMapExtractor
                extractor = FeatureMapExtractor(self.model, self.layer)

        if extractor is None and is_tf_available():
            import tensorflow as tf

            if isinstance(self.model, tf.keras.Model):
                from .tf.feature_maps import FeatureMapExtractor
                extractor = FeatureMapExtractor(self.model, self.layer)

        if extractor is None:
            raise TypeError(
                f"The model ({type(self.model)}) is neither a PyTorch model nor a TensorFlow model.")
        self.extractor = extractor

    @staticmethod
    def _normalize(x):
        if len(x.shape) == 3:
            min_val = x.min(axis=(0, 1), keepdims=True)
            max_val = x.max(axis=(0, 1), keepdims=True)
            x = (x - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = x.min()
            max_val = x.max()
            x = (x - min_val) / (max_val - min_val + 1e-8)
        return (x * 255).astype(int)

    @staticmethod
    def _resize(x, min_size=20):
        if min(x.shape[0], x.shape[1]) >= min_size:
            return x
        else:
            from omnixai.preprocessing.image import Resize
            im = Image(x, batched=False)
            im = Resize(min_size).transform(im)
            return im.to_numpy(keepdim=False)[0]

    def explain(self, X: Image, **kwargs):
        """
        Generates feature map visualizations for the specified layer and inputs.

        :param X: A batch of input images.
        :return: The feature maps.
        """
        explanations = PlainExplanation()
        inputs = self.preprocess(X) if self.preprocess is not None else X.to_numpy()
        outputs = self.extractor.extract(inputs)
        if len(outputs.shape) <= 2:
            raise RuntimeError("The dimension of the layer outputs <= 2. Please try a different layer.")

        image_width, pad = 512, 1
        for feature_map in outputs:
            feature_map = self._normalize(feature_map)
            if len(feature_map.shape) == 2:
                feature_map = self._resize(feature_map)
                image = Image(feature_map, batched=False)
            else:
                x = self._resize(feature_map[..., 0])
                h = x.shape[0] + pad * 2
                w = x.shape[1] + pad * 2
                num_cols = image_width // w
                num_rows = int(np.ceil(feature_map.shape[-1] / num_cols))
                image = np.zeros((h * num_rows, w * num_cols), dtype=int)
                for i in range(feature_map.shape[-1]):
                    x = self._resize(feature_map[..., i])
                    x = np.pad(x, (pad, pad))
                    r, c = divmod(i, num_cols)
                    image[r * h: (r + 1) * h, c * w: (c + 1) * w] = x
                image = Image(image, batched=False)

            image = image.to_pil()
            if not isinstance(image, list):
                image = [image]
            explanations.add(image)
        return explanations
