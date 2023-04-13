#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for vision tasks.
"""
import shap
import warnings
import numpy as np

from ...base import ExplainerBase
from ....data.image import Image
from ....explanations.image.pixel_importance import PixelImportance


class ShapImage(ExplainerBase):
    """
    The SHAP explainer for vision tasks.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "local"
    alias = ["shap"]

    def __init__(
        self, model, preprocess_function, mode: str = "classification", background_data: Image = Image(), **kwargs
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw input features
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param background_data: The background images to compare with.
        """
        super().__init__()
        self.mode = mode
        assert isinstance(background_data, Image), "`background_data` should be an instance of Image."

        self.data = background_data
        self.model_type = self._get_model_type(model)
        self.model = model
        if preprocess_function is None:
            warnings.warn("The preprocessing function is None. " "Please check whether this setup is correct.")
        self.preprocess_function = preprocess_function

    @staticmethod
    def _get_model_type(model):
        """
        Gets the model type, e.g., `tf.keras.Model` or `torch.nn.Module`.

        :return: The model type.
        :rtype: str
        """
        from ....utils.misc import is_tf_available, is_torch_available

        torch_ok, tf_ok = is_torch_available(), is_tf_available()
        if not torch_ok and not tf_ok:
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        model_type = None
        if torch_ok:
            import torch.nn as nn

            if isinstance(model, nn.Module):
                model_type = "torch"
                model.eval()
        if model_type is None and tf_ok:
            import tensorflow as tf

            if isinstance(model, tf.keras.Model):
                model_type = "tf"
        if model_type is None:
            raise ValueError(
                f"`predict_function` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}"
            )
        return model_type

    def _preprocess(self, x: Image, **kwargs):
        """
        Converts the raw inputs into the inputs of the model.

        :param x: The raw input images.
        :return: The processed inputs.
        """
        if self.preprocess_function is not None:
            inputs = self.preprocess_function(x)
        else:
            if self.model_type == "torch":
                import torch

                inputs = torch.tensor(x.to_numpy(), dtype=torch.get_default_dtype())
            else:
                inputs = x.to_numpy()
        if self.model_type == "torch":
            return inputs.to(next(self.model.parameters()).device)
        else:
            return inputs

    def _resize(self, image):
        """
        Scales the raw input image to the input size of the model.

        :param image: The raw input image.
        :return: The resized image.
        """
        assert image.shape[0] == 1, "`image` can contain one instance only."
        if self.preprocess_function is None:
            return image

        y = image.to_numpy()
        x = self.preprocess_function(image)
        if not isinstance(x, np.ndarray):
            x = x.numpy() if self.model_type == "tf" else x.detach().cpu().numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)

    def explain(self, X: Image, y=None, **kwargs) -> PixelImportance:
        """
        Generates the pixel-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Additional parameters, e.g., ``nsamples`` -- the maximum number of images
            sampled for the background.
        :return: The explanations for all the input instances, e.g., pixel importance scores.
        """
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode)
        inputs = self._preprocess(X, **kwargs)

        if self.data.values is not None:
            background = self.data[
                np.random.choice(
                    self.data.shape[0], min(kwargs.get("nsamples", 100), self.data.shape[0]), replace=False
                )
            ]
            background = self._preprocess(background, **kwargs)
        else:
            background = Image(
                data=np.ones((1,) + X.shape[1:]) * int(np.mean(X.to_numpy())), batched=True, channel_last=True
            )
            background = self._preprocess(background, **kwargs)

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
                scores = (
                    self.model(inputs).numpy() if self.model_type == "tf" else self.model(inputs).detach().cpu().numpy()
                )
                y = np.argmax(scores, axis=1).astype(int)

        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(inputs)

        for i, image in enumerate(X):
            label, scores = (y[i], shap_values[y[i]][i]) if self.mode == "classification" else (None, shap_values[i])
            scores = scores.squeeze()
            if scores.ndim == 3 and scores.shape[0] == 3:
                scores = np.transpose(scores, (1, 2, 0))
            explanations.add(image=self._resize(image).to_numpy()[0], target_label=label, importance_scores=scores)
        return explanations
