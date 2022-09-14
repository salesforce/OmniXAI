#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from omnixai.utils.misc import is_tf_available, is_torch_available
from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image
from omnixai.explanations.image.pixel_importance import PixelImportance
from .utils import smooth_grad


class SmoothGrad(ExplainerBase):
    """
    The Smooth-Grad method for generating visual explanations.
    If using this explainer, please cite `SmoothGrad: removing noise by adding noise,
    Smilkov et al., https://arxiv.org/abs/1706.03825`.
    """

    explanation_type = "local"
    alias = ["smoothgrad", "smooth-grad"]

    def __init__(
            self,
            model,
            preprocess_function: Callable,
            mode: str = "classification",
            **kwargs
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        self.model = model
        self.preprocess_function = preprocess_function
        self.mode = mode

    def _resize(self, image):
        """
        Rescales the raw input image to the input size of the model.

        :param image: The raw input image.
        :return: The resized image.
        """
        assert image.shape[0] == 1, "`image` can contain one instance only."
        if self.preprocess_function is None:
            return image

        y = image.to_numpy()
        x = self.preprocess_function(image)
        if not isinstance(x, np.ndarray):
            try:
                x = x.detach().cpu().numpy()
            except:
                x = x.numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)

    def explain(self, X: Image, y=None, num_samples=50, sigma=0.1, **kwargs):
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param num_samples: The number of images used to compute smooth gradients.
        :param sigma: The sigma for calculating standard deviation of noise.
        :param kwargs: Additional parameters.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        :rtype: PixelImportance
        """
        explanations = PixelImportance(self.mode)

        gradients, y = smooth_grad(
            X=X,
            y=y,
            model=self.model,
            preprocess_function=self.preprocess_function,
            mode=self.mode,
            num_samples=num_samples,
            sigma=sigma
        )
        for i in range(len(X)):
            label = y[i] if y is not None else None
            explanations.add(
                image=self._resize(X[i]).to_numpy()[0],
                target_label=label,
                importance_scores=gradients[i]
            )
        return explanations
