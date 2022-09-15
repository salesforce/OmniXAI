#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Callable
from omnixai.utils.misc import is_tf_available, is_torch_available
from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image
from omnixai.explanations.image.pixel_importance import PixelImportance
from .utils import GradMixin, smooth_grad, guided_bp


class SmoothGrad(ExplainerBase, GradMixin):
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
            use_guided_bp: bool = False,
            **kwargs
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param use_guided_bp: Whether to use guided back propagation when computing gradients.
        """
        super().__init__()
        if not is_tf_available() and not is_torch_available():
            raise EnvironmentError("Both Torch and Tensorflow cannot be found.")

        self.model = model
        self.preprocess_function = preprocess_function
        self.mode = mode
        self.use_guided_bp = use_guided_bp

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

        if not self.use_guided_bp:
            gradients, y = smooth_grad(
                X=X,
                y=y,
                model=self.model,
                preprocess_function=self.preprocess_function,
                mode=self.mode,
                num_samples=num_samples,
                sigma=sigma
            )
        else:
            gradients, y = guided_bp(
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
                image=self._resize(self.preprocess_function, X[i]).to_numpy()[0],
                target_label=label,
                importance_scores=gradients[i]
            )
        return explanations
