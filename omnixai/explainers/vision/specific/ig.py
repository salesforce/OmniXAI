#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The integrated-gradient explainer for vision tasks.
"""
import warnings
import numpy as np
from typing import Callable

from ...base import ExplainerBase
from ...tabular.specific.ig import IntegratedGradient
from ....data.image import Image
from ....explanations.image.pixel_importance import PixelImportance
from .utils import GradMixin


class IntegratedGradientImage(ExplainerBase, IntegratedGradient, GradMixin):
    """
    The integrated-gradient explainer for vision tasks.
    If using this explainer, please cite the original work: https://github.com/ankurtaly/Integrated-Gradients.
    """

    explanation_type = "local"
    alias = ["ig", "integrated_gradient"]

    def __init__(
        self,
        model,
        preprocess_function: Callable,
        mode: str = "classification",
        background_data: Image = Image(),
        **kwargs,
    ):
        """
        :param model: The model to explain, whose type can be `tf.keras.Model` or `torch.nn.Module`.
        :param preprocess_function: The pre-processing function that converts the raw input features
            into the inputs of ``model``.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param background_data: The background images to compare with. When ``background_data``
            is empty, the baselines for computing integrated gradients will be sampled randomly.
        :param kwargs: Additional parameters to initialize the IG explainer,
            e.g., ``num_random_trials`` -- the number of trials in generating baselines.
        """
        super().__init__()
        self.mode = mode
        assert isinstance(background_data, Image), "`background_data` should be an instance of Image."

        self.model = model
        if preprocess_function is None:
            warnings.warn("The preprocessing function is None. " "Please check whether this setup is correct.")
        self.preprocess_function = preprocess_function

        self.data = background_data
        if self.data.values is not None:
            self.baselines = self._sample_baseline(num_random_trials=kwargs.get("num_random_trials", -1))
        else:
            self.baselines = None

    def _sample_baseline(self, num_random_trials=10, height=224, width=224, channels=3) -> Image:
        """
        Constructs the baselines for computing the integrated gradients.

        :param num_random_trials: The number of trials for randomly sampling
            instances as the baselines.
        :return: The baseline images.
        :rtype: Image
        """
        if self.data.values is None:
            num_random_trials = max(num_random_trials, 10)
            return Image(
                data=np.random.random([num_random_trials, height, width, channels]) * 255,
                batched=True,
                channel_last=True,
            )
        else:
            if num_random_trials > 0:
                replace = self.data.shape[0] < num_random_trials
                indices = np.random.choice(self.data.shape[0], size=num_random_trials, replace=replace)
                return self.data[indices]
            else:
                return Image(data=np.mean(self.data.to_numpy(), axis=0, keepdims=True), batched=True, channel_last=True)

    def _predict(self, inputs):
        """
        Predicts class labels in classification.

        :param inputs: The input instances.
        :return: The predicted labels.
        """
        try:
            import torch

            self.model.eval()
            param = next(self.model.parameters())
            X = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.get_default_dtype())
            scores = self.model(X.to(param.device)).detach().cpu().numpy()
        except:
            scores = self.model(inputs).numpy()
        y = np.argmax(scores, axis=1).astype(int)
        return y

    def explain(self, X: Image, y=None, baseline=None, **kwargs) -> PixelImportance:
        """
        Generates the pixel-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when ``y = None``.
        :param baseline: The baselines for computing integrated gradients. When it is `None`,
            the baselines will be sampled randomly.
        :param kwargs: Additional parameters, e.g., ``steps`` for
            `IntegratedGradient.compute_integrated_gradients`.
        :return: The explanations for all the instances, e.g., pixel importance scores.
        """
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        explanations = PixelImportance(self.mode)

        baselines = self.baselines if baseline is None else baseline
        if baselines is None:
            baselines = self._sample_baseline(
                num_random_trials=10, height=X.shape[1], width=X.shape[2], channels=X.shape[3]
            )
        if self.preprocess_function is not None:
            inputs = self.preprocess_function(X)
            if not isinstance(inputs, np.ndarray):
                try:
                    inputs = inputs.detach().cpu().numpy()
                except:
                    inputs = inputs.numpy()
            baselines = self.preprocess_function(baselines)
            if not isinstance(baselines, np.ndarray):
                try:
                    baselines = baselines.detach().cpu().numpy()
                except:
                    baselines = baselines.numpy()
        else:
            inputs = X.to_numpy()
            baselines = baselines.to_numpy()

        steps = kwargs.get("steps", 50)
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
                y = self._predict(inputs)

        for i, instance in enumerate(inputs):
            output_index = y[i] if y is not None else None
            all_gradients = []
            for baseline in baselines:
                integrated_grads = self.compute_integrated_gradients(
                    model=self.model, inp=instance, baseline=baseline, output_index=output_index, steps=steps
                )
                all_gradients.append(integrated_grads)
            scores = np.average(np.array(all_gradients), axis=0).squeeze()
            if scores.ndim == 3 and scores.shape[0] == 3:
                scores = np.transpose(scores, (1, 2, 0))
            explanations.add(
                image=self._resize(self.preprocess_function, X[i]).to_numpy()[0],
                target_label=output_index,
                importance_scores=scores
            )
        return explanations
