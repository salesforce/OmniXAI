#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The counterfactual explainer for image classification.
"""
import numpy as np
from typing import Callable

from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image
from omnixai.explainers.tabular.counterfactual.ce import CounterfactualOptimizer
from omnixai.explanations.image.counterfactual import CFExplanation
from omnixai.utils.misc import is_torch_available, is_tf_available

if is_torch_available():
    import torch
    import torch.nn as nn

if is_tf_available():
    import tensorflow as tf


class CounterfactualExplainer(ExplainerBase):
    """
    The counterfactual explainer for image classification.
    If using this explainer, please cite the paper `Counterfactual Explanations without
    Opening the Black Box: Automated Decisions and the GDPR, Sandra Wachter, Brent Mittelstadt, Chris Russell,
    https://arxiv.org/abs/1711.00399`.
    """

    explanation_type = "local"
    alias = ["ce", "counterfactual"]

    def __init__(
        self,
        model,
        preprocess_function: Callable,
        mode: str = "classification",
        c=10.0,
        kappa=10.0,
        binary_search_steps=5,
        learning_rate=1e-2,
        num_iterations=100,
        grad_clip=1e3,
        **kwargs,
    ):
        """
        :param model: The classification model which can be `torch.nn.Module` or `tf.keras.Model`.
        :param preprocess_function: The preprocessing function that converts the raw data
            into the inputs of ``model``.
        :param mode: It can be `classification` only.
        :param c: The weight of the hinge loss term.
        :param kappa: The parameter in the hinge loss function.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        :param kwargs: Not used.
        """
        super().__init__()
        assert mode == "classification", "CE supports classification tasks only."

        model_type = None
        if is_tf_available():
            if isinstance(model, tf.keras.Model):
                model_type = "tf"
        if model_type is None and is_torch_available():
            if isinstance(model, nn.Module):
                model_type = "torch"
        if model_type is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.model = model
        self.preprocess_function = preprocess_function
        self.create_optimizer = lambda x, y, m: CounterfactualOptimizer(
            x,
            y,
            m,
            c=c,
            kappa=kappa,
            binary_search_steps=binary_search_steps,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            grad_clip=grad_clip,
        )

    def _preprocess(self, inputs: Image):
        """
        Does preprocessing on the input images.

        :return: The processed images.
        :rtype: np.ndarray
        """
        if inputs.values is None:
            return None
        if self.preprocess_function is not None:
            inputs = self.preprocess_function(inputs)
            if not isinstance(inputs, np.ndarray):
                try:
                    inputs = inputs.detach().cpu().numpy()
                except AttributeError:
                    inputs = inputs.numpy()
        else:
            inputs = inputs.to_numpy()
        return inputs

    def _predict(self, inputs):
        """
        Predicts class labels in classification.

        :param inputs: The input instances.
        :return: The predicted labels.
        :rtype: np.ndarray
        """
        try:
            import torch

            self.model.eval()
            param = next(self.model.parameters())
            x = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.get_default_dtype())
            scores = self.model(x.to(param.device)).detach().cpu().numpy()
        except:
            scores = self.model(inputs).numpy()
        y = np.argmax(scores, axis=1).astype(int)
        return y

    def explain(self, X: Image, **kwargs) -> CFExplanation:
        """
        Generates the counterfactual explanations for the input images.
        Note that the returned results including the original input images and the
        counterfactual images have been processed by the ``preprocess_function``,
        e.g., if the ``preprocess_function`` rescales [0, 255] to [0, 1], the return
        results will have range [0, 1].

        :param X: A batch of the input images.
        :return: The counterfactual explanations for all the images, e.g., counterfactual images.
        """
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        verbose = kwargs.get("kwargs", True)
        explanations = CFExplanation()
        y = self._predict(self._preprocess(X))

        for i in range(len(X)):
            x = self._preprocess(X[i])
            optimizer = self.create_optimizer(x=x, y=y[i], m=self.model)
            # Original image
            x = x.squeeze()
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))

            # Get the counterfactual example
            cf = optimizer.optimize(verbose=verbose)
            if cf is not None:
                cf_label = self._predict(cf)[0]
                cf = cf.squeeze()
                if cf.ndim == 3 and cf.shape[0] == 3:
                    cf = np.transpose(cf, (1, 2, 0))
            else:
                cf_label = None
            explanations.add(image=x, label=y[i], cf=cf, cf_label=cf_label)
        return explanations
