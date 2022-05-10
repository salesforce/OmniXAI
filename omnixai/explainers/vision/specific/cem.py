#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The contrastive explainer for image classification.
"""
import numpy as np
from typing import Callable

from omnixai.explainers.base import ExplainerBase
from omnixai.data.image import Image
from omnixai.explanations.image.contrast import ContrastiveExplanation
from omnixai.utils.misc import is_torch_available, is_tf_available, ProgressBar


if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.autograd import grad

    class _GFuncTorch(nn.Module):
        def __init__(self, x0, target, model, mode, c, gamma, kappa, ae):
            super().__init__()
            self.x0 = x0
            self.target = target
            self.model = model
            self.mode = mode
            self.c = c
            self.gamma = gamma
            self.kappa = kappa
            self.ae = ae
            self.reduce_dims = list(range(1, len(x0.shape)))

        def forward(self, x):
            # Regularization terms
            if self.mode == "pn":
                regularization = torch.sum(torch.square(self.x0 - x), dim=self.reduce_dims)
            else:
                regularization = torch.sum(x * x, dim=self.reduce_dims)
            if self.ae is not None:
                regularization += torch.sum(torch.square(x - self.ae(x)), dim=self.reduce_dims) * self.gamma
            # Loss function
            prob = self.model(x)
            a = torch.sum(prob * self.target, dim=1)
            b = torch.max((1 - self.target) * prob - self.target * 10000, dim=1)[0]
            if self.mode == "pn":
                loss = nn.functional.relu(a - b + self.kappa)
            else:
                loss = nn.functional.relu(b - a + self.kappa)
            return torch.mean(self.c * loss + regularization)


if is_tf_available():
    import tensorflow as tf

    class _GFuncTF(tf.keras.Model):
        def __init__(self, x0, target, model, mode, c, gamma, kappa, ae):
            super().__init__()
            self.x0 = x0
            self.target = target
            self.model = model
            self.mode = mode
            self.c = c
            self.gamma = gamma
            self.kappa = kappa
            self.ae = ae
            self.reduce_dims = list(range(1, len(x0.shape)))

        def call(self, x):
            # Regularization terms
            if self.mode == "pn":
                regularization = tf.reduce_sum(tf.square(self.x0 - x), axis=self.reduce_dims)
            else:
                regularization = tf.reduce_sum(tf.square(x), axis=self.reduce_dims)
            if self.ae is not None:
                regularization += tf.reduce_sum(tf.square(x - self.ae(x)), axis=self.reduce_dims) * self.gamma
            # Loss function
            prob = self.model(x)
            a = tf.reduce_sum(prob * self.target, axis=1)
            b = tf.reduce_max((1 - self.target) * prob - self.target * 10000, axis=1)
            if self.mode == "pn":
                loss = tf.maximum(0.0, a - b + self.kappa)
            else:
                loss = tf.maximum(0.0, b - a + self.kappa)
            return tf.reduce_mean(self.c * loss + regularization)


class CEMOptimizer:
    """
    The optimizer for contrastive explanation. The module is implemented based
    on the paper: https://arxiv.org/abs/1802.07623.
    """

    def __init__(
        self,
        x0,
        target,
        model,
        c=10.0,
        beta=0.1,
        gamma=0.0,
        kappa=10.0,
        ae_model=None,
        binary_search_steps=5,
        learning_rate=1e-2,
        num_iterations=1000,
        grad_clip=1e3,
        background_data=None,
    ):
        """
        :param x0: The input image.
        :param target: The predicted label of the input image.
        :param model: The classification model which can be `torch.nn.Module` or `tf.keras.Model`.
        :param c: The weight of the loss term.
        :param beta: The weight of the L1 regularization term.
        :param gamma: The weight of the AE regularization term.
        :param kappa: The parameter in the hinge loss function.
        :param ae_model: The auto-encoder model used for regularization.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        :param background_data: Sampled images for estimating background values.
        """
        assert x0.shape[0] == 1
        if not isinstance(x0, np.ndarray):
            try:
                x0 = x0.detach().cpu().numpy()
            except AttributeError:
                x0 = x0.numpy()

        self.x0 = x0
        self.target = target
        self.model = model
        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.ae_model = ae_model
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.grad_clip = grad_clip
        self.background_data = background_data
        self.bounds = (np.min(x0), np.max(x0))

        self.model_type = None
        if is_torch_available():
            if isinstance(self.model, nn.Module):
                self.model_type = "torch"
        if self.model_type is None and is_tf_available():
            if isinstance(model, tf.keras.Model):
                self.model_type = "tf"
        if self.model_type is None:
            raise ValueError(f"`model` should be a tf.keras.Model " f"or a torch.nn.Module instead of {type(model)}")

        self.num_classes = self._predict(self.x0).shape[1]

    def _init_functions(self, c):
        if self.model_type == "torch":
            param = next(self.model.parameters())
            inputs = torch.tensor(self.x0, dtype=torch.get_default_dtype()).to(param.device)
            target = torch.tensor(np.eye(1, self.num_classes, self.target), dtype=torch.get_default_dtype()).to(
                param.device
            )
            self.pn_g_func = _GFuncTorch(
                inputs, target, self.model, mode="pn", c=c, gamma=self.gamma, kappa=self.kappa, ae=self.ae_model
            )
            self.pp_g_func = _GFuncTorch(
                inputs, target, self.model, mode="pp", c=c, gamma=self.gamma, kappa=self.kappa, ae=self.ae_model
            )
        else:
            inputs = tf.convert_to_tensor(self.x0, dtype=tf.keras.backend.floatx())
            target = tf.convert_to_tensor(np.eye(1, self.num_classes, self.target), dtype=tf.keras.backend.floatx())
            self.pn_g_func = _GFuncTF(
                inputs, target, self.model, mode="pn", c=c, gamma=self.gamma, kappa=self.kappa, ae=self.ae_model
            )
            self.pp_g_func = _GFuncTF(
                inputs, target, self.model, mode="pp", c=c, gamma=self.gamma, kappa=self.kappa, ae=self.ae_model
            )

    def _predict(self, inputs):
        if self.model_type == "tf":
            inputs = tf.convert_to_tensor(inputs, dtype=tf.keras.backend.floatx())
            return self.model(inputs).numpy()
        elif self.model_type == "torch":
            self.model.eval()
            param = next(self.model.parameters())
            inputs = torch.tensor(inputs, dtype=torch.get_default_dtype()).to(param.device)
            return self.model(inputs).detach().cpu().numpy()
        else:
            return self.model(inputs)

    def _compute_gradient(self, model, inputs):
        if self.model_type == "tf":
            inputs = tf.convert_to_tensor(inputs, dtype=tf.keras.backend.floatx())
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                predictions = model(inputs)
                gradients = tape.gradient(predictions, inputs).numpy()

        elif self.model_type == "torch":
            model.eval()
            param = next(model.parameters())
            inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.get_default_dtype()).to(param.device)
            predictions = model(inputs)
            gradients = (
                grad(outputs=predictions, inputs=inputs, grad_outputs=torch.ones_like(predictions).to(param.device))[0]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            raise NotImplementedError
        gradients = np.maximum(np.minimum(gradients, self.grad_clip), -self.grad_clip)
        return gradients

    def _learning_rate(self, i):
        return self.learning_rate * (1 - i / self.num_iterations) ** 0.5

    @staticmethod
    def _update_const(c, c_lb, c_ub, sol):
        if sol is not None:
            c_ub = min(c_ub, c)
            if c_ub < 1e9:
                c = (c_lb + c_ub) * 0.5
        else:
            c_lb = max(c_lb, c)
            if c_ub < 1e9:
                c = (c_lb + c_ub) * 0.5
            else:
                c *= 10
        return c, c_lb, c_ub

    def pn_optimize(self, verbose=True) -> np.ndarray:
        """
        Optimizes pertinent negatives.

        :return: The pertinent negative.
        :rtype: np.ndarray
        """
        if self.background_data is None:
            background = np.zeros(self.x0.shape)
        else:
            background = np.median(self.background_data, axis=0, keepdims=True)
        bar = ProgressBar(self.num_iterations) if verbose else None

        c_lb, c_ub, c = 0, 1e10, self.c
        best_solution, best_loss = None, 1e8
        for step in range(self.binary_search_steps):
            self._init_functions(c)
            delta = self.x0.copy()
            y = self.x0.copy()
            current_best_sol, current_best_loss = None, 1e8

            for iteration in range(self.num_iterations):
                learning_rate = self._learning_rate(iteration)
                # Update delta
                gradient = self._compute_gradient(self.pn_g_func, y)
                z = y - learning_rate * gradient - self.x0
                cond1 = (z > self.beta).astype(int)
                cond2 = (np.abs(z) <= self.beta).astype(int)
                cond3 = (z < -self.beta).astype(int)
                s = (z - self.beta) * cond1 + self.x0 * cond2 + (z + self.beta) * cond3
                s = np.minimum(np.maximum(s, self.bounds[0]), self.bounds[1])
                p = (np.abs(s - background) > np.abs(self.x0 - background)).astype(int)
                new_delta = p * s + (1 - p) * self.x0

                # Update y
                s = new_delta + (new_delta - delta) * (iteration / (iteration + 3))
                s = np.minimum(np.maximum(s, self.bounds[0]), self.bounds[1])
                p = (np.abs(s - background) > np.abs(self.x0 - background)).astype(int)
                y = p * s + (1 - p) * self.x0
                if np.sum(np.abs(delta - new_delta)) < 1e-8:
                    break
                delta = new_delta

                score = self._predict(delta)[0]
                if np.argmax(score) != self.target:
                    z = delta - self.x0
                    f = self.beta * np.sum(np.abs(z)) + np.sum(z * z)
                    if f < current_best_loss:
                        current_best_loss, current_best_sol = f, delta
                    if f < best_loss:
                        best_loss, best_solution = f, delta
                if verbose:
                    bar.print(iteration, prefix=f"Binary step: {step + 1}", suffix="")

            c, c_lb, c_ub = self._update_const(c, c_lb, c_ub, current_best_sol)
        return best_solution

    def pp_optimize(self, verbose=True) -> np.ndarray:
        """
        Optimizes pertinent positives.

        :return: The pertinent positive.
        :rtype: np.ndarray
        """
        if self.background_data is None:
            background = np.zeros(self.x0.shape)
        else:
            background = np.median(self.background_data, axis=0, keepdims=True)
        bar = ProgressBar(self.num_iterations) if verbose else None

        c_lb, c_ub, c = 0, 1e10, self.c
        best_solution, best_loss = None, 1e8
        for step in range(self.binary_search_steps):
            self._init_functions(c)
            delta = self.x0.copy()
            y = self.x0.copy()
            current_best_sol, current_best_loss = None, 1e8

            for iteration in range(self.num_iterations):
                learning_rate = self._learning_rate(iteration)
                # Update delta
                gradient = self._compute_gradient(self.pp_g_func, y)
                z = y - learning_rate * gradient
                s = (z - self.beta) * (z > self.beta) + (z + self.beta) * (z < -self.beta)
                s = np.minimum(np.maximum(s, self.bounds[0]), self.bounds[1])
                p = (np.abs(s - background) <= np.abs(self.x0 - background)).astype(int)
                new_delta = p * s

                # Update y
                s = new_delta + (new_delta - delta) * (iteration / (iteration + 3))
                s = np.minimum(np.maximum(s, self.bounds[0]), self.bounds[1])
                p = (np.abs(s - background) <= np.abs(self.x0 - background)).astype(int)
                y = p * s
                if np.sum(np.abs(delta - new_delta)) < 1e-8:
                    break
                delta = new_delta

                score = self._predict(delta)[0]
                predicted_label = np.argmax(score)
                if predicted_label == self.target:
                    f = self.beta * np.sum(np.abs(delta)) + np.sum(delta * delta)
                    if f < current_best_loss:
                        current_best_loss, current_best_sol = f, delta
                    if f < best_loss:
                        best_loss, best_solution = f, delta
                if verbose:
                    bar.print(iteration, prefix=f"Binary step: {step + 1}", suffix="")

            c, c_lb, c_ub = self._update_const(c, c_lb, c_ub, current_best_sol)
        return best_solution


class ContrastiveExplainer(ExplainerBase):
    """
    The contrastive explainer for image classification.
    If using this explainer, please cite the original work: https://arxiv.org/abs/1802.07623.
    This explainer only supports classification tasks.
    """

    explanation_type = "local"
    alias = ["cem", "contrastive"]

    def __init__(
        self,
        model,
        preprocess_function: Callable,
        mode: str = "classification",
        background_data: Image = Image(),
        c=10.0,
        beta=0.1,
        gamma=0.0,
        kappa=10.0,
        ae_model=None,
        binary_search_steps=5,
        learning_rate=1e-2,
        num_iterations=1000,
        grad_clip=1e3,
        **kwargs,
    ):
        """
        :param model: The model to explain, whose type is `torch.nn.Module` or `tf.keras.Model`.
        :param preprocess_function: The pre-processing function that converts the raw input features
            into the inputs of ``model``.
        :param mode: It can be `classification` only.
        :param background_data: Sampled images for estimating background values.
        :param c: The weight of the loss term.
        :param beta: The weight of the L1 regularization term.
        :param gamma: The weight of the AE regularization term.
        :param kappa: The parameter in the hinge loss function.
        :param ae_model: The auto-encoder model used for regularization.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        """
        super().__init__()
        assert mode == "classification", "CEM supports classification tasks only."
        assert isinstance(background_data, Image), "`background_data` should be an instance of Image."

        self.model = model
        self.preprocess_function = preprocess_function
        self.create_optimizer = lambda x, y: CEMOptimizer(
            x,
            y,
            model,
            c=c,
            beta=beta,
            gamma=gamma,
            kappa=kappa,
            ae_model=ae_model,
            binary_search_steps=binary_search_steps,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            grad_clip=grad_clip,
            background_data=self._preprocess(background_data),
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

    def explain(self, X: Image, **kwargs) -> ContrastiveExplanation:
        """
        Generates the explanations corresponding to the input images.
        Note that the returned results including the original input images,
        the pertinent negatives and the pertinent positives have been processed
        by the ``preprocess_function``, e.g., if the ``preprocess_function`` rescales
        [0, 255] to [0, 1], the return results will have range [0, 1].

        :param X: A batch of the input images.
        :return: The explanations for all the images, e.g., pertinent negatives and pertinent positives.
        """
        assert min(X.shape[1:3]) > 4, f"The image size ({X.shape[1]}, {X.shape[2]}) is too small."
        verbose = kwargs.get("kwargs", True)
        explanations = ContrastiveExplanation()
        y = self._predict(self._preprocess(X))

        for i in range(len(X)):
            x = self._preprocess(X[i])
            optimizer = self.create_optimizer(x=x, y=y[i])
            # Original image
            x = x.squeeze()
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))

            # Get the pertinent negative and the label
            pn = optimizer.pn_optimize(verbose=verbose)
            if pn is not None:
                pn_label = self._predict(pn)[0]
                pn = pn.squeeze()
                if pn.ndim == 3 and pn.shape[0] == 3:
                    pn = np.transpose(pn, (1, 2, 0))
            else:
                pn_label = None

            # Get the pertinent positive and the label
            pp = optimizer.pp_optimize(verbose=verbose)
            if pp is not None:
                pp_label = self._predict(pp)[0]
                pp = pp.squeeze()
                if pp.ndim == 3 and pp.shape[0] == 3:
                    pp = np.transpose(pp, (1, 2, 0))
            else:
                pp_label = None

            explanations.add(image=x, label=y[i], pn=pn, pn_label=pn_label, pp=pp, pp_label=pp_label)
        return explanations
