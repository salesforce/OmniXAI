#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The basic counterfactual explainer for tabular data.
"""
import numpy as np
from typing import Callable

from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular.base import TabularExplainer
from omnixai.explanations.tabular.counterfactual import CFExplanation
from omnixai.utils.misc import is_torch_available, is_tf_available, ProgressBar

if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.autograd import grad


    class _ObjectiveTorch(nn.Module):
        def __init__(self, x0, target, model, c, kappa, gamma=None):
            super().__init__()
            param = next(model.parameters())
            if isinstance(x0, np.ndarray):
                self.x0 = torch.tensor(x0, dtype=torch.get_default_dtype()).to(param.device)
            else:
                self.x0 = x0.to(param.device)
            self.num_classes = model(self.x0).shape[1]
            self.target = torch.tensor(np.eye(1, self.num_classes, target), dtype=torch.get_default_dtype()).to(
                param.device
            )
            if gamma is None:
                self.gamma = 1
            else:
                self.gamma = torch.tensor(
                    np.expand_dims(np.abs(gamma) + 1e-8, axis=0), dtype=torch.get_default_dtype()
                ).to(param.device)

            self.model = model.eval()
            self.c = c
            self.kappa = kappa
            self.reduce_dims = list(range(1, len(x0.shape)))

        def forward(self, x):
            # Regularization term
            regularization = torch.sum(torch.abs(self.x0 - x) / self.gamma, dim=self.reduce_dims)
            # Loss function
            prob = self.model(x)
            a = torch.sum(prob * self.target, dim=1)
            b = torch.max((1 - self.target) * prob - self.target * 10000, dim=1)[0]
            loss = nn.functional.relu(a - b + self.kappa)
            return torch.mean(self.c * loss + regularization), torch.mean(a - b)

if is_tf_available():
    import tensorflow as tf


    class _ObjectiveTF(tf.keras.Model):
        def __init__(self, x0, target, model, c, kappa, gamma=None):
            super().__init__()
            if isinstance(x0, np.ndarray):
                self.x0 = tf.convert_to_tensor(x0, dtype=tf.keras.backend.floatx())
            else:
                self.x0 = x0
            self.num_classes = model(self.x0).shape[1]
            self.target = tf.convert_to_tensor(np.eye(1, self.num_classes, target), dtype=tf.keras.backend.floatx())
            if gamma is None:
                self.gamma = 1
            else:
                self.gamma = tf.convert_to_tensor(
                    np.expand_dims(np.abs(gamma) + 1e-8, axis=0), dtype=tf.keras.backend.floatx()
                )

            self.model = model
            self.c = c
            self.kappa = kappa
            self.reduce_dims = list(range(1, len(x0.shape)))

        def call(self, x):
            # Regularization term
            regularization = tf.reduce_sum(tf.abs(self.x0 - x) / self.gamma, axis=self.reduce_dims)
            # Loss function
            prob = self.model(x)
            a = tf.reduce_sum(prob * self.target, axis=1)
            b = tf.reduce_max((1 - self.target) * prob - self.target * 10000, axis=1)
            loss = tf.maximum(0.0, a - b + self.kappa)
            return tf.reduce_mean(self.c * loss + regularization), tf.reduce_mean(a - b)


class CounterfactualOptimizer:
    """
    The optimizer for counterfactual explanation, which is implemented based
    on the paper `Counterfactual Explanations without Opening the Black Box: Automated Decisions
    and the GDPR, Sandra Wachter, Brent Mittelstadt, Chris Russell, https://arxiv.org/abs/1711.00399`.
    """

    def __init__(
            self,
            x0,
            target,
            model,
            c=10.0,
            kappa=10.0,
            binary_search_steps=5,
            learning_rate=1e-2,
            num_iterations=1000,
            grad_clip=1e3,
            gamma=None,
            bounds=None,
    ):
        """
        :param x0: The input image.
        :param target: The predicted label of the input image.
        :param model: The classification model which can be `torch.nn.Module` or `tf.keras.Model`.
        :param c: The weight of the hinge loss term.
        :param kappa: The parameter in the hinge loss function.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        :param gamma: The denominator of the regularization term, e.g., `|x - x0| / gamma`.
            ``gamma`` will be set to 1 if it is None.
        :param bounds: The upper and lower bounds of the feature values. `None` if the default
            bounds `(min(x0), max(x0))` is used.
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
        self.kappa = kappa
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.bounds = (np.min(x0), np.max(x0)) if bounds is None else bounds

        self.model_type = None
        if is_torch_available():
            if isinstance(self.model, nn.Module):
                self.model_type = "torch"
        if self.model_type is None and is_tf_available():
            if isinstance(self.model, tf.keras.Model):
                self.model_type = "tf"
        if self.model_type is None:
            self.model_type = "other"
        self.num_classes = self._predict(self.x0).shape[1]

    def _init_functions(self, c):
        if self.model_type == "torch":
            self.func = _ObjectiveTorch(self.x0, self.target, self.model, c=c, kappa=self.kappa, gamma=self.gamma)
        elif self.model_type == "tf":
            self.func = _ObjectiveTF(self.x0, self.target, self.model, c=c, kappa=self.kappa, gamma=self.gamma)
        else:
            self.func = self.model

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
                predictions, loss = model(inputs)
                gradients = tape.gradient(predictions, inputs).numpy()
                loss = loss.numpy()

        elif self.model_type == "torch":
            model.eval()
            param = next(model.parameters())
            inputs = torch.tensor(inputs, requires_grad=True, dtype=torch.get_default_dtype()).to(param.device)
            predictions, loss = model(inputs)
            gradients = (
                grad(outputs=predictions, inputs=inputs, grad_outputs=torch.ones_like(predictions).to(param.device))[0]
                    .detach()
                    .cpu()
                    .numpy()
            )
            loss = loss.detach().cpu().numpy()
        else:
            # TODO: Numerical differentiation
            raise NotImplementedError
        gradients = np.maximum(np.minimum(gradients, self.grad_clip), -self.grad_clip)
        return gradients, loss

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

    def optimize(self, verbose=True) -> np.ndarray:
        """
        Generates counterfactual examples.

        :return: The counterfactual example.
        :rtype: np.ndarray
        """
        bar = ProgressBar(self.num_iterations) if verbose else None

        c_lb, c_ub, c = 0, 1e10, self.c
        best_solution, best_loss = None, 1e8
        for step in range(self.binary_search_steps):
            self._init_functions(c)
            x = self.x0.copy()
            current_best_sol, current_best_loss = None, 1e8

            for iteration in range(self.num_iterations):
                # Compute the gradient and loss
                gradient, loss = self._compute_gradient(self.func, x)
                if loss < 0:
                    f = np.sum(np.abs(x - self.x0))
                    if f < current_best_loss:
                        current_best_loss, current_best_sol = f, x
                    if f < best_loss:
                        best_loss, best_solution = f, x
                # Update x
                new_x = x - self._learning_rate(iteration) * gradient
                new_x = np.minimum(np.maximum(new_x, self.bounds[0]), self.bounds[1])
                if np.sum(np.abs(x - new_x)) < 1e-6:
                    break
                x = new_x
                if verbose:
                    bar.print(iteration, prefix=f"Binary step: {step + 1}", suffix="")

            c, c_lb, c_ub = self._update_const(c, c_lb, c_ub, current_best_sol)
        return best_solution


class CounterfactualExplainer(TabularExplainer):
    """
    The basic counterfactual explainer for tabular data. It only supports continuous-valued features.
    If using this explainer, please cite the paper `Counterfactual Explanations without
    Opening the Black Box: Automated Decisions and the GDPR, Sandra Wachter, Brent Mittelstadt, Chris Russell,
    https://arxiv.org/abs/1711.00399`.
    """

    explanation_type = "local"
    alias = ["ce", "counterfactual"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            mode: str = "classification",
            c=10.0,
            kappa=10.0,
            binary_search_steps=5,
            learning_rate=1e-2,
            num_iterations=1000,
            grad_clip=1e3,
            **kwargs,
    ):
        """
        :param training_data: The data used to extract information such as medians of
            continuous-valued features. ``training_data`` can be the training dataset for training
            the machine learning model. If the training dataset is large, ``training_data`` can be
            its subset by applying `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, which only supports `classification`.
        :param c: The weight of the hinge loss term.
        :param kappa: The parameter in the hinge loss function.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        assert mode == "classification", "CE supports classification tasks only."
        assert (
                len(self.categorical_columns) == 0
        ), "The integrated-gradient explainer only supports continuous-valued features"

        model_type = None
        if is_tf_available():
            import tensorflow as tf

            if isinstance(predict_function, tf.keras.Model):
                model_type = "tf"
        if model_type is None and is_torch_available():
            import torch.nn as nn

            if isinstance(predict_function, nn.Module):
                model_type = "torch"
        if model_type is None:
            raise ValueError(
                f"`predict_function` should be a tf.keras.Model "
                f"or a torch.nn.Module instead of {type(predict_function)}"
            )

        self.model = predict_function
        self.model_type = model_type
        self.predict_fn = None
        # Means
        medians = training_data.get_continuous_medians()
        medians = np.array([medians[col] for col in self.continuous_columns])
        self.mean = np.mean(np.abs(self.data.astype(float) - np.expand_dims(medians, axis=0)), axis=0)
        # Bounds
        bounds = training_data.get_continuous_bounds()
        self.bounds = (np.expand_dims(bounds[0], axis=0), np.expand_dims(bounds[1], axis=0))

        # Optimizer
        self.create_optimizer = lambda x, y, model: CounterfactualOptimizer(
            x,
            y,
            model,
            c=c,
            kappa=kappa,
            binary_search_steps=binary_search_steps,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            grad_clip=grad_clip,
            gamma=self.mean,
            bounds=self.bounds,
        )

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts class labels in classification.

        :param inputs: The input instances.
        :return: The predicted labels.
        :rtype: np.ndarray
        """
        if self.model_type == "other":
            scores = self.model(inputs)
        elif self.model_type == "tf":
            scores = self.model(inputs).numpy()
        else:
            import torch

            self.model.eval()
            param = next(self.model.parameters())
            X = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.get_default_dtype())
            scores = self.model(X.to(param.device)).detach().cpu().numpy()
        y = np.argmax(scores, axis=1).astype(int)
        return y

    def _revise(self, instance, label, cf, batch_size=32):
        """
        Revises the found counterfactual example to improve sparsity.

        :param instance: The input instance.
        :param label: The predicted label of the input instance.
        :param cf: The found counterfactual example.
        :param batch_size: The batch size during the revision.
        :return: The optimized counterfactual example.
        :rtype: np.ndarray
        """
        assert instance.shape[0] == 1 and cf.shape[0] == 1
        # Sort according to the changes
        d = np.abs(instance - cf) / (np.expand_dims(self.mean, axis=0) + 1e-8)
        d = sorted(zip(d[0], range(d.shape[1])), key=lambda x: x[0], reverse=True)
        indices = [t[1] for t in d]
        # A greedy method
        masks = np.zeros(instance.shape)
        for i in range(0, cf.shape[1], batch_size):
            j = min(cf.shape[1], i + batch_size)
            inputs = []
            for k in range(i, j):
                masks[0, indices[k]] = 1
                inputs.append(masks * cf + (1 - masks) * instance)
            inputs = np.concatenate(inputs, axis=0)
            # Check if it is a valid counterfactual example
            cf_labels = self._predict(inputs)
            for k, cf_label in enumerate(cf_labels):
                if cf_label != label:
                    return inputs[k: k + 1]
        return cf

    def explain(self, X, **kwargs) -> CFExplanation:
        """
        Generates the counterfactual explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :return: The counterfactual explanations for all the input instances.
        """
        verbose = kwargs.get("kwargs", True)
        explanations = CFExplanation()
        X = self._to_tabular(X).remove_target_column()
        instances = self._to_numpy(X)
        y = self._predict(instances)

        for i in range(instances.shape[0]):
            optimizer = self.create_optimizer(x=instances[i: i + 1], y=y[i], model=self.model)
            cf = optimizer.optimize(verbose=verbose)
            instance_df = X.iloc(i).to_pd()
            instance_df["label"] = y[i]
            if cf is not None:
                cf = self._revise(instance=instances[i: i + 1], label=y[i], cf=cf)
                cf_df = self._to_tabular(cf).to_pd()
                cf_df["label"] = self._predict(cf)[0]
            else:
                cf_df = None
            explanations.add(query=instance_df, cfs=cf_df)
        return explanations

    def save(
            self,
            directory: str,
            filename: str = None,
            **kwargs
    ):
        """
        Saves the initialized explainer.

        :param directory: The folder for the dumped explainer.
        :param filename: The filename (the explainer class name if it is None).
        """
        super().save(
            directory=directory,
            filename=filename,
            ignored_attributes=["data"],
            **kwargs
        )
