#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The basic counterfactual explainer for time series tasks.
"""
import numpy as np
from typing import Callable
from scipy.optimize import approx_fprime

from omnixai.data.timeseries import Timeseries
from omnixai.explainers.base import ExplainerBase
from omnixai.explanations.timeseries.counterfactual import CFExplanation
from omnixai.utils.misc import ProgressBar


class CounterfactualOptimizer:
    """
    The optimizer for the counterfactual explainer for time series anomaly detection.
    """

    def __init__(
            self,
            x0,
            ts_len,
            bounds,
            threshold,
            predict_function,
            gamma=None,
            c=10.0,
            kappa=0.0,
            smooth_weight=1e-3,
            binary_search_steps=5,
            learning_rate=1e-2,
            num_iterations=1000,
            grad_clip=1e3,
            grid_size=1e3,
            **kwargs
    ):
        """
        :param x0: The input time series reshaped in one dimension.
        :param ts_len: The length of the time series.
        :param bounds: The upper and lower bounds of each value in ``x0``.
        :param threshold: The threshold to determine whether an instance is anomalous,
            e.g., `anomaly score > threshold`.
        :param predict_function: The prediction function for computing anomaly scores.
        :param gamma: The denominator of the regularization term, e.g., `|x - x0| / gamma`.
            ``gamma`` will be set to 1 if it is None.
        :param c: The weight of the hinge loss term.
        :param kappa: The parameter in the hinge loss function.
        :param smooth_weight: The weight of the smoothness regularization term, e.g., `|x(t) - x(t-1)|`.
        :param binary_search_steps: The number of iterations to adjust the weight of the loss term.
        :param learning_rate: The learning rate.
        :param num_iterations: The maximum number of iterations during optimization.
        :param grad_clip: The value for clipping gradients.
        :param grid_size: The number of bins in each dimension in ``x0`` for computing numerical gradients.
        """
        assert x0.ndim == 1
        assert x0.shape[0] == bounds.shape[1]
        assert bounds.shape[0] == 2
        if gamma is not None:
            assert gamma.ndim == 1 and gamma.shape == x0.shape

        self.x0 = x0
        self.ts_len = ts_len
        self.bounds = bounds
        self.threshold = threshold
        self.predict_function = predict_function
        self.gamma = gamma if gamma is not None else 1
        self.c = c
        self.kappa = kappa
        self.smooth_weight = smooth_weight
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.grad_clip = grad_clip
        self.diff_eps = (self.bounds[1] - self.bounds[0]) / grid_size

    def _init_functions(self, c):
        def _predict(x):
            _s = self.predict_function(x)
            _loss = max(0, _s - self.threshold + self.kappa)
            _regularization = np.mean(np.abs(x - self.x0) / (self.gamma + 1e-8))
            _y = (x / (self.gamma + 1e-8)).reshape((self.ts_len, -1))
            _smoothness = np.mean(np.abs(_y[:-1, :] - _y[1:, :]))
            return c * _loss + _regularization + self.smooth_weight * _smoothness

        self.func = _predict

    def _predict(self, x):
        score = self.predict_function(x)
        label = int(score > self.threshold)
        return label

    def _loss(self, x):
        return np.sum(np.abs(x - self.x0))

    def _compute_gradient(self, func, x):
        assert x.ndim == 1
        gradients = approx_fprime(x, func, self.diff_eps)
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

    def _revise(self, instance, label, cf):
        """
        Revises the found counterfactual example to improve sparsity.

        :param instance: The input instance.
        :param label: The predicted label of the input instance.
        :param cf: The found counterfactual example.
        :return: The optimized counterfactual example.
        :rtype: np.ndarray
        """
        assert instance.ndim == 1 and cf.ndim == 1
        d = np.abs(instance - cf) / (self.gamma + 1e-8)
        d = sorted(zip(d, range(len(d))), key=lambda x: x[0], reverse=True)
        indices = [t[1] for t in d]
        masks = np.zeros(instance.shape)

        for i in range(cf.shape[0]):
            masks[indices[i]] = 1
            x = masks * cf + (1 - masks) * instance
            cf_label = self._predict(x)
            if cf_label != label:
                return x
        return cf

    def optimize(self, revise=True, verbose=True):
        """
        Generates counterfactual examples.

        :return: The counterfactual example or None if no counterfactual example is found.
        :rtype: np.ndarray or None
        """
        bar = ProgressBar(self.num_iterations) if verbose else None
        original_label = self._predict(self.x0)

        c_lb, c_ub, c = 0, 1e10, self.c
        best_solution, best_loss = None, 1e8
        for step in range(self.binary_search_steps):
            self._init_functions(c)
            x = self.x0.copy()
            current_best_sol, current_best_loss = None, 1e8

            for iteration in range(self.num_iterations):
                label = self._predict(x)
                gradient = self._compute_gradient(self.func, x)
                if abs(original_label - label) != 0:
                    f = self._loss(x)
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

        if best_solution is not None and revise:
            return self._revise(self.x0, original_label, best_solution)
        else:
            return best_solution


class CounterfactualExplainer(ExplainerBase):
    """
    The basic counterfactual explainer for time series anomaly detection or forecasting.
    """

    explanation_type = "local"
    alias = ["ce", "counterfactual"]

    def __init__(
            self,
            training_data: Timeseries,
            predict_function: Callable,
            mode: str = "anomaly_detection",
            threshold: float = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize the explainer.
        :param predict_function: The prediction function corresponding to the model to explain.
            The inputs of ``predict_function`` should be a batch (list) of time series, i.e., a `Timeseries`
            instance. The outputs of ``predict_function`` are anomaly scores (higher scores imply more anomalous)
            for anomaly detection or predicted values for forecasting.
        :param mode: The task type, e.g., `anomaly_detection` or `forecasting`.
        :param threshold: The threshold to determine whether an instance is anomalous,
            e.g., anomaly score > threshold.
        :param kwargs: Additional parameters for `CounterfactualOptimizer`.
        """
        super().__init__()
        assert isinstance(training_data, Timeseries), \
            "`training_data` should be an instance of Timeseries."
        assert mode in ["anomaly_detection", "forecasting"], \
            "`mode` can either be `anomaly_detection` or `forecasting`"
        assert threshold is not None, \
            "Please set the detection threshold, e.g., a data point is an anomaly " \
            "if its anomaly score is greater than `threshold`."
        self.kwargs = kwargs

        self.mode = mode
        self.data = training_data
        self.predict_function = predict_function
        self.threshold = threshold
        self.variable_names = list(self.data.columns)
        # The lengths of test instances must be the same
        self.test_ts_length = None
        self.predictor = None

    def _build_predictor(self, ts_len):
        def _predict(x: np.ndarray):
            x = x.reshape((ts_len, len(self.variable_names)))
            ts = Timeseries(x, variable_names=self.variable_names)
            try:
                return np.array([self.predict_function(ts)]).flatten()[0]
            except:
                return np.array([self.predict_function([ts])]).flatten()[0]
        return _predict

    def _build_explainer(self, ts_len):
        if self.predictor is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"
        self.test_ts_length = ts_len
        self.predictor = self._build_predictor(ts_len)

        ts = self.data.to_numpy(copy=False)
        # The lower and upper bounds
        bounds = np.stack([np.min(ts, axis=0), np.max(ts, axis=0)])
        bound_min = np.stack([bounds[0]] * ts_len).reshape((1, -1))
        bound_max = np.stack([bounds[1]] * ts_len).reshape((1, -1))
        self.bounds = np.concatenate([bound_min, bound_max], axis=0)
        # The absolute median value for each variable
        gamma = np.median(np.abs(ts), axis=0)
        self.gamma = np.stack([gamma] * ts_len).flatten()

    def explain(self, X: Timeseries, **kwargs) -> CFExplanation:
        """
        Generates the counterfactual examples for the input instances.

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.
        :param kwargs: Additional parameters for `CounterfactualOptimizer`.
        :return: The counterfactual explanations for all the input instances.
        """
        self.kwargs.update(kwargs)
        self._build_explainer(X.ts_len)
        explanations = CFExplanation()
        instance = X.values.flatten()

        optimizer = CounterfactualOptimizer(
            x0=instance,
            ts_len=X.ts_len,
            bounds=self.bounds,
            threshold=self.threshold,
            predict_function=self.predictor,
            gamma=self.gamma,
            **self.kwargs
        )
        x = optimizer.optimize(
            revise=kwargs.get("revise", True),
            verbose=kwargs.get("verbose", True)
        )
        explanations.add(
            query=Timeseries(
                data=instance.reshape((X.ts_len, -1)),
                timestamps=X.index,
                variable_names=X.columns
            ).to_pd(),
            cfs=Timeseries(
                data=x.reshape((X.ts_len, -1)),
                timestamps=X.index,
                variable_names=X.columns
            ).to_pd() if x is not None else None
        )
        return explanations
