#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Model-Agnostic Counterfactual Explanation (MACE) designed for time series tasks.
"""
import numpy as np
from typing import Callable
from scipy.optimize import approx_fprime

from omnixai.data.timeseries import Timeseries
from omnixai.explainers.base import ExplainerBase
from omnixai.explanations.timeseries.counterfactual import CFExplanation
from omnixai.utils.misc import is_torch_available
from omnixai.preprocessing.encode import KBins
from omnixai.utils.misc import ProgressBar

if is_torch_available():
    import torch
    import torch.nn as nn


    class _Policy(nn.Module):

        def __init__(self, x, candidates, regularization_weight, entropy_weight):
            super().__init__()
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            self.x = x
            self.candidates = candidates
            self.regularization_weight = regularization_weight
            self.entropy_weight = entropy_weight
            self.columns = sorted(candidates.keys())
            self.column_prob = None
            self.action_prob = None
            self._build_layers()

        def _build_layers(self):
            self.column_prob = nn.Parameter(
                torch.zeros(len(self.columns)), requires_grad=True)
            self.action_prob = []
            for c in self.columns:
                self.action_prob.append(nn.Parameter(
                    torch.zeros(len(self.candidates[c])), requires_grad=True))
            self.action_prob = nn.ParameterList(self.action_prob)

        def _get_column_prob(self):
            return torch.clamp(torch.sigmoid(self.column_prob), min=0.1, max=0.9)

        @staticmethod
        def _clip_softmax(p, dim, min_val=0.1, max_val=0.9):
            min_val = min(1.0 / p.shape[0], min_val)
            x = torch.clamp(torch.softmax(p, dim=dim), min=min_val, max=max_val)
            return x / torch.sum(x)

        def forward(self, inputs):
            columns, actions, rewards = inputs
            # shape of pa and pb: (batch_size, num_columns)
            column_prob = self._get_column_prob()
            c = column_prob.view((1, column_prob.shape[0]))
            pa = torch.log(c) * columns + torch.log(1.0 - c) * (1.0 - columns)

            pb = []
            for i, p in enumerate(self.action_prob):
                prob, action = self._clip_softmax(p, dim=0), actions[:, i]
                pb.append(prob[action].view((action.shape[0], 1)))
            pb = torch.log(torch.cat(pb, dim=1))

            loss = torch.mean(torch.sum(pa + pb, dim=1) * rewards)
            regularization = torch.mean(column_prob) * self.regularization_weight
            entropy = (torch.mean(column_prob * torch.log(column_prob))) * self.entropy_weight
            return -loss + regularization + entropy

        def get_actions(self, column_probs, action_probs, num_actions):
            batch_samples, batch_columns, batch_actions = [], [], []
            for _ in range(num_actions):
                x = self.x.copy()
                columns = np.random.binomial(1, column_probs)
                actions = [np.random.choice(range(len(p)), p=p) for p in action_probs]
                for k, (col, act) in enumerate(zip(columns, actions)):
                    if col > 0:
                        key = self.columns[k]
                        x[0, key] = self.candidates[key][act]
                batch_samples.append(x)
                batch_columns.append(columns)
                batch_actions.append(actions)
            return np.concatenate(batch_samples, axis=0), \
                   np.array(batch_columns), \
                   np.array(batch_actions)

        def get_probabilities(self):
            column_probs = self._get_column_prob().detach().cpu().numpy()
            action_probs = [self._clip_softmax(p, dim=0).detach().cpu().numpy()
                            for p in self.action_prob]
            return column_probs, action_probs


class MACEExplainer(ExplainerBase):
    """
    The Model-Agnostic Counterfactual Explanation (MACE) developed by Yang et al. Please
    cite the paper `MACE: An Efficient Model-Agnostic Framework for Counterfactual Explanation`.
    This is a special version designed for time series anomaly detection and forecasting.
    """
    explanation_type = "local"
    alias = ["mace"]

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
        """
        super().__init__()
        assert is_torch_available(), \
            "MACEExplainer for time series requires the installation of PyTorch."
        assert isinstance(training_data, Timeseries), \
            "`training_data` should be an instance of Timeseries."
        assert mode in ["anomaly_detection", "forecasting"], \
            "`mode` can either be `anomaly_detection` or `forecasting`"
        assert threshold is not None, \
            "Please set the detection threshold, e.g., a data point is an anomaly " \
            "if its anomaly score is greater than `threshold`."

        self.mode = mode
        self.data = training_data
        self.predict_function = predict_function
        self.threshold = threshold
        self.variable_names = list(self.data.columns)
        # The lengths of test instances must be the same
        self.test_ts_length = None
        self.predictor = None
        self.candidates = None
        self.gamma = None
        self.bounds = None
        self.diff_eps = None

    def _candidates(self, ts_len, n_bins=8):
        # Discretize continuous values
        ts = self.data.to_numpy(copy=False)
        transformer = KBins(n_bins=n_bins).fit(ts)
        values = transformer.invert(np.array([range(n_bins)] * ts.shape[1]).T)
        # The lower and upper bounds
        bounds = np.stack([np.quantile(ts, 0.01, axis=0), np.quantile(ts, 0.99, axis=0)])
        values = np.concatenate([bounds[:1], values, bounds[1:]], axis=0)
        # Construct candidate lists
        indices = np.array([range(len(self.variable_names))] * ts_len).flatten()
        candidates = {k: values[:, i].tolist() for k, i in enumerate(indices)}
        # Gamma
        gamma = np.median(np.abs(ts), axis=0)
        gamma = np.stack([gamma] * ts_len).flatten()
        # Eps for gradients
        eps = np.stack([(bounds[1] - bounds[0]) / 1000] * ts_len).flatten()
        return candidates, gamma, bounds, eps

    def _build_predictor(self, ts_len):
        def _predict(x: np.ndarray):
            xs = x.reshape((-1, ts_len, len(self.variable_names)))
            ts = [Timeseries(x, variable_names=self.variable_names) for x in xs]
            try:
                return np.array(self.predict_function(ts)).flatten()
            except:
                return np.array([self.predict_function(t) for t in ts]).flatten()
        return _predict

    def _build_explainer(self, ts_len):
        if self.predictor is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"
        self.test_ts_length = ts_len
        self.predictor = self._build_predictor(ts_len)
        self.candidates, self.gamma, self.bounds, self.diff_eps = \
            self._candidates(ts_len)

    def _reward(self, x, less_than_threshold=True, is_train=True):
        scores = self.predictor(x)
        s = self.threshold - scores if less_than_threshold else scores - self.threshold
        if is_train:
            return s + 0.001 * (np.random.rand(*s.shape) * 2 - 1)
        else:
            return s

    def _optimize(
            self,
            x,
            learning_rate,
            reg_weight,
            entropy_weight,
            batch_size,
            num_iterations,
            less_than_threshold,
            verbose
    ):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        assert x.shape[0] == 1

        policy = _Policy(
            x=x,
            candidates=self.candidates,
            regularization_weight=reg_weight,
            entropy_weight=entropy_weight
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        bar = ProgressBar(num_iterations) if verbose else None

        # RL optimization
        for iteration in range(num_iterations):
            column_prob, action_prob = policy.get_probabilities()
            samples, batch_cols, batch_acts = \
                policy.get_actions(column_prob, action_prob, num_actions=batch_size)
            rewards = self._reward(samples, less_than_threshold, is_train=True)
            rewards = rewards - np.percentile(rewards, 50)
            optimizer.zero_grad()
            loss = policy((
                torch.LongTensor(batch_cols), torch.LongTensor(batch_acts), torch.FloatTensor(rewards)))
            loss.backward()
            optimizer.step()
            if verbose:
                bar.print(iteration, prefix="", suffix="")

        # Pick the optimal policy
        column_probs, action_probs = policy.get_probabilities()
        d, optimal = zip(policy.columns, column_probs, action_probs), []
        for column, _, prob in sorted(d, key=lambda c: c[1], reverse=True):
            vs = sorted(zip(policy.candidates[column], prob), key=lambda c: c[1], reverse=True)
            optimal.append((column, vs[0][0]))

        # A greedy method for picking counterfactual examples
        y = x.copy()
        for i in range(len(optimal)):
            column, feature = optimal[i]
            y[0, column] = feature
            if self._reward(y, less_than_threshold, is_train=False) > 0:
                return y
        return None

    def _revise(
            self,
            x,
            y,
            learning_rate,
            num_iterations,
            smoothness_weight,
            less_than_threshold
    ):
        if y is None:
            return None
        if x.ndim == 2:
            assert x.shape[0] == 1
            x = x.flatten()
        if y.ndim == 2:
            assert y.shape[0] == 1
            y = y.flatten()

        def _predict(t):
            _x = x.reshape((self.test_ts_length, -1)) / (self.gamma + 1e-6)
            _z = t.reshape((self.test_ts_length, -1)) / (self.gamma + 1e-6)
            _d = np.square(_x - _z)
            if _z.shape[0] >= 3:
                a = np.mean(np.square(_z[:-2, :] - _z[1:-1, :]) * _d[1:-1, :])
                b = np.mean(np.square(_z[2:, :] - _z[1:-1, :]) * _d[1:-1, :])
                smoothness = (a + b) * 0.5
            else:
                smoothness = 0
            return np.mean(_d) + smoothness * smoothness_weight

        def _learning_rate(it):
            return learning_rate * (1 - it / num_iterations) ** 0.5

        for i in range(num_iterations):
            grad = approx_fprime(y, _predict, self.diff_eps)
            grad = np.maximum(np.minimum(grad, 10), -10)
            z = y - _learning_rate(i) * grad
            z = np.minimum(np.maximum(z, self.bounds[0]), self.bounds[1])
            if self._reward(z, less_than_threshold, is_train=False) > 0:
                y = z
        return y

    def explain(self, X: Timeseries, **kwargs) -> CFExplanation:
        """
        Generates the counterfactual examples for the input instances.

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.
        :param kwargs: Additional parameters for MACE, e.g., "learning_rate" - the learning rate
            for RL, "batch_size" - the number of samples for RL,
            "num_iterations" - the number of optimization steps.
        :return: The counterfactual explanations for all the input instances.
        """
        self._build_explainer(X.ts_len)
        explanations = CFExplanation()
        label = int(self.predict_function(X) > self.threshold)
        instance = X.values.flatten()

        y = self._optimize(
            x=instance,
            learning_rate=kwargs.get("learning_rate", 0.1),
            reg_weight=kwargs.get("regularization_weight", 1e-4),
            entropy_weight=kwargs.get("entropy_weight", 1e-3),
            batch_size=kwargs.get("batch_size", 50),
            num_iterations=kwargs.get("num_iterations", 50),
            less_than_threshold=(label > 0),
            verbose=kwargs.get("verbose", True),
        )
        y = self._revise(
            x=instance,
            y=y,
            learning_rate=kwargs.get("revise_learning_rate", 1.0),
            num_iterations=kwargs.get("revise_num_iterations", 100),
            smoothness_weight=kwargs.get("smoothness_weight", 0.1),
            less_than_threshold=(label > 0)
        )
        explanations.add(
            query=Timeseries(
                data=instance.reshape((X.ts_len, -1)),
                timestamps=X.timestamps,
                variable_names=X.columns
            ).to_pd(),
            cfs=Timeseries(
                data=y.reshape((X.ts_len, -1)),
                timestamps=X.timestamps,
                variable_names=X.columns
            ).to_pd() if y is not None else None
        )
        return explanations
