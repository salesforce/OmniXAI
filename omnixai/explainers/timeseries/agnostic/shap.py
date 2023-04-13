#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for time series tasks.
"""
import shap
import random
import numpy as np
import pandas as pd
from typing import Callable

from ...base import ExplainerBase
from ....data.timeseries import Timeseries
from ....explanations.timeseries.feature_importance import FeatureImportance


class ShapTimeseries(ExplainerBase):
    """
    The SHAP explainer for time series forecasting and anomaly detection.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "local"
    alias = ["shap"]

    def __init__(
            self,
            training_data: Timeseries,
            predict_function: Callable,
            mode: str = "anomaly_detection",
            **kwargs
    ):
        """
        :param training_data: The data used to initialize the explainer.
        :param predict_function: The prediction function corresponding to the model to explain.
            The input of ``predict_function`` is an `Timeseries` instance. The output of ``predict_function``
            is the anomaly score (higher scores imply more anomalous) for anomaly detection or the predicted
            value for forecasting.
        :param mode: The task type, e.g., `anomaly_detection` or `forecasting`.
        """
        super().__init__()
        assert isinstance(training_data, Timeseries), \
            "`training_data` should be an instance of Timeseries."
        assert mode in ["anomaly_detection", "forecasting"], \
            "`mode` can either be `anomaly_detection` or `forecasting`"

        self.mode = mode
        self.data = training_data
        self.predict_function = predict_function
        self.variables = list(self.data.columns)

        # The lengths of test instances must be the same
        self.explainer = None
        self.test_ts_length = None
        self.index2timestamps = None
        self.all_idx2ts = None

    def _build_data(self, ts_len, num_samples):
        interval = range(self.data.ts_len - ts_len)
        ps = random.sample(interval, min(num_samples, len(interval)))
        samples, index2timestamps = [], {}
        for i, p in enumerate(ps):
            index2timestamps[i] = self.data.index[p:p + ts_len]
            x = self.data.values[p:p + ts_len]
            y = np.zeros((x.shape[0] * x.shape[1] + 1,))
            y[:-1], y[-1] = x.flatten(), i
            samples.append(y.flatten())
        return np.array(samples), index2timestamps

    def _build_predictor(self, ts_len):
        def _predict(xs: np.ndarray):
            xs = xs.reshape((xs.shape[0], -1))
            ts = [
                Timeseries(
                    data=x[:-1].reshape((ts_len, -1)),
                    variable_names=self.variables,
                    timestamps=self.all_idx2ts[x[-1]]
                ) for x in xs
            ]
            try:
                return np.array(self.predict_function(ts)).flatten()
            except:
                return np.array([self.predict_function(t) for t in ts]).flatten()
        return _predict

    def _build_explainer(self, ts_len, num_samples=100):
        if self.explainer is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"

        data, self.index2timestamps = self._build_data(ts_len, num_samples)
        self.all_idx2ts = self.index2timestamps.copy()
        self.explainer = shap.KernelExplainer(
            model=self._build_predictor(ts_len),
            data=data,
            link="identity"
        )
        self.test_ts_length = ts_len

    def explain(self, X: Timeseries, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`, e.g.,
            "nsamples" for the number of times to re-evaluate the model when explaining each prediction.
        :return: The feature-importance explanations for all the input instances.
        """
        # Initialize the SHAP explainer if it is not created.
        self._build_explainer(X.ts_len)
        explanations = FeatureImportance(self.mode)

        index = max(self.index2timestamps.keys()) + 1
        self.all_idx2ts = self.index2timestamps.copy()
        self.all_idx2ts[index] = X.index

        instances = np.zeros((1, X.shape[0] * X.shape[1] + 1))
        instances[:, :-1] = X.values.reshape((1, -1))
        instances[:, -1] = index

        shap_values = self.explainer.shap_values(instances, **kwargs)
        shap_values = shap_values.flatten()
        metric_shap_values = shap_values[:-1].reshape(X.shape)
        timestamp_shap_value = shap_values[-1]

        scores = pd.DataFrame(
            metric_shap_values,
            columns=X.columns,
            index=X.index
        )
        scores["@timestamp"] = timestamp_shap_value
        explanations.add(X.to_pd(), scores)
        return explanations
