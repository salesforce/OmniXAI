#
# Copyright (c) 2022 salesforce.com, inc.
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
        self.variables = list(self.data.columns) + ["@timestamp"]
        # The timestamp info for the training data
        self.train_ts_info = Timeseries.get_timestamp_info(self.data)
        # The timestamp info for both training and test data
        self.ts_info = self.train_ts_info.copy()
        # The lengths of test instances must be the same
        self.explainer = None
        self.test_ts_length = None

    def _build_predictor(self, ts_len):
        def _predict(xs: np.ndarray):
            xs = xs.reshape((xs.shape[0], ts_len, len(self.variables)))
            ts = [Timeseries.restore_timestamp_index(
                Timeseries(x, variable_names=self.variables), self.ts_info) for x in xs]
            return np.array([self.predict_function(t) for t in ts]).flatten()
        return _predict

    def _build_explainer(self, ts_len, num_samples=100):
        if self.explainer is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"

        interval = range(self.data.ts_len - ts_len)
        ps = random.sample(interval, min(num_samples, len(interval)))
        x = Timeseries.reset_timestamp_index(self.data, self.ts_info)
        self.explainer = shap.KernelExplainer(
            model=self._build_predictor(ts_len),
            data=np.array([x.values[p:p + ts_len].flatten() for p in ps]),
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

        self.ts_info = self.train_ts_info.copy()
        info = Timeseries.get_timestamp_info(X)
        self.ts_info["ts2val"].update(info["ts2val"])
        self.ts_info["val2ts"].update(info["val2ts"])

        data = Timeseries.reset_timestamp_index(X, self.ts_info)
        instances = data.values.reshape((1, -1))
        shap_values = self.explainer.shap_values(instances, **kwargs)
        shap_values = shap_values.reshape(data.shape)
        scores = pd.DataFrame(
            shap_values,
            columns=data.columns,
            index=X.index
        )
        explanations.add(X.to_pd(), scores)
        return explanations
