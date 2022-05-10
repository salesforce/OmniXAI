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
            The inputs of ``predict_function`` should be a batch (list) of time series, e.g.,
            an `Timeseries` instance. The outputs of ``predict_function`` are anomaly scores (higher scores
            imply more anomalous) for anomaly detection or predicted values for forecasting.
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
        self.variable_names = list(self.data.columns)
        # The lengths of test instances must be the same
        self.explainer = None
        self.test_ts_length = None

    def _build_predictor(self, ts_len):
        def _predict(xs: np.ndarray):
            ts = Timeseries(
                data=xs.reshape((xs.shape[0], ts_len, len(self.variable_names))),
                variable_names=self.variable_names
            )
            return self.predict_function(ts)
        return _predict

    def _build_explainer(self, ts_len, num_samples=100):
        if self.explainer is not None:
            return
        assert self.data.ts_len > ts_len, \
            "`ts_length` should be less than the length of the training time series"

        interval = range(self.data.ts_len - ts_len)
        ps = random.sample(interval, min(num_samples, len(interval)))
        self.explainer = shap.KernelExplainer(
            model=self._build_predictor(ts_len),
            data=np.array([self.data.values[0][p:p + ts_len].flatten() for p in ps]),
            link="identity"
        )
        self.test_ts_length = ts_len

    def explain(self, X: Timeseries, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`.
        :return: The feature-importance explanations for all the input instances.
        """
        # Initialize the SHAP explainer if it is not created.
        self._build_explainer(X.ts_len)
        explanations = FeatureImportance(self.mode)

        instances = X.values.reshape((X.batch_size, -1))
        shap_values = self.explainer.shap_values(instances, **kwargs)
        scores = Timeseries(
            data=shap_values.reshape(
                (shap_values.shape[0], self.test_ts_length, len(self.variable_names))),
            timestamps=X.index,
            variable_names=self.variable_names
        ).to_pd()

        ts = X.to_pd()
        if isinstance(ts, pd.DataFrame):
            explanations.add(ts, scores)
        else:
            for t, score in zip(ts, scores):
                explanations.add(t, score)
        return explanations
