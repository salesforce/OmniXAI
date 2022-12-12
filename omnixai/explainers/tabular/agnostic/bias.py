#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The model bias analyzer for tabular data.
"""
import numpy as np
from typing import List

from ...base import ExplainerBase
from ....data.tabular import Tabular


class BiasAnalyzer(ExplainerBase):
    """
    The bias analysis for a classification or regression model.
    """
    explanation_type = "global"
    alias = ["bias"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function,
            mode="classification",
            training_targets: List = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize the explainer.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param training_targets: The training labels/targets. If it is None, the target column in
            ``training_data`` will be used. The values of ``training_targets`` can only be integers
            (e.g., classification labels) or floats (regression targets).
        """
        super().__init__()
        assert mode in ["classification", "regression"], \
            "`BiasAnalyzer` only supports classification and regression models."
        assert predict_function is not None, \
            "`predict_function` is not None."
        if training_targets is None:
            assert training_data.target_column, \
                "`training_data` has no label/target column. Please set `training_targets`"
            assert training_data.to_pd(copy=False)[training_data.target_column] in \
                [np.int, np.int32, np.int64, np.float, np.float32, np.float64], \
                "The targets/labels in `training_data` must be either int or float."
            training_targets = training_data.get_target_column()
            training_data = training_data.remove_target_column()

        self.mode = mode
        self.predict_function = predict_function
        self.targets = training_targets
        self.preds = self._predict(training_data, batch_size=kwargs.get("batch_size", 64))

    def _predict(self, X: Tabular, batch_size=64):
        n, predictions = X.shape[0], []
        for i in range(0, n, batch_size):
            predictions.append(self.predict_function(X[i: i + batch_size]))
        z = np.concatenate(predictions, axis=0)
        return z.flatten() if self.mode == "regression" else np.argmax(z, axis=1)

    def explain(
            self,
            feature_column,
            feature_value,
            target_value_or_threshold,
            **kwargs
    ):
        pass
