#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The permutation feature importance explanation for tabular data.
"""
import numpy as np
import pandas as pd
from typing import Callable, Union
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance

from ..base import ExplainerBase, TabularExplainerMixin
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import GlobalFeatureImportance


class _Estimator:
    def fit(self):
        pass


class PermutationImportance(ExplainerBase, TabularExplainerMixin):
    """
    The permutation feature importance explanations for tabular data. The permutation feature
    importance is defined to be the decrease in a model score when a single feature value
    is randomly shuffled.
    """

    explanation_type = "global"
    alias = ["permutation"]

    def __init__(self, training_data: Tabular, predict_function, mode="classification", **kwargs):
        """
        :param training_data: The training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), \
            "training_data should be an instance of Tabular."
        assert mode in ["classification", "regression"], \
            "`mode` can only be `classification` or `regression`."

        self.categorical_columns = training_data.categorical_columns
        self.predict_function = predict_function
        self.mode = mode

    def _build_score_function(self, score_func=None):
        if score_func is not None:
            def _score(estimator, x, y):
                z = self.predict_function(
                    Tabular(x, categorical_columns=self.categorical_columns)
                )
                return score_func(y, z)
        elif self.mode == "classification":
            def _score(estimator, x, y):
                z = self.predict_function(
                    Tabular(x, categorical_columns=self.categorical_columns)
                )
                return -log_loss(y, z)
        else:
            def _score(estimator, x, y):
                z = self.predict_function(
                    Tabular(x, categorical_columns=self.categorical_columns)
                )
                return -np.mean((z - y) ** 2)
        return _score

    def explain(
            self,
            X: Tabular,
            y: Union[np.ndarray, pd.DataFrame],
            n_repeats: int = 30,
            score_func: Callable = None
    ) -> GlobalFeatureImportance:
        """
        Generate permutation feature importance scores.

        :param X: Data on which permutation importance will be computed.
        :param y: Targets or labels.
        :param n_repeats: The number of times a feature is randomly shuffled.
        :param score_func: The score function measuring the difference between
            ground-truth targets and predictions, e.g., -sklearn.metrics.log_loss(y_true, y_pred).
        :return: The permutation feature importance explanations.
        """
        assert X is not None and y is not None, \
            "The test data `X` and target `y` cannot be None."
        y = y.values if isinstance(y, pd.DataFrame) else np.array(y)
        if y.ndim > 1:
            y = y.flatten()
        assert X.shape[0] == len(y), \
            "The numbers of samples in `X` and `y` are different."
        X = X.remove_target_column()

        results = permutation_importance(
            estimator=_Estimator(),
            X=X.to_pd(copy=False),
            y=y,
            scoring=self._build_score_function(score_func)
        )
        explanations = GlobalFeatureImportance()
        explanations.add(
            feature_names=list(X.columns),
            importance_scores=results["importances_mean"]
        )
        return explanations
