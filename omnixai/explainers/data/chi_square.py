#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for computing chi-squared stats between each non-negative feature and target.
"""
import warnings
import numpy as np
from sklearn.feature_selection import chi2

from ..base import ExplainerBase
from ...data.tabular import Tabular
from ...preprocessing.tabular import TabularTransform
from ...preprocessing.encode import Ordinal, KBins, LabelEncoder
from ...explanations.tabular.feature_importance import GlobalFeatureImportance


class ChiSquare(ExplainerBase):
    """
    The class for computing chi-squared stats between each non-negative feature and target.
    """

    explanation_type = "global"
    alias = ["chi2", "chi_square"]

    def __init__(self, training_data: Tabular, mode="classification", **kwargs):
        """
        :param training_data: The dataset for training an ML model.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), "training_data should be an instance of Tabular."
        assert training_data.target_column is not None, "training_data has no target/label column."
        cont_transform = KBins(n_bins=10)
        target_transform = LabelEncoder() if mode == "classification" else KBins(n_bins=10)
        self.transformer = TabularTransform(
            cate_transform=Ordinal(), cont_transform=cont_transform, target_transform=target_transform
        ).fit(training_data)

        self.mode = mode
        self.feature_columns = self.transformer.get_feature_names()
        self.data = self.transformer.transform(training_data)

        if mode == "classification" and len(set(self.data[:, -1])) > 100:
            warnings.warn(f"There are more than 100 classes. Please check if it is a regression task."
                          f"If the task is regression, please set `mode` to `regression` instead of `classification`.")

    def explain(self, **kwargs):
        """
        Computes chi-squared stats between each non-negative feature and target.

        :return: The chi-squared stats.
        :rtype: GlobalFeatureImportance
        """
        explanations = GlobalFeatureImportance()
        importance_scores = chi2(self.data[:, :-1], self.data[:, -1])[0]
        importance_scores = np.nan_to_num(importance_scores, nan=0)
        explanations.add(self.feature_columns, importance_scores)
        return explanations
