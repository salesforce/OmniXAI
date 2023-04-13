#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for estimating mutual information.
"""
import warnings
from sklearn.feature_selection import mutual_info_classif

from ..base import ExplainerBase
from ...data.tabular import Tabular
from ...preprocessing.tabular import TabularTransform
from ...preprocessing.base import Identity
from ...preprocessing.encode import Ordinal, KBins, LabelEncoder
from ...explanations.tabular.feature_importance import GlobalFeatureImportance


class MutualInformation(ExplainerBase):
    """
    The class for estimating mutual information. It computes the Information gain of each feature
    with respect to the target.
    """

    explanation_type = "global"
    alias = ["mutual"]

    def __init__(self, training_data: Tabular, mode="classification", discrete=False, **kwargs):
        """
        :param training_data: The dataset for training an ML model.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param discrete: `True` if all the continuous-valued features are discretized or
            `False` if all the categorical features are converted into continuous-valued features.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), "training_data should be an instance of Tabular."
        assert training_data.target_column is not None, "training_data has no target/label column."
        cont_transform = KBins(n_bins=10) if discrete else Identity()
        target_transform = LabelEncoder() if mode == "classification" else KBins(n_bins=10)
        self.transformer = TabularTransform(
            cate_transform=Ordinal(), cont_transform=cont_transform, target_transform=target_transform
        ).fit(training_data)

        self.mode = mode
        self.discrete = discrete
        self.feature_columns = self.transformer.get_feature_names()
        self.data = self.transformer.transform(training_data)

        if mode == "classification" and len(set(self.data[:, -1])) > 100:
            warnings.warn(f"There are more than 100 classes. Please check if it is a regression task."
                          f"If the task is regression, please set `mode` to `regression` instead of `classification`.")

    def explain(self, **kwargs):
        """
        Computes the mutual information between each feature and the target.

        :return: The mutual information between each feature and the target.
        :rtype: GlobalFeatureImportance
        """
        explanations = GlobalFeatureImportance()
        importance_scores = mutual_info_classif(
            self.data[:, :-1], self.data[:, -1], discrete_features=self.discrete)
        explanations.add(self.feature_columns, importance_scores)
        return explanations
