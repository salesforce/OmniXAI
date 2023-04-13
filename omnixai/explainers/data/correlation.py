#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for feature correlation analysis.
"""
import numpy as np
import pandas as pd
from typing import Sequence
from scipy.stats import spearmanr

from ..base import ExplainerBase
from ...data.tabular import Tabular
from ...preprocessing.tabular import TabularTransform
from ...preprocessing.base import Identity
from ...preprocessing.encode import Ordinal
from ...explanations.tabular.correlation import CorrelationExplanation


class CorrelationAnalyzer(ExplainerBase):
    """
    The class for feature correlation analysis. It computes the feature correlation matrix
    given the input dataset.
    """

    explanation_type = "global"
    alias = ["correlation"]

    def __init__(self, training_data: Tabular, **kwargs):
        """
        :param training_data: The dataset for training an ML model.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), "training_data should be an instance of Tabular."
        self.transformer = TabularTransform(cate_transform=Ordinal(), cont_transform=Identity()).fit(training_data)
        self.feature_columns = self.transformer.get_feature_names()
        self.df = pd.DataFrame(
            data=self.transformer.transform(training_data.remove_target_column()), columns=self.feature_columns
        )

    def explain(self, features: Sequence = None, **kwargs):
        """
        Computes the correlation matrix via scipy.stats.spearmanr.

        :param features: A list of feature to analyze or `None` if all the features are considered.
        :return: The feature correlation matrix.
        :rtype: CorrelationExplanation
        """
        explanations = CorrelationExplanation()
        if features is None:
            df = self.df
            features = df.columns
        else:
            assert len(features) > 1, "The number of features to analyze should be greater than 1."
            df = self.df[features]
        num_feats = len(features)
        corr = spearmanr(df).correlation
        if num_feats == 2:
            corr = np.array([[1, corr], [corr, 1]])

        explanations.add(features=features, correlation=corr)
        return explanations
