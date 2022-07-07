#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for checking feature imbalances.
"""
import pandas as pd
from typing import Sequence
from collections import defaultdict

from ..base import ExplainerBase
from ...data.tabular import Tabular
from ...preprocessing.tabular import TabularTransform
from ...preprocessing.encode import Ordinal, KBins
from omnixai.explanations.tabular.imbalance import ImbalanceExplanation


class ImbalanceAnalyzer(ExplainerBase):
    """
    The class for checking feature imbalances. It counts the appearances
    of each feature value in different classes. For example, if the feature to analyze is
    "gender", it computes the counts of "gender = male" and "gender = female" for each class
    separately. If the features to analyze are ["gender", "age"], it will count the cross-feature
    values.
    """

    explanation_type = "global"
    alias = ["imbalance"]

    def __init__(self, training_data: Tabular, mode: str = "classification", n_bins: int = 10, **kwargs):
        """
        :param training_data: The dataset for training an ML model.
        :param mode: The task type can be `classification` only.
        :param n_bins: The number of bins for discretizing continuous-valued features.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), "training_data should be an instance of Tabular."
        assert mode == "classification", "ImbalanceAnalyzer only support classification tasks."
        self.training_data = training_data
        self.transformer = TabularTransform(cate_transform=Ordinal(), cont_transform=KBins(n_bins=n_bins)).fit(
            training_data
        )
        self.feature_columns = self.transformer.get_feature_names()

    def _split_by_label(self):
        """
        Splits the dataset by class labels.

        :return: The dataset splits.
        :rtype: Dict
        """
        if self.training_data.target_column is None:
            return self.training_data
        else:
            target_column = self.training_data.target_column
            df = self.training_data.to_pd()
            labels = set(df[target_column].values)
            return {
                label: Tabular(
                    data=df[df[target_column] == label],
                    categorical_columns=self.training_data.categorical_columns,
                    target_column=target_column,
                )
                for label in labels
            }

    def _get_counts_per_class(self, tabular_data, features):
        """
        Counts the appearances of each feature or cross-feature value.

        :param tabular_data: The input tabular data.
        :param features: A list of features to analyze.
        :return: A list of tuples with format `(feature-values, count)`.
        """
        feature2index = {f: i for i, f in enumerate(self.feature_columns)}
        df = pd.DataFrame(
            data=self.transformer.transform(tabular_data.remove_target_column()), columns=self.feature_columns
        )
        counts = df.groupby(by=list(features))[features[0]].count()

        x, group_counts, results = df.values[0], [], []
        for values, count in zip(counts.keys(), counts.values):
            if not isinstance(values, (list, tuple)):
                values = [values]
            for feat, value in zip(features, values):
                x[feature2index[feat]] = value
            y = self.transformer.invert(x).to_pd(copy=False)
            results.append((y[features].values[0].tolist(), count))
        return results

    def _get_counts(self, features):
        """
        Counts the appearances of each feature value in different classes.

        :param features: A list of features to analyze.
        :return: A list of tuples with format `(feature-values, count)` where
            `count` is a number if there is a single class or a dict whose keys are labels
            and values are counts if there are multiple classes.
        """
        splits = self._split_by_label()
        if not isinstance(splits, dict):
            return self._get_counts_per_class(splits, features)

        results = defaultdict(list)
        for label, data in splits.items():
            counts = self._get_counts_per_class(data, features)
            for values, count in counts:
                key = "#".join([str(v) for v in values])
                results[key].append({"values": values, "label": label, "count": count})

        counts = []
        labels = list(splits.keys())
        for key, items in results.items():
            c = {label: 0 for label in labels}
            for item in items:
                c[item["label"]] = item["count"]
            counts.append((items[0]["values"], c))
        return counts

    def explain(self, features: Sequence, **kwargs):
        """
        Computes the count for each cross-feature.

        :param features: A list of features to analyze.
        :return: The counts for each feature or cross-feature values in different classes.
        :rtype: ImbalanceExplanation
        """
        explanations = ImbalanceExplanation()
        counts = self._get_counts(features)
        for values, count in counts:
            explanations.add(feature=values, count=count)
        return explanations
