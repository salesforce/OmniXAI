#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The model bias analyzer for tabular data.
"""
import numpy as np
from typing import List
from collections import defaultdict

from ...base import ExplainerBase
from ....data.tabular import Tabular
from ....explanations.tabular.bias import BiasExplanation


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
        self.data = training_data.to_pd(copy=False)
        self.cate_columns = training_data.categorical_columns
        self.predict_function = predict_function
        self.targs = np.array(training_targets)
        self.preds = self._predict(training_data, batch_size=kwargs.get("batch_size", 128))

    def _predict(self, X: Tabular, batch_size=128):
        n, predictions = X.shape[0], []
        for i in range(0, n, batch_size):
            predictions.append(self.predict_function(X[i: i + batch_size]))
        z = np.concatenate(predictions, axis=0)
        return z.flatten() if self.mode == "regression" else np.argmax(z, axis=1)

    def _predictions_by_groups(self, group_a, group_b, targets):
        if not isinstance(targets, (list, tuple, np.ndarray)):
            targets = [targets]
        targ_a, targ_b = self.targs[group_a], self.targs[group_b]
        pred_a, pred_b = self.preds[group_a], self.preds[group_b]
        return targ_a, targ_b, pred_a, pred_b, targets

    def explain(
            self,
            feature_column,
            feature_value_or_threshold,
            label_value_or_threshold,
            **kwargs
    ) -> BiasExplanation:
        """
        Runs bias analysis on the given model and dataset.

        :param feature_column: The feature column to analyze.
        :param feature_value_or_threshold: The feature value for a categorical feature or feature value
            threshold for a continuous-value feature. It can either be a single value or a list/tuple.
            When it is a single value, (a) for categorical features, the advantaged group will be those samples contains
            this feature value and the disadvantaged group will be the other samples, (b) for continuous-valued
            features, the advantaged group will be those samples whose values of `feature_column` <= `feature_value_or_threshold`
            and the disadvantaged group will be the other samples. When it is a list/tuple, (a) for categorical features,
            the advantaged group will be the samples contains the feature values in the first element in the list and
            the disadvantaged group will be the samples contains the feature values in the second element in the list.
            (b) for continuous-valued features, if `feature_value_or_threshold` is [a, b], then the advantaged group
            will be the samples whose values of `feature_column` <= a and the disadvantaged group will be the samples
            whose values of `feature_column` > b. If `feature_value_or_threshold` is [a, [b, c]], the disadvantaged
            group will be the samples whose values of `feature_column` is in (b, c].
        :param label_value_or_threshold: The target label for classification or target
            threshold for regression. For regression, it will be converted into a binary classification
            problem when computing bias metrics, i.e., label = 0 if target value <= target_value_or_threshold,
            and label = 1 if target value > target_value_or_threshold.
        :return: The bias analysis results stored in ``BiasExplanation``.
        """
        assert feature_column in self.data, \
            f"Feature column {feature_column} does not exist."
        assert feature_value_or_threshold is not None, \
            "`feature_value_or_threshold` cannot be None."
        if isinstance(feature_value_or_threshold, (list, tuple)):
            assert len(feature_value_or_threshold) == 2, \
                "`feature_value_or_threshold` is either a single value or a list/tuple " \
                "of two lists indicating two feature groups, e.g., `feature_value_or_threshold = 'X'` " \
                "or `feature_value_or_threshold = (['X', 'Y'], 'Z')`."
        assert label_value_or_threshold is not None, \
            "`label_value_or_threshold` cannot be None."

        group_a, group_b = [], []
        if feature_column in self.cate_columns:
            feat_value2idx = defaultdict(list)
            for i, feat in enumerate(self.data[feature_column].values):
                feat_value2idx[feat].append(i)
            if isinstance(feature_value_or_threshold, (list, tuple)):
                features = feature_value_or_threshold[0] if \
                    isinstance(feature_value_or_threshold[0], (list, tuple)) \
                    else [feature_value_or_threshold[0]]
                for feat in features:
                    assert feat in feat_value2idx, f"Feature {feat} does not exist."
                    group_a += feat_value2idx[feat]
                features = feature_value_or_threshold[1] if \
                    isinstance(feature_value_or_threshold[1], (list, tuple)) \
                    else [feature_value_or_threshold[1]]
                for feat in features:
                    assert feat in feat_value2idx, f"Feature {feat} does not exist."
                    group_b += feat_value2idx[feat]
            else:
                assert feature_value_or_threshold in feat_value2idx, \
                    f"Feature {feature_value_or_threshold} does not exist."
                group_a = feat_value2idx[feature_value_or_threshold]
                for feat, indices in feat_value2idx.items():
                    if feat != feature_value_or_threshold:
                        group_b += indices
        else:
            values = self.data[feature_column].values
            if isinstance(feature_value_or_threshold, (list, tuple)):
                a = feature_value_or_threshold[0]
                b = feature_value_or_threshold[1]
                if not isinstance(a, (list, tuple)):
                    group_a = [i for i, v in enumerate(values) if v <= a]
                else:
                    assert len(a) == 2, \
                        "The element in `feature_value_or_threshold` is either a number " \
                        "or a tuple `(min_value, max_value)`."
                    group_a = [i for i, v in enumerate(values) if a[0] < v <= a[1]]
                if not isinstance(b, (list, tuple)):
                    group_b = [i for i, v in enumerate(values) if v > b]
                else:
                    assert len(b) == 2, \
                        "The element in `feature_value_or_threshold` is either a number " \
                        "or a tuple `(min_value, max_value)`."
                    group_b = [i for i, v in enumerate(values) if b[0] < v <= b[1]]
            else:
                assert type(feature_value_or_threshold) in [int, float], \
                    "For continuous-valued features, if `feature_value_or_threshold` is not a list, " \
                    "it must be either int or float."
                group_a = [i for i, v in enumerate(values) if v <= feature_value_or_threshold]
                group_b = [i for i, v in enumerate(values) if v > feature_value_or_threshold]

        assert len(group_a) > 0, "The 1st group (advantaged group) for bias analysis is empty."
        assert len(group_b) > 0, "The 2nd group (disadvantaged group) for bias analysis is empty."
        metric_class = _BiasMetricsForClassification if self.mode == "classification" \
            else _BiasMetricsForRegression
        targ_a, targ_b, pred_a, pred_b, targets = \
            self._predictions_by_groups(group_a, group_b, label_value_or_threshold)

        explanations = BiasExplanation(mode=self.mode)
        stats = metric_class.compute_stats(targ_a, targ_b, pred_a, pred_b, targets)
        for metric_name in ["DPL", "DI", "DCO", "RD", "DLR", "AD", "TE", "CDDPL"]:
            func = getattr(metric_class, f"{metric_name.lower()}")
            explanations.add(metric_name, func(stats, self.preds, len(pred_a), len(pred_b), targets))
        return explanations


class _BiasMetricsForClassification:

    @staticmethod
    def compute_stats(targ_a, targ_b, pred_a, pred_b, labels):
        stats = defaultdict(dict)
        for label in labels:
            stats[label]["na"] = len([x for x in targ_a if x == label])
            stats[label]["nb"] = len([x for x in targ_b if x == label])
            stats[label]["na_hat"] = len([x for x in pred_a if x == label])
            stats[label]["nb_hat"] = len([x for x in pred_b if x == label])
            stats[label]["tpa"] = len([x for x, y in zip(targ_a, pred_a) if x == label and y == label])
            stats[label]["fpa"] = len([x for x, y in zip(targ_a, pred_a) if x != label and y == label])
            stats[label]["fna"] = len([x for x, y in zip(targ_a, pred_a) if x == label and y != label])
            stats[label]["tpb"] = len([x for x, y in zip(targ_b, pred_b) if x == label and y == label])
            stats[label]["fpb"] = len([x for x, y in zip(targ_b, pred_b) if x != label and y == label])
            stats[label]["fnb"] = len([x for x, y in zip(targ_b, pred_b) if x == label and y != label])
            stats[label]["acc_a"] = np.sum((targ_a == pred_a).astype(int)) / len(pred_a)
            stats[label]["acc_b"] = np.sum((targ_b == pred_b).astype(int)) / len(pred_b)
        return stats

    @staticmethod
    def dpl(stats, pred_all, len_a, len_b, labels):
        """
        Difference in proportions in predicted labels
        """
        return {label: stats[label]["na_hat"] / len_a -
                       stats[label]["nb_hat"] / len_b
                for label in labels}

    @staticmethod
    def di(stats, pred_all, len_a, len_b, labels):
        """
        Disparate Impact.
        """
        return {label: (stats[label]["nb_hat"] / len_b) /
                       (stats[label]["na_hat"] / len_a + 1e-4)
                for label in labels}

    @staticmethod
    def dco(stats, pred_all, len_a, len_b, labels):
        """
        Difference in conditional outcomes.
        """
        return {label: stats[label]["na"] / max(stats[label]["na_hat"], 1) -
                       stats[label]["nb"] / max(stats[label]["nb_hat"], 1)
                for label in labels}

    @staticmethod
    def rd(stats, pred_all, len_a, len_b, labels):
        """
        Recall difference.
        """
        return {label: stats[label]["tpa"] / max(stats[label]["na"], 1) -
                       stats[label]["tpb"] / max(stats[label]["nb"], 1)
                for label in labels}

    @staticmethod
    def dlr(stats, pred_all, len_a, len_b, labels):
        """
        Difference in Label rates (precision difference).
        """
        return {label: stats[label]["tpa"] / max(stats[label]["na_hat"], 1) -
                       stats[label]["tpb"] / max(stats[label]["nb_hat"], 1)
                for label in labels}

    @staticmethod
    def ad(stats, pred_all, len_a, len_b, labels):
        """
        Accuracy difference.
        """
        return {label: stats[label]["acc_a"] - stats[label]["acc_b"]
                for label in labels}

    @staticmethod
    def te(stats, pred_all, len_a, len_b, labels):
        """
        Treatment equality.
        """
        return {label: stats[label]["fnb"] / max(stats[label]["fpb"], 1) -
                       stats[label]["fna"] / max(stats[label]["fpa"], 1)
                for label in labels}

    @staticmethod
    def cddpl(stats, pred_all, len_a, len_b, labels):
        """
        Conditional demographic disparity of predicted labels.
        """
        metrics = {}
        for label in labels:
            na1 = stats[label]["na_hat"]
            nb1 = stats[label]["nb_hat"]
            na0 = len_a - stats[label]["na_hat"]
            nb0 = len_b - stats[label]["nb_hat"]
            n0 = max(len([x for x in pred_all if x != label]), 1)
            n1 = max(len(pred_all) - n0, 1)
            s = (na0 / n0 - na1 / n1) * len_a + (nb0 / n0 - nb1 / n1) * len_b
            metrics[label] = s / (len_a + len_b)
        return metrics


class _BiasMetricsForRegression(_BiasMetricsForClassification):

    @staticmethod
    def compute_stats(targ_a, targ_b, pred_a, pred_b, targets):
        stats = defaultdict(dict)
        for target in targets:
            stats[target]["na"] = len([x for x in targ_a if x <= target])
            stats[target]["nb"] = len([x for x in targ_b if x <= target])
            stats[target]["na_hat"] = len([x for x in pred_a if x <= target])
            stats[target]["nb_hat"] = len([x for x in pred_b if x <= target])
            stats[target]["tpa"] = len([x for x, y in zip(targ_a, pred_a) if x <= target and y <= target])
            stats[target]["fpa"] = len([x for x, y in zip(targ_a, pred_a) if x > target and y <= target])
            stats[target]["fna"] = len([x for x, y in zip(targ_a, pred_a) if x <= target and y > target])
            stats[target]["tpb"] = len([x for x, y in zip(targ_b, pred_b) if x <= target and y <= target])
            stats[target]["fpb"] = len([x for x, y in zip(targ_b, pred_b) if x > target and y <= target])
            stats[target]["fnb"] = len([x for x, y in zip(targ_b, pred_b) if x <= target and y > target])
            stats[target]["acc_a"] = np.sum(((targ_a <= target) == (pred_a <= target)).astype(int)) / len(pred_a)
            stats[target]["acc_b"] = np.sum(((targ_b <= target) == (pred_b <= target)).astype(int)) / len(pred_b)
        return stats
