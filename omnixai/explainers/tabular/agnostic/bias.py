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
from collections import defaultdict

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
        self.data = training_data.to_pd(copy=False)
        self.predict_function = predict_function
        self.targs = np.array(training_targets)
        self.preds = self._predict(training_data, batch_size=kwargs.get("batch_size", 64))
        self.all_labels = list(set(self.targs.astype(int))) if mode == "classification" \
            else self.targs

    def _predict(self, X: Tabular, batch_size=64):
        n, predictions = X.shape[0], []
        for i in range(0, n, batch_size):
            predictions.append(self.predict_function(X[i: i + batch_size]))
        z = np.concatenate(predictions, axis=0)
        return z.flatten() if self.mode == "regression" else np.argmax(z, axis=1)

    def _get_labels(self, group_a, group_b, labels=None):
        if labels is None:
            labels = self.all_labels
        if not isinstance(labels, (list, tuple, np.ndarray)):
            labels = [labels]
        targ_a, targ_b = self.targs[group_a], self.targs[group_b]
        pred_a, pred_b = self.preds[group_a], self.preds[group_b]
        return targ_a, targ_b, pred_a, pred_b, labels

    def explain(
            self,
            feature_column,
            feature_value_or_groups,
            target_value_or_threshold=None,
            **kwargs
    ):
        assert feature_column in self.data, \
            f"Feature column {feature_column} does not exist."
        if isinstance(feature_value_or_groups, (list, tuple)):
            assert len(feature_value_or_groups) == 2, \
                "`feature_value_or_groups` is either a single value or a list/tuple " \
                "of two lists indicating two feature groups, e.g., `feature_value_or_groups = 'X'` " \
                "or `feature_value_or_groups = (['X', 'Y'], ['Z'])`."

        feat_value2idx = defaultdict(list)
        for i, feat in enumerate(self.data[feature_column].values):
            feat_value2idx[feat].append(i)
        if isinstance(feature_value_or_groups, (list, tuple)):
            for feat in feature_value_or_groups[0]:
                assert feat in feat_value2idx, f"Feature {feat} does not exist."
            for feat in feature_value_or_groups[1]:
                assert feat in feat_value2idx, f"Feature {feat} does not exist."
        else:
            assert feature_value_or_groups in feat_value2idx, \
                f"Feature {feature_value_or_groups} does not exist."

        group_a, group_b = [], []
        if isinstance(feature_value_or_groups, (list, tuple)):
            for feat in feature_value_or_groups[0]:
                group_a += feat_value2idx[feat]
            for feat in feature_value_or_groups[1]:
                group_b += feat_value2idx[feat]
        else:
            group_a = feat_value2idx[feature_value_or_groups]
            for feat, indices in feat_value2idx.items():
                if feat != feature_value_or_groups:
                    group_b += indices

        targ_a, targ_b, pred_a, pred_b, labels = \
            self._get_labels(group_a, group_b, target_value_or_threshold)

        res = {}
        stats = self._compute_stats(targ_a, targ_b, pred_a, pred_b, labels)
        for metric_name in ["DPL", "DI", "DCO", "RD", "DLR", "AD"]:
            func = getattr(BiasAnalyzer, f"_{metric_name.lower()}")
            res[metric_name] = func(stats, targ_a, targ_b, pred_a, pred_b, labels)
        print(res)

    @staticmethod
    def _compute_stats(targ_a, targ_b, pred_a, pred_b, labels):
        stats = defaultdict(dict)
        for label in labels:
            stats[label]["na"] = len([x for x in targ_a if x == label])
            stats[label]["nb"] = len([x for x in targ_b if x == label])
            stats[label]["na_hat"] = len([x for x in pred_a if x == label])
            stats[label]["nb_hat"] = len([x for x in pred_b if x == label])
            stats[label]["tpa"] = len([x for x, y in zip(targ_a, pred_a) if x == label and y == label])
            stats[label]["tpb"] = len([x for x, y in zip(targ_b, pred_b) if x == label and y == label])
            stats[label]["acc_a"] = np.sum((targ_a == pred_a).astype(int)) / len(pred_a)
            stats[label]["acc_b"] = np.sum((targ_b == pred_b).astype(int)) / len(pred_b)
        return stats

    @staticmethod
    def _dpl(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Difference in proportions in predicted labels
        """
        return {label: stats[label]["na_hat"] / len(pred_a) -
                       stats[label]["nb_hat"] / len(pred_b)
                for label in labels}

    @staticmethod
    def _di(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Disparate Impact.
        """
        return {label: stats[label]["nb_hat"] / len(pred_b) /
                       (stats[label]["na_hat"] / len(pred_a) + 1e-8)
                for label in labels}

    @staticmethod
    def _dco(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Difference in conditional outcomes.
        """
        return {label: stats[label]["na"] / max(stats[label]["na_hat"], 1) -
                       stats[label]["nb"] / max(stats[label]["nb_hat"], 1)
                for label in labels}

    @staticmethod
    def _rd(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Recall difference.
        """
        return {label: stats[label]["tpa"] / stats[label]["na"] -
                       stats[label]["tpb"] / stats[label]["nb"]
                for label in labels}

    @staticmethod
    def _dlr(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Difference in Label rates (precision difference).
        """
        return {label: stats[label]["tpa"] / max(stats[label]["na_hat"], 1) -
                       stats[label]["tpb"] / max(stats[label]["nb_hat"], 1)
                for label in labels}

    @staticmethod
    def _ad(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Accuracy difference.
        """
        return {label: stats[label]["acc_a"] - stats[label]["acc_b"]
                for label in labels}

    @staticmethod
    def _te(stats, targ_a, targ_b, pred_a, pred_b, labels):
        """
        Treatment equality.
        """