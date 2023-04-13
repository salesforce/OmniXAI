#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Ranking Explainer for tabular data.
"""
import numpy as np
import scipy
import itertools
import pandas as pd
from typing import Callable, List
from ...base import ExplainerBase
from ....data.tabular import Tabular
from ....explanations.tabular.validity import ValidityRankingExplanation


class ValidityRankingExplainer(ExplainerBase):
    """
    Ranking Explainer for Tabular Data. If using this explainer, please cite the paper
    `Extracting per Query Valid Explanations for Blackbox Learning-to-Rank Models,
    https://dl.acm.org/doi/10.1145/3471158.3472241`.
    """

    explanation_type = "local"
    alias = ["validity"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            ignored_features: List = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize a Ranking explainer. ``training_data``
            can be the training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            the outputs of the ``predict_function`` are the ranking scores. The output must be
            a numpy array.
        :param ignored_features: The features ignored by the valid per-query algorithm.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), \
            "`training_data` should be an instance of Tabular."
        ignored_features = [] if ignored_features is None else ignored_features
        for f in ignored_features:
            assert f in training_data.columns, f"`training_data` has no feature {f}."

        self.predict_fn = predict_function
        self.features = [f for f in training_data.feature_columns if f not in ignored_features]
        self.stats_features = {
            "mean": self._compute_stats(training_data, np.mean),
            "median": self._compute_stats(training_data, np.median)
        }

    def _compute_stats(self, data: Tabular, func: Callable):
        stats, df = {}, data.to_pd(copy=False)
        for f in data.categorical_columns:
            if f in self.features:
                most_common_value = df[f].value_counts().idxmax()
                stats[f] = most_common_value
        for f in data.continuous_columns:
            if f in self.features:
                stats[f] = func(df[f].values.astype(float))
        return stats

    def _compute_mask(self, x, mask, idx):
        val = self.stats_features[mask][self.features[idx]] \
            if mask in self.stats_features else 0.0
        if isinstance(x, np.ndarray):
            x[:, :, idx] = val
        elif isinstance(x, pd.DataFrame):
            x[self.features[idx]] = val
        else:
            raise ValueError("Input must be either numpy array or pandas DataFrame")
        return x

    @staticmethod
    def _compute_pairs(n_items):
        positions = list(range(1, n_items + 1))
        combs = list(itertools.combinations(positions, r=2)) + \
                list(itertools.combinations(positions[::-1], r=2))
        pairs = np.array(list(filter(lambda x: x[0] < x[1], combs)))
        return pairs

    @staticmethod
    def _compute_rank(scores):
        return (-scores).flatten().argsort().argsort() + 1

    @staticmethod
    def _calculate_propensity(scores, ranks, i, j):
        return (scores[i] - scores[j]) * abs(ranks[i] - ranks[j])

    def _compute_validity(self, pi, minimal_features, mask, sample, categorical_cols):
        x = sample.copy()
        for i in range(len(self.features)):
            if i not in minimal_features:
                x = self._compute_mask(x, mask, i)
        scores = self.predict_fn(
            Tabular(x, categorical_columns=categorical_cols)
        ).flatten()
        ranks = self._compute_rank(scores)
        return {
            "Tau": scipy.stats.kendalltau(ranks, pi),
            "Weighted_Tau": scipy.stats.weightedtau(ranks, pi),
            "Top_K_Ranking": ranks,
            "Ranks": pi,
        }

    def explain(
            self,
            X: Tabular,
            k: int = 3,
            n_items: int = None,
            mask: str = "median",
            weighted: bool = False,
            epsilon: float = -1.0,
            query_id: str = None,
            verbose: bool = False,
            **kwargs
    ) -> ValidityRankingExplanation:
        """
        Generates the valid per-query feature-importance explanations for the input instances.

        :param X: A set of input items for a query.
        :param k: The maximum number of features to be accounted as explanation
        :param n_items: The number of items to be considered for the explanation
        :param mask: The type of feature masking to be performed (median, mode, None) default=median
        :param weighted: Flag for calculating weighted propensity
        :param epsilon: The epsilon value for the greedy-cover procedure. Negative epsilon will replace
            greedy-cover with simple greedy
        :param query_id: The feature column representing the query_id if present
        :param verbose: Flag for verbosity of print statements
        :return: The valid per-query feature-importance explanations for the given items.
        """
        assert mask in ["median", "mean", "zero"], \
            f"`mask` should be 'median', 'mean' or 'zero' instead of {mask}."
        if n_items is None or n_items <= 0:
            n_items = X.shape[0]
        if verbose:
            print("Num samples: ", n_items)

        pairs = self._compute_pairs(n_items)
        weights = np.array([(1 / p[0] + 1 / p[1]) for p in pairs])
        sample = X.to_pd(copy=False).iloc[:n_items]
        scores = self.predict_fn(
            Tabular(sample, categorical_columns=X.categorical_columns)
        )
        assert isinstance(scores, np.ndarray), \
            "The output of the prediction function should be a numpy array."
        pi = self._compute_rank(scores).tolist()
        rank2index = {p: i for i, p in enumerate(pi)}
        if verbose:
            print(f"Ranks of items from given model: {pi}")

        minimal_feat_set = {}
        max_utility = -np.inf
        for trials in range(k):
            propensity = {}
            utility = []
            for feature in range(len(self.features)):
                if feature not in minimal_feat_set:
                    x = sample.copy()
                    for i in range(len(self.features)):
                        if i != feature and i not in minimal_feat_set:
                            x = self._compute_mask(x, mask, i)
                    scores = self.predict_fn(
                        Tabular(x, categorical_columns=X.categorical_columns)
                    ).flatten()
                    ranks = self._compute_rank(scores)
                    propensity[feature] = [
                        self._calculate_propensity(scores, ranks, rank2index[p[0]], rank2index[p[1]])
                        for p in pairs
                    ]
                    if weighted:
                        utility.append(np.sum(weights * np.array(propensity[feature])))
                    else:
                        utility.append(sum(propensity[feature]))
                else:
                    utility.append(-np.inf)

            curr_max_utility = np.max(utility)
            curr_argmax_utility = int(np.argmax(utility))

            if epsilon >= 0.0:
                retain_indices = [z < epsilon for z in propensity[curr_argmax_utility]]
                pairs = pairs[retain_indices]
                weights = weights[retain_indices]
            elif curr_max_utility <= max_utility:
                break
            max_utility = max(curr_max_utility, max_utility)
            minimal_feat_set[curr_argmax_utility] = curr_max_utility
            if len(pairs) == 0:
                break
        validity = self._compute_validity(
            pi, minimal_feat_set, mask, sample, X.categorical_columns
        )
        minimal_feat_set = {self.features[u]: v for u, v in minimal_feat_set.items()}

        explanations = ValidityRankingExplanation()
        explanations.add(
            query=query_id,
            df=X.to_pd(),
            top_features=minimal_feat_set,
            validity=validity,
        )
        return explanations
