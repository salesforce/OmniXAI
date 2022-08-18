#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Ranking Explainer for tabular data.
"""
import numpy as np
from typing import Callable, List
from ...base import ExplainerBase
from ....data.tabular import Tabular
import scipy
import itertools
import pandas as pd

from ....explanations.ranking.agnostic.validity import ValidExplanation


class ValidityRankingExplainer(ExplainerBase):
    """
    Ranking Explainer for Tabular Data.
    """

    explanation_type = "local"
    alias = ["rank"]

    def __init__(
        self,
        training_data: Tabular,
        features: List,
        predict_function: Callable,
    ):
        """
        :param training_data: The data used to initialize a Ranking explainer. ``training_data``
            can be the training dataset for training the machine learning model.
        :param features: The list of features to be explained by the valid per-query algorithm
        :param predict_function: The prediction function corresponding to the model to explain.
            the outputs of the ``predict_function`` are the document scores.
        """
        super().__init__()
        assert isinstance(training_data, Tabular)
        for f in features:
            assert f in training_data.columns

        training_data = training_data.to_pd()
        self.features = features
        self.mean_features = self._compute_stats(
            training_data, self.features, "mean"
        )
        self.median_features = self._compute_stats(
            training_data, self.features, "median"
        )

        self.predict_fn = predict_function

    @staticmethod
    def _compute_stats(data: pd.DataFrame, features: List, mask_type):
        if mask_type == "mean":
            stats = data.mean()
        if mask_type == "median":
            stats = data.median()

        for f in features:
            if f not in stats:
                most_common_value = data[f].value_counts().idxmax()
                stats[f] = most_common_value

        return stats

    def _compute_mask(self, mask, idx):
        if mask == "mean":
            return self.mean_features[self.features[idx]]
        elif mask == "median":
            return self.median_features[self.features[idx]]
        else:
            return 0.0

    def compute_mask(self, x, mask, idx):
        if isinstance(x, np.ndarray):
            x[:, :, idx] = self._compute_mask(mask, idx)
        elif isinstance(x, pd.DataFrame):
            x[self.features[idx]] = self._compute_mask(mask, idx)
        else:
            raise ValueError("input must be either numpy array or pandas DataFrame")
        return x

    @staticmethod
    def compute_pairs(n_docs):
        positions = list(range(1, n_docs + 1))
        combs = list(itertools.combinations(positions, r=2)) + list(
            itertools.combinations(positions[::-1], r=2)
        )
        pairs = np.array(list(filter(lambda x: x[0] < x[1], combs)))
        return pairs

    @staticmethod
    def compute_rank(scores):
        return (-scores).flatten().argsort().argsort() + 1

    @staticmethod
    def calculate_propensity(scores, ranks, i, j):
        return (scores[i] - scores[j]) * abs(ranks[i] - ranks[j])

    def compute_validity(self, pi, minimal_features, mask, sample, categorical_cols):
        x = sample.copy()
        for i in range(0, len(self.features)):
            if i not in minimal_features:
                x = self.compute_mask(x, mask, i)
        scores = self.predict_fn(
            Tabular(x, categorical_columns=categorical_cols)
        ).flatten()
        ranks = self.compute_rank(scores)
        return {
            'Tau': scipy.stats.kendalltau(ranks, pi),
            'Weighted_Tau': scipy.stats.weightedtau(ranks, pi),
            'Top_K_Ranking': ranks,
            'Ranks': pi
        }

    def explain(
        self,
        tabular_data: Tabular,
        k: int = 3,
        n_docs: int = None,
        mask: str = "median",
        weighted: bool = False,
        epsilon: float = -1.0,
        query_feature: str = None,
        verbose: bool = False
    ) -> ValidExplanation:
        if not n_docs:
            n_docs = tabular_data.shape[0]
        if verbose:
            print("Num samples: ", n_docs)
        pairs = self.compute_pairs(n_docs)
        weights = np.array([(1 / p[0] + 1 / p[1]) for p in pairs])
        sample = tabular_data.to_pd().iloc[:n_docs]
        pi = self.compute_rank(
            self.predict_fn(
                Tabular(sample, categorical_columns=tabular_data.categorical_cols)
            )
        )
        pi = pi.tolist()
        if verbose:
            print(f"Ranks of documents from given model: {pi}")
        minimal_feat_set = {}
        max_utility = -np.inf
        for trials in range(0, k):
            propensity = {}
            utility = []
            for feature in range(0, len(self.features)):
                if feature not in minimal_feat_set:
                    propensity[feature] = []
                    x = sample.copy()
                    for i in range(0, len(self.features)):
                        if i != feature and i not in minimal_feat_set:
                            x = self.compute_mask(x, mask, i)
                    scores = self.predict_fn(
                        Tabular(x, categorical_columns=tabular_data.categorical_cols)
                    ).flatten()
                    ranks = (-scores).argsort().argsort() + 1
                    for p in pairs:
                        propensity[feature].append(
                            self.calculate_propensity(
                                scores, ranks, pi.index(p[0]), pi.index(p[1])
                            )
                        )
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
        validity = self.compute_validity(
            pi, minimal_feat_set, mask, sample, tabular_data.categorical_cols
        )
        minimal_feat_set = {self.features[u]: v for u, v in minimal_feat_set.items()}
        explanations = ValidExplanation()
        explanations.set(query=query_feature,
                         df=tabular_data.to_pd(),
                         top_features=minimal_feat_set,
                         validity=validity,
                         )
        return explanations
