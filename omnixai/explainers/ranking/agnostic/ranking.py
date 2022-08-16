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
        preprocessing_fn: Callable = None,
        **kwargs
    ):
        """
        :param training_data: The data used to initialize a Ranking explainer. ``training_data``
            can be the training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            the outputs of the ``predict_function`` are the document scores.
        """
        super().__init__()
        self.training_data = training_data.to_pd()
        self.features = features
        self.mean_features = self._compute_stats(self.training_data, self.features, 'mean')
        self.median_features = self._compute_stats(self.training_data, self.features, 'median')

        if preprocessing_fn:
            self.preprocessing_fn = preprocessing_fn
        else:
            self.preprocessing_fn = lambda x: x.to_pd()
        self.predict_fn = predict_function

    @staticmethod
    def _compute_stats(data: pd.DataFrame, features: List, mask_type):
        if mask_type == 'mean':
            stats = data.mean()
        if mask_type == 'median':
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

    @staticmethod
    def _create_copy(sample):
        if type(sample) == np.ndarray or type(sample) == pd.DataFrame:
            x = sample.copy()
        else:
            x = sample.clone()
        return x

    def compute_mask(self, x, mask, idx):
        if type(x) == np.ndarray:
            x[:, :, idx] = self._compute_mask(mask, idx)
        elif type(x) == pd.DataFrame:
            x[self.features[idx]] = self._compute_mask(mask, idx)
        else:
            raise Exception("input must be either numpy array or pandas DataFrame")
        return x

    @staticmethod
    def compute_pairs(num_samples):
        positions = list(range(1, num_samples + 1))
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
        x = self._create_copy(sample)
        for i in range(0, len(self.features)):
            if i not in minimal_features:
                x = self.compute_mask(x, mask, i)
        scores = self.predict_fn(Tabular(x, categorical_columns=categorical_cols)).flatten()
        print(scores)
        ranks = self.compute_rank(scores)
        print(ranks)
        return scipy.stats.kendalltau(ranks, pi), scipy.stats.weightedtau(ranks, pi)

    @staticmethod
    def compute_num_samples(tabular_data: Tabular):
        return tabular_data.to_pd().shape[0]

    def explain(
        self,
        tabular_data: Tabular,
        k: int = 3,
        num_samples: int = None,
        mask: str = "median",
        weighted: bool = False,
        epsilon: float = -1.0,
    ):
        if not num_samples:
            num_samples = self.compute_num_samples(tabular_data)
        print('Num samples: ', num_samples)
        pairs = self.compute_pairs(num_samples)
        weights = np.array([(1 / p[0] + 1 / p[1]) for p in pairs])
        sample = self.preprocessing_fn(tabular_data).iloc[:num_samples]
        pi = self.compute_rank(self.predict_fn(Tabular(sample, categorical_columns=tabular_data.categorical_cols)))
        pi = pi.tolist()
        print(pi)
        minimal_feat_set = {}
        max_utility = -np.inf
        validity = None
        for trials in range(0, k):
            propensity = {}
            utility = []
            for feature in range(0, len(self.features)):
                if feature not in minimal_feat_set:
                    propensity[feature] = []
                    x = self._create_copy(sample)
                    for i in range(0, len(self.features)):
                        if i != feature and i not in minimal_feat_set:
                            x = self.compute_mask(x, mask, i)
                    scores = self.predict_fn(Tabular(x, categorical_columns=tabular_data.categorical_cols)).flatten()
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
            validity = self.compute_validity(pi, minimal_feat_set, mask, sample, tabular_data.categorical_cols)
            if len(pairs) == 0:
                break
        minimal_feat_set = {self.features[u]: v for u, v in minimal_feat_set.items()}
        return minimal_feat_set, validity


