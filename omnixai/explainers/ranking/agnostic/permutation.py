#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature Permutation Ranking Explainer for tabular data.
"""
import numpy as np
import itertools
import pandas as pd
from typing import Callable, List
from ...base import ExplainerBase
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import GlobalFeatureImportance
from tqdm import tqdm


class PermutationRankingExplainer(ExplainerBase):
    """
    Feature Permutation Ranking Explainer for Tabular Data.
    """

    explanation_type = "local"
    alias = ["permutation"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            ignored_features: List = None,
            random_state: int = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize a Ranking explainer. ``training_data``
            can be the training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            the outputs of the ``predict_function`` are the ranking scores. The output must be
            a numpy array.
        :param ignored_features: The features ignored by the permutation per-query algorithm.
        """
        super().__init__()
        ignored_features = [] if ignored_features is None else ignored_features
        for f in ignored_features:
            assert f in training_data.columns, f"`training_data` has no feature {f}."

        self.predict_fn = predict_function
        self.features = [f for f in training_data.feature_columns if f not in ignored_features]
        self.random_state = random_state

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
    def _calculate_propensity(scores, ranks, i, j, ideal_scores, quotient):
        propensity = (scores[i] - scores[j]) * abs(ranks[i] - ranks[j])
        score_diff = (ideal_scores[i] - ideal_scores[j]) * abs(ranks[i] - ranks[j])
        if quotient:
            return abs(propensity) / (abs(score_diff) + 1e-6)
        return abs(score_diff - propensity)

    def _permute(self, x, idx):
        if isinstance(x, np.ndarray):
            x[:, :, idx] = np.random.permutation(x[:, :, idx])
        elif isinstance(x, pd.DataFrame):
            x[self.features[idx]] = x[self.features[idx]].sample(
                frac=1.0, random_state=self.random_state).values
        else:
            raise Exception("Input must be either numpy array or pandas DataFrame")
        return x

    def explain(
            self,
            X: Tabular,
            n_items: int = None,
            weighted: bool = False,
            quotient: bool = False,
            query_id: str = None,
            n_iter: int = 100,
            verbose: bool = False,
            **kwargs
    ) -> GlobalFeatureImportance:
        """
        Generates the permutation based per-query feature-importance explanations for the input instances.

        :param X: A set of input items for a query.
        :param n_items: The number of items to be considered for the explanation
        :param weighted: Flag for calculating weighted propensity
        :param quotient: Flag for propensity normalization
        :param query_id: The feature column representing the query_id if present
        :param n_iter: The number of iterations to run for each feature permutation. Higher is more stable.
        :param verbose: Flag for verbosity of print statements
        :return: The per-query feature-importance explanations for the given items.
        """
        if n_items is None or n_items <= 0:
            n_items = X.shape[0]
        if verbose:
            print("Num samples: ", n_items)

        average_utility = {}
        pairs = self._compute_pairs(n_items)
        weights = np.array([(1 / p[0] + 1 / p[1]) for p in pairs])
        sample = X.to_pd(copy=False).iloc[:n_items]
        ideal_scores = self.predict_fn(
            Tabular(sample, categorical_columns=X.categorical_columns)
        )
        assert isinstance(ideal_scores, np.ndarray), \
            "The output of the prediction function should be a numpy array."
        pi = self._compute_rank(ideal_scores).tolist()
        rank2index = {p: i for i, p in enumerate(pi)}
        if verbose:
            print(f"Ranks of items from given model: {pi}")

        for idx, feature in tqdm(enumerate(self.features)):
            utility = []
            for trials in range(n_iter):
                x = sample.copy()
                x = self._permute(x, idx)
                scores = self.predict_fn(
                    Tabular(x, categorical_columns=X.categorical_columns)
                ).flatten()
                ranks = self._compute_rank(scores)
                propensity = [
                    self._calculate_propensity(
                        scores=scores,
                        ranks=ranks,
                        i=rank2index[p[0]],
                        j=rank2index[p[1]],
                        ideal_scores=ideal_scores,
                        quotient=quotient
                    ) for p in pairs
                ]
                if weighted:
                    utility.append(np.sum(weights * np.array(propensity)))
                else:
                    utility.append(sum(propensity))
            average_utility[feature] = np.mean(utility)

        explanations = GlobalFeatureImportance()
        explanations.add(
            feature_names=average_utility.keys(),
            importance_scores=average_utility.values(),
            sort=True
        )
        return explanations
