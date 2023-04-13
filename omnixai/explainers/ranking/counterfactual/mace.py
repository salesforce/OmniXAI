#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Model-Agnostic Counterfactual Explanation (MACE) for ranking tasks.
"""
import numpy as np
from typing import List, Callable, Dict, Union
from collections import Counter

from ...base import ExplainerBase
from ....data.tabular import Tabular
from ....preprocessing.tabular import TabularTransform
from ....preprocessing.encode import Ordinal, KBins
from ...tabular.counterfactual.mace.rl import RLOptimizer
from ...tabular.counterfactual.mace.gld import GLDOptimizer
from ...tabular.counterfactual.mace.diversify import DiversityModule
from ...tabular.counterfactual.mace.refine import BinarySearchRefinement
from ....explanations.tabular.counterfactual import CFExplanation


class MACEExplainer(ExplainerBase):
    """
    The Model-Agnostic Counterfactual Explanation (MACE) developed by Yang et al. Please
    cite the paper `MACE: An Efficient Model-Agnostic Framework for Counterfactual Explanation`.
    This version of MACE is designed for ranking tasks only.
    """

    explanation_type = "local"
    alias = ["mace"]

    def __init__(
        self,
        training_data: Union[Tabular, None],
        predict_function: Callable,
        ignored_features: List = None,
        method: str = "gld",
        **kwargs,
    ):
        """
        :param training_data: The data used to initialize a MACE explainer. ``training_data``
            can either be the training dataset for training the machine learning model or None.
            If ``training_data`` is None, it will use the features in the recommended items (test instances)
            to construct counterfactual explanations. The type of ``training_data`` is `Tabular`
            (which combines both query and item into one Tabular instance).
        :param predict_function: The prediction function corresponding to the ranking model to explain.
            The outputs of the ``predict_function`` are the ranking scores.
        :param ignored_features: The features ignored in generating counterfactual examples.
        :param method: The method for generating counterfactual examples, e.g., "gld", "rl" or "greedy".
        """
        super().__init__()
        assert training_data is None or isinstance(training_data, Tabular), \
            f"`training_data` should be Tabular or None instead of {type(training_data)}."
        assert method in ["gld", "rl", "greedy"], \
            "`method` should be `gld`, `rl` or `greedy`."
        self.method = method
        self.kwargs = kwargs
        self.max_num_candidates = kwargs.get("max_num_candidates", 10)

        self.predict_function = lambda x: predict_function(x).flatten()
        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        self.candidate_features = self._candidate_features(training_data) \
            if training_data is not None else None
        self.cont_feature_medians = training_data.get_continuous_medians() \
            if training_data is not None else None
        self.diversity = DiversityModule(training_data) \
            if training_data is not None else None

    def _candidate_features(self, data):
        cate_features = [c for c in data.categorical_columns if c not in self.ignored_features]
        cont_features = [c for c in data.continuous_columns if c not in self.ignored_features]

        df = data.to_pd(copy=False)
        x = Tabular(
            data=df[cate_features + cont_features],
            categorical_columns=cate_features
        )
        transformer = TabularTransform(
            cate_transform=Ordinal(), cont_transform=KBins(n_bins=10)
        ).fit(x)
        y = transformer.invert(transformer.transform(x)).to_pd(copy=False).dropna(axis=1)

        counts = [Counter(y[f].values).most_common(self.max_num_candidates) for f in y.columns]
        candidates = {f: [c[0] for c in count] for f, count in zip(y.columns, counts) if len(count) > 1}
        return candidates

    def _build_predict_function(self, x: Tabular, index_a: int, index_b: int):
        """
        Because the prediction function of a ranking task can be a list-wise score function,
        we need to convert it into an element-wise function for generating counterfacutal examples.

        :param x: The list of the recommended items given a query.
        :param index_a: The index of the baseline example (item A).
        :param index_b: The index of the example to explain (item B).
        :return:
        """
        scores = self.predict_function(x)
        greater_than = bool(scores[index_a] > scores[index_b])
        df = x.to_pd()

        def _predict(_y: Tabular):
            _z = _y.to_pd(copy=False)
            _scores = []
            for i in range(_z.shape[0]):
                df.iloc[index_b] = _z.iloc[i]
                _s = self.predict_function(
                    Tabular(df, categorical_columns=x.categorical_columns)
                )
                if greater_than:
                    _scores.append(_s[index_b] - _s[index_a])
                else:
                    _scores.append(_s[index_a] - _s[index_b])
            return np.array(_scores, dtype=float)

        return _predict

    @staticmethod
    def _greedy(
            predict_function: Callable,
            instance: Tabular,
            oracle_function: Callable,
            candidate_features: Dict
    ) -> Dict:
        assert isinstance(instance, Tabular), "Input ``instance`` should be an instance of Tabular."
        assert instance.shape[0] == 1, "The input ``instance`` can only contain one instance."

        x = instance.remove_target_column()
        y = x.to_pd(copy=False)
        column2loc = {c: y.columns.get_loc(c) for c in y.columns}

        example, visited = None, {}
        for _ in range(len(candidate_features)):
            best_score, all_scores, update = -np.inf, None, None
            for feature, values in candidate_features.items():
                if feature in visited:
                    continue
                for v in values:
                    z = y.copy()
                    z.iloc[0, column2loc[feature]] = v
                    score = predict_function(Tabular(data=z, categorical_columns=x.categorical_columns))[0]
                    if oracle_function(score) > best_score:
                        best_score = oracle_function(score)
                        all_scores = score
                        update = (feature, v)

            visited[update[0]] = True
            y.iloc[0, column2loc[update[0]]] = update[1]
            if all_scores is not None:
                if oracle_function(all_scores) > 0:
                    example = Tabular(data=y, categorical_columns=x.categorical_columns)
                    break

        if example is not None:
            return {"cfs": example}
        else:
            return {}

    @staticmethod
    def _generate_cf_examples_gld(
            x: Tabular,
            predict_function: Callable,
            candidate_features: Dict,
            cont_feature_medians: Dict,
            oracle_function: Callable,
            *,
            min_radius: float = 0.0005,
            max_radius: float = 0.25,
            num_epochs: int = 20,
            num_starts: int = 3,
    ):
        optimizer = GLDOptimizer(
            x=x,
            predict_function=predict_function,
            candidate_features=candidate_features,
            oracle_function=oracle_function,
            desired_label=-1,
            num_features=len(x.feature_columns),
            cont_feature_medians=cont_feature_medians,
        )
        y, ys = optimizer.optimize(
            min_radius=min_radius,
            max_radius=max_radius,
            num_epochs=num_epochs,
            num_starts=num_starts,
            loss_weight=0
        )
        score = predict_function(y)
        if oracle_function(score) > 0:
            return {"cfs": ys.remove_target_column()}
        else:
            return {}

    @staticmethod
    def _generate_cf_examples_rl(
            x: Tabular,
            predict_function,
            candidate_features: Dict,
            oracle_function: Callable,
            *,
            batch_size=40,
            learning_rate=0.1,
            num_iterations=10,
            regularization_weight=2.0,
            entropy_weight=2.0,
            base_score_percentile=50,
            feature_column_top_k=10,
            feature_values_top_k=2
    ):
        optimizer = RLOptimizer(
            x=x,
            predict_function=predict_function,
            candidate_features=candidate_features,
            oracle_function=oracle_function,
            desired_label=-1
        )
        y, ys = optimizer.optimize(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            regularization_weight=regularization_weight,
            entropy_weight=entropy_weight,
            base_score_percentile=base_score_percentile,
            feature_column_top_k=feature_column_top_k,
            feature_values_top_k=feature_values_top_k
        )
        if ys is not None:
            return {"cfs": ys.remove_target_column()}
        else:
            return {}

    def explain(
            self,
            X: Tabular,
            item_a_index: Union[int, List],
            item_b_index: Union[int, List],
            max_number_examples: int = 5,
            **kwargs
    ) -> CFExplanation:
        """
        Generates counterfactual explanations. It considers two items A and B in the list where
        A is the baseline example and B is the example to compare. Suppose the ranking scores A > B,
        it will changes the features in B such that the new example B' > A, so that the explanations
        will be 'if B changes features x, y and z, B will have a high ranking score than A'.

        :param X: The list of the recommended items given a query. This Tabular instance contains
            the features of the query and the items.
        :param item_a_index: The index of the baseline example (item A).
        :param item_b_index: The index of the example to explain (item B).
        :param max_number_examples: The maximum number of the generated counterfactual examples.
        :return: A CFExplanation object containing the generated explanations.
        """
        explanations = CFExplanation()
        if isinstance(item_a_index, int):
            item_a_index = [item_a_index]
            if isinstance(item_b_index, (list, tuple)):
                item_a_index = item_a_index * len(item_b_index)
        if isinstance(item_b_index, int):
            item_b_index = [item_b_index]
            if isinstance(item_a_index, (list, tuple)):
                item_b_index = item_b_index * len(item_a_index)
        assert len(item_a_index) == len(item_a_index)
        X = X.remove_target_column()

        df = X.to_pd(copy=False)
        candidate_features = self._candidate_features(X) \
            if self.candidate_features is None else self.candidate_features
        cont_feature_medians = {c: np.mean(np.abs(df[c].values.astype(float)))
                                for c in X.continuous_columns} \
            if self.cont_feature_medians is None else self.cont_feature_medians

        def _rank(_x: Tabular):
            return (-self.predict_function(_x)).argsort().argsort() + 1

        for a_index, b_index in zip(item_a_index, item_b_index):
            x = X.iloc(b_index)
            predict_fn = self._build_predict_function(X, a_index, b_index)
            oracle_function = lambda s: s

            examples = {}
            if self.method == "gld":
                examples = self._generate_cf_examples_gld(
                    x, predict_fn, candidate_features, cont_feature_medians, oracle_function, **self.kwargs)
            elif self.method == "rl":
                examples = self._generate_cf_examples_rl(
                    x, predict_fn, candidate_features, oracle_function, **self.kwargs)
            if not examples:
                examples = self._greedy(
                    predict_fn, x, oracle_function, candidate_features)

            if "cfs" in examples:
                cfs_df = examples["cfs"].to_pd()
            else:
                cfs_df = None

            if examples and max_number_examples > 1:
                diversity = DiversityModule(X) if self.diversity is None else self.diversity
                cfs = diversity.get_diverse_cfs(
                    predict_fn, x, examples["cfs"], oracle_function,
                    desired_label=-1, k=max_number_examples
                )
                cfs = BinarySearchRefinement(x).refine(predict_fn, x, cfs, oracle_function)
                cfs_df = cfs.to_pd()

            if cfs_df is not None:
                df, ranks = X.to_pd(), []
                for i in range(cfs_df.shape[0]):
                    df.iloc[b_index] = cfs_df.iloc[i]
                    ranks.append(_rank(Tabular(
                        df, categorical_columns=x.categorical_columns))[b_index])
                cfs_df["@rank"] = ranks

            ranks = _rank(X)
            query = x.to_pd()
            query["@rank"] = ranks[b_index]
            context = X.iloc(a_index).to_pd()
            context["@rank"] = ranks[a_index]
            explanations.add(query=query, cfs=cfs_df, context=context)
        return explanations
