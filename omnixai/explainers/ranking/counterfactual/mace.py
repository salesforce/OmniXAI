#
# Copyright (c) 2022 salesforce.com, inc.
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
from ...tabular.counterfactual.mace.gld import GLDOptimizer
from ...tabular.counterfactual.mace.diversify import DiversityModule
from ...tabular.counterfactual.mace.refine import BinarySearchRefinement
from ....explanations.tabular.counterfactual import CFExplanation


class MACEExplainer(ExplainerBase):
    """
    The Model-Agnostic Counterfactual Explanation (MACE) developed by Yang et al. Please
    cite the paper `MACE: An Efficient Model-Agnostic Framework for Counterfactual Explanation`.
    This explainer is designed for ranking tasks only.
    """

    explanation_type = "local"
    alias = ["mace"]

    def __init__(
        self,
        training_data: Union[Tabular, None],
        predict_function: Callable,
        ignored_features: List = None,
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
        """
        super().__init__()
        assert training_data is None or isinstance(training_data, Tabular), \
            f"`training_data` should be Tabular or None instead of {type(training_data)}."
        self.kwargs = kwargs

        self.predict_function = lambda x: predict_function(x).flatten()
        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        self.candidate_features = self._candidate_features(training_data) \
            if training_data is not None else None
        self.cont_feature_medians = training_data.get_continuous_medians() \
            if training_data is not None else None
        self.diversity = DiversityModule(training_data) \
            if training_data is not None else None

    def _candidate_features(self, data, max_num_candidates=10):
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

        counts = [Counter(y[f].values).most_common(max_num_candidates) for f in y.columns]
        candidates = {f: [c[0] for c in count] for f, count in zip(y.columns, counts)}
        return candidates

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
        best_score, all_scores = -1, None
        for _ in range(len(candidate_features)):
            update = None
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
            if update is None:
                break

            visited[update[0]] = True
            y.iloc[0, column2loc[update[0]]] = update[1]
            if all_scores is not None:
                if oracle_function(all_scores) >= 0:
                    example = Tabular(data=y, categorical_columns=x.categorical_columns)
                    break

        if example is not None:
            return {"best_cf": example, "cfs": example}
        else:
            return {}

    def _generate_cf_examples(
            self,
            x: Tabular,
            candidate_features: Dict,
            cont_feature_medians: Dict,
            oracle_function: Callable,
            min_radius: float = 0.0005,
            max_radius: float = 0.25,
            num_epochs: int = 20,
            num_starts: int = 3,
    ):
        optimizer = GLDOptimizer(
            x=x,
            predict_function=self.predict_function,
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
        score = self.predict_function(y)
        if oracle_function(score) >= 0:
            return {"best_cf": y.remove_target_column(), "cfs": ys.remove_target_column()}
        else:
            return {}

    def explain(
            self,
            X: Tabular,
            item_a_index: Union[int, List],
            item_b_index: Union[int, List],
            max_number_examples: int = 5
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

        df = X.to_pd(copy=False)
        candidate_features = self._candidate_features(X) \
            if self.candidate_features is None else self.candidate_features
        cont_feature_medians = {c: np.mean(np.abs(df[c].values.astype(float)))
                                for c in X.continuous_columns} \
            if self.cont_feature_medians is None else self.cont_feature_medians

        for a_index, b_index in zip(item_a_index, item_b_index):
            x = X.iloc(b_index)
            score_a = self.predict_function(X.iloc(a_index))[0]
            score_b = self.predict_function(x)[0]
            if score_a >= score_b:
                oracle_function = lambda s: s - score_a
            else:
                oracle_function = lambda s: score_a - s

            examples = self._generate_cf_examples(
                x, candidate_features, cont_feature_medians, oracle_function, **self.kwargs)
            if not examples:
                examples = self._greedy(
                    self.predict_function, x, oracle_function, candidate_features)

            cfs_df = None
            if examples:
                diversity = DiversityModule(X) if self.diversity is None else self.diversity
                cfs = diversity.get_diverse_cfs(
                    self.predict_function, x, examples["cfs"], oracle_function,
                    desired_label=-1, k=max_number_examples
                )
                cfs = BinarySearchRefinement(x).refine(self.predict_function, x, cfs, oracle_function)
                cfs_df = cfs.to_pd().reset_index()
                cfs_df["@ranking_score"] = self.predict_function(cfs)

            instance_df = x.to_pd()
            instance_df["@ranking_score"] = score_b
            explanations.add(query=instance_df, cfs=cfs_df)
        return explanations
