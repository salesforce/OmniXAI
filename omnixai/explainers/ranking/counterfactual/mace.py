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
import pandas as pd
from typing import List, Callable, Dict, Union
from collections import Counter, deque

from ...base import ExplainerBase
from ....data.tabular import Tabular
from ....preprocessing.tabular import TabularTransform
from ....preprocessing.encode import Ordinal, KBins
from ....explanations.tabular.counterfactual import CFExplanation


class _GLDOptimizer:
    """
    Gradientless Descent: High-Dimensional Zeroth-Order Optimization.
    """

    def __init__(
        self,
        x: Tabular,
        predict_function: Callable,
        candidate_features: Dict,
        oracle_function: Callable,
        cont_feature_medians: Dict,
    ):
        """
        :param x: The query instance.
        :param predict_function: The prediction function.
        :param candidate_features: The candidate features for generating counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param cont_feature_medians: The median values of the continuous-valued features.
        """
        assert x.target_column is None, "Input ``x`` cannot have a target column."

        self.x = x
        self.x_dict = self.x.to_pd(copy=False).to_dict("records")[0]
        self.predict_function = predict_function
        self.oracle_function = oracle_function

        self.candidates = {k: sorted(v) for k, v in candidate_features.items()}
        self.columns = sorted(candidate_features.keys())
        self.cont_feature_medians = cont_feature_medians

    def _build_solutions(self, solutions: List):
        """
        Constructs possible counterfactual examples based on the optimized values.

        :param solutions: A list of tuples containing the values for feature column/value selection.
        :return: The best solution and all valid counterfactual examples.
        """
        ys, ss = [], []
        for col_vs, val_vs in solutions:
            y, s = self._build_example(col_vs, val_vs, return_df=False)
            ys.append(y)
            ss.append(s)
        scores = self.predict_function(
            Tabular(data=pd.DataFrame(ys, columns=self.x.columns), categorical_columns=self.x.categorical_columns)
        )

        best_objective, best_solution, sols = 1e8, None, []
        for i in range(len(solutions)):
            score, objective = scores[i], 1e8
            if self.oracle_function(score) >= 0:
                sols.append(ys[i])
                objective = ss[i]
                if objective < best_objective:
                    best_objective, best_solution = objective, solutions[i]

        sols = pd.DataFrame(sols, columns=self.x.columns) if len(sols) > 0 else None
        return best_solution, sols

    def _build_example(self, col_vs: np.ndarray, val_vs: List, return_df: bool = True):
        """
        Constructs an example based on the values `col_vs` and `val_vs`.

        :param col_vs: The values for feature column selection (a numpy.ndarray).
        :param val_vs: The values for feature value selection (a list of numpy.ndarray).
        :param return_df: True if it returns a pd.DataFrame instance.
        :return:
        """
        y = self.x_dict.copy()
        cols = np.argwhere(col_vs > 0.5).flatten()
        y.update({self.columns[c]: self.candidates[self.columns[c]][np.argmax(val_vs[c])] for c in cols})
        if return_df:
            r = pd.DataFrame(data=[[y[c] for c in self.x.columns]], columns=self.x.columns)
        else:
            r = [y[c] for c in self.x.columns]

        loss = 0
        for c in cols:
            name = self.columns[c]
            if name in self.cont_feature_medians:
                loss += abs(float(y[name]) - float(self.x_dict[name])) / (abs(self.cont_feature_medians[name]) + 1e-6)
            else:
                loss += 1
        return r, loss

    def optimize(
        self, min_radius: float, max_radius: float, num_epochs: int, num_starts: int,
    ) -> (Tabular, Tabular):
        """
        Finds counterfactual examples based on gradient-less descent.

        :param min_radius: The minimum search radius.
        :param max_radius: The maximum search radius.
        :param num_epochs: The number of epochs for each start.
        :param num_starts: The number of starts for encouraging sparsity.
        :return: The best counterfactual examples (in terms of the loss) and
            all the valid counterfactual examples found in the optimization procedure.
        """
        steps = min(max(int(np.log2(max_radius / (min_radius + 1e-5))), 5), 20)

        all_solutions, final_solution = deque([]), None
        column_scores = np.ones((len(self.columns),), dtype=float) * 0.5
        values_scores = [np.ones((len(self.candidates[col]),), dtype=float) * 0.5 for col in self.columns]

        for iteration in range(num_starts):
            if iteration > 0:
                column_scores = (column_scores >= 0.5) * 0.55

            found = False
            for epoch in range(num_epochs):
                # Ball Sampling
                solutions = []
                for k in range(steps):
                    r = 2 ** (-k) * max_radius
                    a = np.clip(column_scores + np.random.randn(len(column_scores)) * r, a_min=0, a_max=1)
                    b = [np.clip(v + np.random.randn(len(v)) * r, a_min=0, a_max=1) for v in values_scores]
                    solutions.append((a, b))

                # Get the current best solution
                best_solution, sols = self._build_solutions(solutions)
                if sols is not None:
                    all_solutions.append(sols)
                if best_solution is not None:
                    column_scores, values_scores = best_solution
                    final_solution, found = best_solution, True
            if not found:
                break

        if final_solution:
            column_scores, values_scores = final_solution
        y, _ = self._build_example(column_scores, values_scores)
        ys = pd.concat(all_solutions) if len(all_solutions) > 0 else y

        return (
            Tabular(data=y, categorical_columns=self.x.categorical_columns, target_column=self.x.target_column),
            Tabular(data=ys, categorical_columns=self.x.categorical_columns, target_column=self.x.target_column),
        )


class _DiversityModule:
    """
    The module for generating diverse counterfactual examples.
    """

    def __init__(self, training_data: Tabular, num_random_tries: int = 0):
        """
        :param training_data: The training data.
        :param num_random_tries: The number of random tries for expanding counterfactual examples.
        """
        assert isinstance(training_data, Tabular), "`training_data` should be an instance of Tabular."

        self.columns = training_data.feature_columns
        self.cate_features = training_data.categorical_columns
        self.cont_features = training_data.continuous_columns
        self.cont_feature_medians = training_data.get_continuous_medians()
        self.num_random_tries = num_random_tries

        # Make sure the type of categorical features is str and
        # the types of continuous-valued features is float.
        self.convert_dict = {c: str for c in self.cate_features}
        self.convert_dict.update({c: float for c in self.cont_features})

    def _extend_cfs(
            self,
            predict_function: Callable,
            x: Tabular,
            cfs: Tabular,
            oracle_function: Callable
    ) -> (Tabular, np.ndarray):
        """
        Randomly generates more counterfactual examples.

        :param predict_function: The predict function.
        :param x: The query instance.
        :param cfs: The counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :return: The counterfactual examples (Tabular) and
            the corresponding prediction scores w.r.t the desired label (numpy.ndarray).
        """
        cfs = cfs.remove_target_column()
        if self.num_random_tries == 0:
            extended_cfs = cfs
        else:
            extended_cfs = []
            for i in range(cfs.shape[0]):
                y = cfs.to_pd(copy=False).iloc[[i]]
                extended_cfs.append(y)

                # Randomly pick features to be modified
                for _ in range(self.num_random_tries):
                    for prob in [0.6, 0.7, 0.8, 0.9]:
                        z = x.to_pd(copy=True)
                        probs = np.random.random(size=len(self.columns))
                        for p, col in zip(probs, self.columns):
                            if p <= prob:
                                z.iloc[0, z.columns.get_loc(col)] = y.iloc[0, y.columns.get_loc(col)]
                        extended_cfs.append(z)

            extended_cfs = Tabular(
                data=pd.concat(extended_cfs, sort=False), categorical_columns=cfs.categorical_columns
            )
        scores = predict_function(extended_cfs)
        indices = [i for i, score in enumerate(scores) if oracle_function(score) >= 0]
        return extended_cfs.iloc(indices), scores[indices]

    def _loss(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
    ) -> (float, np.ndarray):
        """
        Computes the loss/metric of a counterfactual example.

        :param x: The query instance.
        :param y: The counterfactual example.
        :return: The loss and the differences between `x` and `y`.
        """
        f = (x.values != y.values).astype(int)[0]
        loss, count = np.sum(f), 0
        for col in self.cont_feature_medians.keys():
            a, b = float(x[col].values[0]), float(y[col].values[0])
            if a != b:
                loss += abs(a - b) / (abs(self.cont_feature_medians[col]) + 1e-8)
                count += 1
        s = (loss - count) / len(self.columns)
        return s, f

    def get_diverse_cfs(
            self,
            predict_function: Callable,
            instance: Tabular,
            counterfactual_examples: Tabular,
            oracle_function: Callable,
            k: int = 5,
    ) -> Tabular:
        """
        Generates a set of diverse counterfactual examples.

        :param predict_function: The predict function.
        :param instance: The query instance.
        :param counterfactual_examples: The candidate counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param k: The max number of the generated diverse counterfactual examples.
        :return: A Tabular including the diverse counterfactual examples.
        """
        original_cfs, scores = self._extend_cfs(
            predict_function, instance, counterfactual_examples, oracle_function)
        x = instance.to_pd(copy=False).astype(self.convert_dict)
        cfs = original_cfs.to_pd(copy=False).astype(self.convert_dict)

        info = []
        x, ys = x[self.columns], cfs[self.columns]
        for i in range(cfs.shape[0]):
            s, f = self._loss(x, ys.iloc[[i]])
            info.append((s, hash("".join(map(str, f))), i, [j for j, v in enumerate(f) if v == 1]))
        info = sorted(info, key=lambda p: p[0])

        results, visited, counts = [], {}, {}
        for _, flag, index, features in info:
            if flag not in visited:
                visited[flag], exceed_limit = True, False
                for f in features:
                    if counts.get(f, 0) >= 3:
                        exceed_limit = True
                        break
                if not exceed_limit:
                    for f in features:
                        counts[f] = counts.get(f, 0) + 1
                    results.append(original_cfs.to_pd(copy=False).iloc[[index]])
                    if len(results) >= k:
                        break
        return (
            Tabular(data=pd.concat(results), categorical_columns=original_cfs.categorical_columns)
            if len(results) > 0
            else None
        )


class _BinarySearchRefinement:
    """
    The module for refining the continuous-valued features
    in the generated counterfactual examples via binary search.
    """

    def __init__(self, training_data: Tabular):
        """
        :param training_data: The training data.
        """
        self.cont_columns = training_data.continuous_columns

    @staticmethod
    def _refine(
            instance: Tabular,
            predict_function: Callable,
            cont_features: Dict,
            oracle_function: Callable
    ) -> pd.DataFrame:
        """
        Refines the continuous-valued features for the given instance.

        :param instance: The instance to be refined.
        :param predict_function: The predict function.
        :param cont_features: The continuous-valued features to be refined.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :return: The refined instance.
        """
        x = instance.to_pd(copy=False)
        y = x.copy()
        column2loc = {c: x.columns.get_loc(c) for c in cont_features}

        for col, (a, b) in cont_features.items():
            gap, r = b - a, None
            while (b - a) / (gap + 1e-3) > 0.1:
                z = (a + b) * 0.5
                y.iloc[0, column2loc[col]] = z
                scores = predict_function(Tabular(data=y, categorical_columns=instance.categorical_columns))
                if oracle_function(scores[0]) >= 0:
                    b, r = z, z
                else:
                    a = z
            y.iloc[0, column2loc[col]] = r if r is not None else x.iloc[0, column2loc[col]]
        return y

    def refine(
            self,
            predict_function: Callable,
            instance: Tabular,
            cfs: Tabular,
            oracle_function: Callable
    ):
        """
        Refines the continuous-valued features in the counterfactual examples.

        :param predict_function: The predict function.
        :param instance: The query instance.
        :param cfs: The counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :return: The refined counterfactual examples.
        """
        assert instance.target_column is None, "Input `instance` cannot have a target column."
        if cfs is None:
            return None

        results = []
        x = instance.to_pd(copy=False)
        for i in range(cfs.shape[0]):
            y = cfs.iloc([i]).to_pd(copy=False)
            cont_features = {}
            for col in self.cont_columns:
                a, b = float(x[col].values[0]), float(y[col].values[0])
                if a != b:
                    cont_features[col] = (a, b) if a <= b else (b, a)
            if len(cont_features) == 0:
                results.append(y)
            else:
                results.append(
                    self._refine(cfs.iloc([i]), predict_function, cont_features, oracle_function)
                )
        return Tabular(data=pd.concat(results), categorical_columns=cfs.categorical_columns)


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
        self.diversity = _DiversityModule(training_data) \
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
        optimizer = _GLDOptimizer(
            x=x,
            predict_function=self.predict_function,
            candidate_features=candidate_features,
            oracle_function=oracle_function,
            cont_feature_medians=cont_feature_medians,
        )
        y, ys = optimizer.optimize(
            min_radius=min_radius,
            max_radius=max_radius,
            num_epochs=num_epochs,
            num_starts=num_starts
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

        :param X: The list of the recommended items.
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
                diversity = _DiversityModule(X) if self.diversity is None else self.diversity
                cfs = diversity.get_diverse_cfs(
                    self.predict_function, x, examples["cfs"], oracle_function, k=max_number_examples)
                cfs = _BinarySearchRefinement(x).refine(self.predict_function, x, cfs, oracle_function)
                cfs_df = cfs.to_pd().reset_index()
                cfs_df["@ranking_score"] = self.predict_function(cfs)

            instance_df = x.to_pd()
            instance_df["@ranking_score"] = score_b
            explanations.add(query=instance_df, cfs=cfs_df)
        return explanations
