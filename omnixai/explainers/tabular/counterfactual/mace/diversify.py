#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from typing import Callable

from .....data.tabular import Tabular


class DiversityModule:
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
            oracle_function: Callable,
            desired_label: int
    ) -> (Tabular, np.ndarray):
        """
        Randomly generates more counterfactual examples.

        :param predict_function: The predict function.
        :param x: The query instance.
        :param cfs: The counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param desired_label: The desired label.
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
        indices = [i for i, score in enumerate(scores) if oracle_function(score) > 0]
        if desired_label >= 0:
            return extended_cfs.iloc(indices), scores[indices, desired_label]
        else:
            return extended_cfs.iloc(indices), scores[indices]

    def _loss(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            score: float,
            predict_score_weight: float
    ) -> (float, np.ndarray):
        """
        Computes the loss/metric of a counterfactual example.

        :param x: The query instance.
        :param y: The counterfactual example.
        :param score: The prediction score of the counterfactual example w.r.t the desired label.
        :param predict_score_weight: The weight of the prediction score in the counterfactual loss.
        :return: The loss and the differences between `x` and `y`.
        """
        f = (x.values != y.values).astype(int)[0]
        loss, count = np.sum(f), 0
        for col in self.cont_feature_medians.keys():
            a, b = float(x[col].values[0]), float(y[col].values[0])
            if a != b:
                loss += abs(a - b) / (abs(self.cont_feature_medians[col]) + 1e-8)
                count += 1
        s = (loss - count) / len(self.columns) - score * predict_score_weight
        return s, f

    def get_diverse_cfs(
            self,
            predict_function: Callable,
            instance: Tabular,
            counterfactual_examples: Tabular,
            oracle_function: Callable,
            desired_label: int,
            k: int = 5,
            predict_score_weight: float = 0.0,
    ) -> Tabular:
        """
        Generates a set of diverse counterfactual examples.

        :param predict_function: The predict function.
        :param instance: The query instance.
        :param counterfactual_examples: The candidate counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param desired_label: The desired label for classification tasks only.
        :param k: The max number of the generated diverse counterfactual examples.
        :param predict_score_weight: The weight of the prediction score in the counterfactual loss.
        :return: A Tabular including the diverse counterfactual examples.
        """
        original_cfs, scores = self._extend_cfs(
            predict_function, instance, counterfactual_examples, oracle_function, desired_label)
        x = instance.to_pd(copy=False).astype(self.convert_dict)
        cfs = original_cfs.to_pd(copy=False).astype(self.convert_dict)

        info = []
        x, ys = x[self.columns], cfs[self.columns]
        for i in range(cfs.shape[0]):
            s, f = self._loss(x, ys.iloc[[i]], scores[i], predict_score_weight)
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
