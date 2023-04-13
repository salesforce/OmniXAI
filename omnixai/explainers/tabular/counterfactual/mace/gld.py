#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Callable, Union

from .....data.tabular import Tabular


class GLDOptimizer:
    """
    Gradientless Descent: High-Dimensional Zeroth-Order Optimization.
    """

    def __init__(
        self,
        x: Tabular,
        predict_function: Callable,
        candidate_features: Dict,
        oracle_function: Callable,
        desired_label: int,
        num_features: int,
        cont_feature_medians: Dict,
    ):
        """
        :param x: The query instance.
        :param predict_function: The prediction function.
        :param candidate_features: The candidate features for generating counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param desired_label: The desired label for classification tasks only.
        :param num_features: The number of the features including both categorical and continuous-valued ones.
        :param cont_feature_medians: The median values of the continuous-valued features.
        """
        assert x.target_column is None, "Input ``x`` cannot have a target column."

        self.x = x
        self.x_dict = self.x.to_pd(copy=False).to_dict("records")[0]
        self.predict_function = predict_function
        self.oracle_function = oracle_function
        self.desired_label = desired_label
        self.num_features = num_features

        self.candidates = {k: sorted(v) for k, v in candidate_features.items()}
        self.columns = sorted(candidate_features.keys())
        self.cont_feature_medians = cont_feature_medians

    def _build_solutions(self, solutions: List, predict_score_weight: float) -> (Union[Tuple, None], List):
        """
        Constructs possible counterfactual examples based on the optimized values.

        :param solutions: A list of tuples containing the values for feature column/value selection.
        :param predict_score_weight: The weight of the prediction score in the loss.
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
            if self.oracle_function(score) > 0:
                sols.append(ys[i])
                objective = ss[i]
                if self.desired_label >= 0 and predict_score_weight > 0:
                    objective -= score[self.desired_label] * predict_score_weight
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
        loss /= float(self.num_features)
        return r, loss

    def optimize(
        self, min_radius: float, max_radius: float, num_epochs: int, num_starts: int, loss_weight: float
    ) -> (Tabular, Tabular):
        """
        Finds counterfactual examples based on gradient-less descent.

        :param min_radius: The minimum search radius.
        :param max_radius: The maximum search radius.
        :param num_epochs: The number of epochs for each start.
        :param num_starts: The number of starts for encouraging sparsity.
        :param loss_weight: The weight of the prediction score in the loss.
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
                best_solution, sols = self._build_solutions(solutions, loss_weight)
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


class GLD:
    """
    The counterfactual explainer based on the gradient-less descent method.
    """

    def __init__(
        self,
        training_data: Tabular,
        gld_min_radius: float = 0.0005,
        gld_max_radius: float = 0.25,
        gld_num_epochs: int = 20,
        gld_num_starts: int = 3,
        gld_loss_weight: float = 0.0,
        **kwargs
    ):
        """
        :param training_data: The training data.
        :param predict_function: The prediction function.
        :param gld_min_radius: The minimum search radius.
        :param gld_max_radius: The maximum search radius.
        :param gld_num_epochs: The number of epochs for each start.
        :param gld_num_starts: The number of starts for encouraging sparsity.
        :param gld_loss_weight: The weight of the prediction score in the loss.
        """
        assert 0 < gld_max_radius <= 0.5, "`gld_max_radius` should be in (0, 0.5]."
        assert 0 < gld_min_radius < gld_max_radius, "`gld_min_radius` should be in (0, gld_max_radius)"

        self.min_radius = gld_min_radius
        self.max_radius = gld_max_radius
        self.num_epochs = gld_num_epochs
        self.num_starts = gld_num_starts
        self.loss_weight = gld_loss_weight

        self.num_features = len(training_data.feature_columns)
        self.cont_feature_medians = training_data.get_continuous_medians()

    def get_cf_examples(
            self,
            predict_function: Callable,
            x: Tabular,
            desired_label: int,
            candidate_features: Dict
    ) -> Dict:
        """
        Generates counterfactual examples given the query instance and the desired label.

        :param predict_function: The prediction function.
        :param x: The query instance.
        :param desired_label: The desired label.
        :param candidate_features: The candidate counterfactual features generated by the retrieval module.
        :return: The generated counterfactual examples.
        """
        assert isinstance(x, Tabular), "Input ``x`` should be an instance of Tabular."
        assert x.shape[0] == 1, "Input ``x`` can only contain one instance."
        assert x.target_column is None, "Input ``x`` cannot have a target column."

        optimizer = GLDOptimizer(
            x=x,
            predict_function=predict_function,
            candidate_features=candidate_features,
            oracle_function=lambda score: int(desired_label == np.argmax(score)),
            desired_label=desired_label,
            num_features=self.num_features,
            cont_feature_medians=self.cont_feature_medians,
        )
        y, ys = optimizer.optimize(
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            num_epochs=self.num_epochs,
            num_starts=self.num_starts,
            loss_weight=self.loss_weight,
        )
        scores = predict_function(y)[0, :]
        if desired_label == np.argmax(scores):
            return {"best_cf": y.remove_target_column(), "cfs": ys.remove_target_column()}
        else:
            return {}
