#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from typing import Dict, Callable

from omnixai.data.tabular import Tabular
from omnixai.utils.misc import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


    class _Policy(nn.Module):

        def __init__(
                self,
                x,
                candidate_features,
                regularization_weight=2.0,
                entropy_weight=2.0
        ):
            super(_Policy, self).__init__()
            self.sample = x
            self.candidate_features = {k: sorted(v) for k, v in candidate_features.items()}
            self.selected_columns = sorted(candidate_features.keys())
            self.regularization_weight = regularization_weight
            self.entropy_weight = entropy_weight
            self.column_prob_tensor = None
            self.action_prob_tensor = None
            self._build_layers()

        def _build_layers(self):
            self.column_prob_tensor = nn.Parameter(
                torch.zeros(len(self.selected_columns)), requires_grad=True)
            self.action_prob_tensor = []
            for column in self.selected_columns:
                self.action_prob_tensor.append(nn.Parameter(
                    torch.zeros(len(self.candidate_features[column])), requires_grad=True)
                )
            self.action_prob_tensor = torch.nn.ParameterList(self.action_prob_tensor)

        def _get_column_prob(self):
            return torch.clamp(torch.sigmoid(self.column_prob_tensor), min=0.1, max=0.8)

        @staticmethod
        def _clip_softmax(p, dim, min_val=0.02, max_val=0.85):
            min_val = min(1.0 / p.shape[0], min_val)
            x = torch.clamp(torch.softmax(p, dim=dim), min=min_val, max=max_val)
            return x / torch.sum(x)

        def forward(self, inputs):
            columns, actions, rewards = inputs
            # shape of pa and pb: (batch_size, num_columns)
            column_prob = self._get_column_prob()
            # prob of selecting feature columns
            column_prob_reshape = column_prob.view((1, column_prob.shape[0]))
            pa = torch.log(column_prob_reshape) * columns + \
                 torch.log(1.0 - column_prob_reshape) * (1.0 - columns)
            # prob of selecting feature values
            pb = []
            for i, p in enumerate(self.action_prob_tensor):
                # shape of prob and action: (num_values,), (batch_size,)
                prob, action = self._clip_softmax(p, dim=0), actions[:, i]
                pb.append(prob[action].view((action.shape[0], 1)))
            pb = torch.log(torch.cat(pb, dim=1))

            loss = torch.mean(torch.sum(pa + pb, dim=1) * rewards)
            regularization = torch.mean(column_prob) * self.regularization_weight
            entropy = (torch.mean(column_prob * torch.log(column_prob))) * self.entropy_weight
            return -loss + regularization + entropy

        # column_probs: (len(self.selected_columns), )
        # feature_probs: a list of probability vectors, e.g., [(len(self.candidate_features[key]), ), ...]
        def get_actions(self, column_probs, action_probs, num_actions):
            batch_samples = []
            df_dict = self.sample.to_dict('records')[0]
            # Sampling
            columns = np.random.binomial(1, column_probs, size=(num_actions, len(column_probs)))
            actions = np.array([np.random.choice(len(prob), p=prob, size=(num_actions,))
                                for prob in action_probs]).T
            for i in range(num_actions):
                y = df_dict.copy()
                y.update({self.selected_columns[k]: self.candidate_features[self.selected_columns[k]][act]
                          for k, (col, act) in enumerate(zip(columns[i], actions[i])) if col > 0})
                batch_samples.append([y[c] for c in self.sample.columns])
            return pd.DataFrame(data=batch_samples, columns=self.sample.columns), columns, actions

        def get_probabilities(self):
            column_probs = self._get_column_prob().detach().cpu().numpy()
            action_probs = [self._clip_softmax(p, dim=0).detach().cpu().numpy()
                            for p in self.action_prob_tensor]
            return column_probs, action_probs


class RLOptimizer:
    """
    The RL-based method for generating counterfactual examples.
    """
    def __init__(
        self,
        x: Tabular,
        predict_function: Callable,
        candidate_features: Dict,
        oracle_function: Callable,
        desired_label: int
    ):
        """
        :param x: The query instance.
        :param predict_function: The prediction function.
        :param candidate_features: The candidate features for generating counterfactual examples.
        :param oracle_function: The function for determining whether a solution is acceptable or not.
        :param desired_label: The desired label for classification tasks only.
        """
        assert x.target_column is None, "Input ``x`` cannot have a target column."
        if not is_torch_available():
            raise RuntimeError("PyTorch is not installed. Please install it, e.g., pip install torch.")

        self.x = x
        self.x_dict = self.x.to_pd(copy=False).to_dict("records")[0]
        self.predict_function = predict_function
        self.oracle_function = oracle_function
        self.candidate_features = candidate_features
        self.desired_label = desired_label

    def _greedy(self, x, optimal_policy):
        z, visited = x.copy(), {}
        column2index = {c: i for i, c in enumerate(x.columns)}

        result = None
        for _ in range(len(optimal_policy)):
            best, all_scores, update = -np.inf, None, None
            for column, features in optimal_policy:
                if column in visited:
                    continue
                for f, _ in features:
                    y = z.copy()
                    y.iloc[0, column2index[column]] = f
                    scores = self.predict_function(
                        Tabular(data=y, categorical_columns=self.x.categorical_columns)
                    )[0]
                    if self.desired_label >= 0:
                        score = scores[self.desired_label]
                    else:
                        score = self.oracle_function(scores)
                    if score > best:
                        best, all_scores = score, scores
                        update = (column, f)

            visited[update[0]] = True
            z.iloc[0, column2index[update[0]]] = update[1]
            if all_scores is not None:
                if self.oracle_function(all_scores) > 0:
                    result = z
                    break
        return result

    def optimize(
            self,
            batch_size=40,
            learning_rate=0.1,
            num_iterations=15,
            regularization_weight=2.0,
            entropy_weight=2.0,
            base_score_percentile=50,
            feature_column_top_k=10,
            feature_values_top_k=2
    ):
        x = self.x.remove_target_column().to_pd()
        policy = _Policy(
            x=x,
            candidate_features=self.candidate_features,
            regularization_weight=regularization_weight,
            entropy_weight=entropy_weight
        )
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate
        )

        # RL optimization
        for iteration in range(num_iterations):
            # Sample actions
            column_prob, action_prob = policy.get_probabilities()
            samples, batch_columns, batch_actions = \
                policy.get_actions(column_prob, action_prob, num_actions=batch_size)
            # Compute rewards
            scores = self.predict_function(
                Tabular(data=samples, categorical_columns=self.x.categorical_columns)
            )
            if self.desired_label >= 0:
                scores = scores[:, self.desired_label]
            if base_score_percentile > 0:
                scores = scores - np.percentile(scores, base_score_percentile)
            # Update policy
            optimizer.zero_grad()
            loss = policy(
                (torch.LongTensor(batch_columns), torch.LongTensor(batch_actions), torch.FloatTensor(scores)))
            loss.backward()
            optimizer.step()

        # Pick the optimal policy
        column_probs, action_probs = policy.get_probabilities()
        d, optimal = zip(policy.selected_columns, column_probs, action_probs), []
        for column, _, action_prob in sorted(d, key=lambda c: c[1], reverse=True):
            vs = sorted(zip(policy.candidate_features[column], action_prob), key=lambda c: c[1], reverse=True)
            optimal.append((column, vs[:feature_values_top_k]))
        cf = self._greedy(x, optimal)

        # Sample potential CF examples
        threshold = sorted(column_probs, reverse=True)[:feature_column_top_k][-1]
        column_probs = [c if c >= threshold else 0.0 for c in column_probs]
        sampled_cfs, _, _ = policy.get_actions(
            column_probs, action_probs,
            num_actions=batch_size * 2
        )
        scores = self.predict_function(
            Tabular(data=sampled_cfs, categorical_columns=self.x.categorical_columns)
        )
        indices = [i for i, score in enumerate(scores) if self.oracle_function(score) > 0]
        if len(indices) > 0:
            cfs = pd.concat([cf, sampled_cfs.iloc[indices]]) if cf is not None \
                else sampled_cfs.iloc[indices]
        else:
            cfs = cf

        if cf is not None:
            cf = Tabular(cf, categorical_columns=self.x.categorical_columns)
        if cfs is not None:
            cfs = Tabular(cfs, categorical_columns=self.x.categorical_columns)
        return cf, cfs


class RL:

    def __init__(
            self,
            batch_size: int = 40,
            learning_rate: float = 0.1,
            num_iterations: int = 15,
            regularization_weight: float = 2.0,
            entropy_weight: float = 2.0,
            base_score_percentile: int = 50,
            feature_column_top_k: int = 10,
            feature_values_top_k: int = 2,
            **kwargs
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_weight = regularization_weight
        self.entropy_weight = entropy_weight
        self.base_score_percentile = base_score_percentile
        self.feature_column_top_k = feature_column_top_k
        self.feature_values_top_k = feature_values_top_k

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

        optimizer = RLOptimizer(
            x=x,
            predict_function=predict_function,
            candidate_features=candidate_features,
            oracle_function=lambda score: int(desired_label == np.argmax(score)),
            desired_label=desired_label
        )
        y, ys = optimizer.optimize(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_iterations=self.num_iterations,
            regularization_weight=self.regularization_weight,
            entropy_weight=self.entropy_weight,
            base_score_percentile=self.base_score_percentile,
            feature_column_top_k=self.feature_column_top_k,
            feature_values_top_k=self.feature_values_top_k
        )
        if ys is not None:
            return {"cfs": ys.remove_target_column()}
        else:
            return {}
