#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The partial dependence plots for tabular data.
"""
import warnings
import numpy as np
from typing import List

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.pdp import PDPExplanation


class PartialDependenceTabular(TabularExplainer):
    """
    The partial dependence plots for tabular data. For more information, please refer to
    https://scikit-learn.org/stable/modules/partial_dependence.html.
    """

    explanation_type = "global"
    alias = ["pdp", "partial_dependence"]

    def __init__(self, training_data: Tabular, predict_function, mode="classification", **kwargs):
        """
        :param training_data: The data used to initialize a PDP explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param kwargs: Additional parameters, e.g., ``grid_resolution`` -- the number of
            candidates for each feature during generating partial dependence plots.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        grid_resolution = kwargs.get("grid_resolution", 10)
        self.candidates = {}
        for column_index, column_name in enumerate(self.feature_columns):
            num_unique_values = len(np.unique(self.data[:, column_index]))
            if column_index in self.categorical_features or num_unique_values <= grid_resolution:
                # Categorical features
                candidates = sorted(np.unique(self.data[:, column_index]))
            else:
                # Continuous-valued features
                percentiles = np.linspace(1, 99, num=grid_resolution)
                candidates = sorted(set(np.percentile(self.data[:, column_index], percentiles)))
            self.candidates[column_index] = candidates

    def _compute_pdp(self, data, column_index):
        """
        Computes partial dependence plots.

        :param data: The dataset used to compute PDP scores.
        :param column_index: A feature column index.
        :return: The candidate features, PDP means and PDP stds.
        :rtype: tuple(List, np.ndarray, np.ndarray)
        """
        x = data.copy()
        baselines = []
        candidates = self.candidates[column_index]
        for i, y in enumerate(candidates):
            x[:, column_index] = y
            baselines.append(self.predict_fn(x))
        baselines = np.swapaxes(np.array(baselines), 0, 1)
        mean = np.mean(baselines, axis=0)
        return mean

    def _global_explain(
            self,
            features,
            monte_carlo,
            monte_carlo_steps,
            monte_carlo_frac
    ) -> PDPExplanation:
        """
        Generates global explanations.

        :return: The global explanations according to the ML model and the training data.
        :rtype: PDPExplanation
        """
        if features is None:
            feature_columns = self.feature_columns
        else:
            if isinstance(features, str):
                features = [features]
            for f in features:
                assert f in self.feature_columns, \
                    f"The dataset doesn't have feature `{f}`."
            feature_columns = features
        column_index = {f: i for i, f in enumerate(self.feature_columns)}
        if len(feature_columns) > 20:
            warnings.warn(f"Too many features ({len(feature_columns)} > 20) for PDP to process, "
                          f"it will take a while to finish. It is better to choose a subset"
                          f"of features to analyze by setting the parameter `features`.")

        explanations = PDPExplanation(self.mode)
        categorical_features = set(self.categorical_features)
        for column_name in feature_columns:
            i = column_index[column_name]
            values = self.candidates[i]
            if i in categorical_features:
                values = [self.categorical_names[i][int(v)] for v in values]

            scores = None
            sampled_scores = []
            if monte_carlo:
                n = int(self.data.shape[0] * monte_carlo_frac)
                if n < 10:
                    warnings.warn(f"The number of samples in each Monte Carlo step is "
                                  f"too small, i.e., {n} < 10. The Monte Carlo sampling is ignored.")
                else:
                    for _ in range(monte_carlo_steps):
                        indices = np.random.choice(range(self.data.shape[0]), n, replace=False)
                        sampled_scores.append(self._compute_pdp(self.data[indices], i))
                    scores = sum(sampled_scores) / len(sampled_scores)
            if scores is None:
                scores = self._compute_pdp(self.data, i)

            explanations.add(
                feature_name=column_name,
                values=values,
                scores=scores,
                sampled_scores=sampled_scores if sampled_scores else None
            )
        return explanations

    def explain(
            self,
            features: List = None,
            monte_carlo: bool = False,
            monte_carlo_steps: int = 10,
            monte_carlo_frac: float = 0.1,
            **kwargs
    ) -> PDPExplanation:
        """
        Generates global PDP explanations.

        :param features: The names of the features to be studied.
        :param monte_carlo: Whether computing PDP for Monte Carlo samples.
        :param monte_carlo_steps: The number of Monte Carlo sampling steps.
        :param monte_carlo_frac: The number of randomly selected samples in each Monte Carlo step.
        :return: The generated PDP explanations.
        """
        return self._global_explain(
            features, monte_carlo, monte_carlo_steps, monte_carlo_frac)
