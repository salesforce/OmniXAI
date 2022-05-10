#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The partial dependence plots for tabular data.
"""
import numpy as np

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

    def _compute_pdp(self, column_index, inputs=None):
        """
        Computes partial dependence plots.

        :param column_index: A feature column index.
        :param inputs: `None` for global explanations or the input instances for local explanations.
        :return: The candidate features, PDP means and PDP stds.
        :rtype: tuple(List, np.ndarray, np.ndarray)
        """
        if inputs is None:
            # For global explanations
            x = self.data.copy()
        else:
            # For local explanations
            x = self.transformer.transform(inputs.remove_target_column()).copy()

        baselines = []
        candidates = self.candidates[column_index]
        for i, y in enumerate(candidates):
            x[:, column_index] = y
            baselines.append(self.predict_fn(x))
        baselines = np.swapaxes(np.array(baselines), 0, 1)

        mean = np.mean(baselines, axis=0)
        std = np.std(baselines, axis=0)
        return candidates, mean, std

    def _global_explain(self) -> PDPExplanation:
        """
        Generates global explanations.

        :return: The global explanations according to the ML model and the training data.
        :rtype: PDPExplanation
        """
        explanations = PDPExplanation(self.mode)
        categorical_features = set(self.categorical_features)
        for i, column_name in enumerate(self.feature_columns):
            values, mean, std = self._compute_pdp(i)
            if i in categorical_features:
                values = [self.categorical_names[i][int(v)] for v in values]
            explanations.add(index="global", feature_name=column_name, values=values, pdp_mean=mean, pdp_std=std)
        return explanations

    def _local_explain(self, X: Tabular) -> PDPExplanation:
        """
        Generates local explanations.

        :param X: The input instances.
        :return: The local explanations according to the ML model and the input instances.
        :rtype: PDPExplanation
        """
        explanations = PDPExplanation(self.mode)
        categorical_features = set(self.categorical_features)
        for k in range(X.shape[0]):
            for i, column_name in enumerate(self.feature_columns):
                values, mean, std = self._compute_pdp(i, inputs=X.iloc(k))
                if i in categorical_features:
                    values = [self.categorical_names[i][int(v)] for v in values]
                explanations.add(index=k, feature_name=column_name, values=values, pdp_mean=mean, pdp_std=std)
        return explanations

    def explain(self, **kwargs) -> PDPExplanation:
        """
        Generates global PDP explanations.

        :return: The generated PDP explanations.
        """
        return self._global_explain()
