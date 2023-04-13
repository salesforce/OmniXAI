#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for global feature importance.
"""
import shap
import numpy as np
from typing import Callable, List

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import GlobalFeatureImportance


class GlobalShapTabular(TabularExplainer):
    """
    The SHAP explainer for global feature importance.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "global"
    alias = ["shap_global"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            mode: str = "classification",
            ignored_features: List = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize a SHAP explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, please set parameter ``nsamples``, e.g., ``nsamples = 100``.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param ignored_features: The features ignored in computing feature importance scores.
        :param kwargs: Additional parameters to initialize `shap.KernelExplainer`, e.g., ``nsamples``.
            Please refer to the doc of `shap.KernelExplainer`.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        if self.target_column is not None:
            assert self.target_column not in self.ignored_features, \
                f"The target column {self.target_column} cannot be in the ignored feature list."
        self.valid_indices = [i for i, f in enumerate(self.feature_columns) if f not in self.ignored_features]

        if "nsamples" not in kwargs:
            kwargs["nsamples"] = 100
        self.background_data = shap.sample(self.data, nsamples=kwargs["nsamples"])
        self.sampled_data = shap.sample(self.data, nsamples=kwargs["nsamples"])

    def _explain_global(self, X, **kwargs) -> GlobalFeatureImportance:
        if "nsamples" not in kwargs:
            kwargs["nsamples"] = 100
        instances = self.sampled_data if X is None else \
            self.transformer.transform(X.remove_target_column())

        explanations = GlobalFeatureImportance()
        explainer = shap.KernelExplainer(
            self.predict_fn, self.background_data,
            link="logit" if self.mode == "classification" else "identity", **kwargs
        )
        shap_values = explainer.shap_values(instances, **kwargs)

        if self.mode == "classification":
            values = 0
            for v in shap_values:
                values += np.abs(v)
            values /= len(shap_values)
            shap_values = values

        importance_scores = np.mean(np.abs(shap_values), axis=0)
        explanations.add(
            feature_names=self.feature_columns,
            importance_scores=importance_scores,
            sort=True
        )
        return explanations

    def explain(
            self,
            X: Tabular = None,
            **kwargs
    ):
        """
        Generates the global SHAP explanations.

        :param X: The data will be used to compute global SHAP values, i.e., the mean of the absolute
            SHAP value for each feature. If `X` is None, a set of training samples will be used.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`,
            e.g., ``nsamples`` -- the number of times to re-evaluate the model when explaining each prediction.
        :return: The global feature importance explanations.
        """
        return self._explain_global(X=X, **kwargs)

    def save(
            self,
            directory: str,
            filename: str = None,
            **kwargs
    ):
        """
        Saves the initialized explainer.

        :param directory: The folder for the dumped explainer.
        :param filename: The filename (the explainer class name if it is None).
        """
        super().save(
            directory=directory,
            filename=filename,
            ignored_attributes=["data"],
            **kwargs
        )
