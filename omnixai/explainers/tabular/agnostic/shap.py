#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for tabular data.
"""
import shap
import numpy as np
from typing import Callable

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import FeatureImportance


class ShapTabular(TabularExplainer):
    """
    The SHAP explainer for tabular data.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "local"
    alias = ["shap"]

    def __init__(self, training_data: Tabular, predict_function: Callable, mode: str = "classification", **kwargs):
        """
        :param training_data: The data used to initialize a SHAP explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, please set parameter ``nsamples``, e.g., ``nsamples = 100``.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param kwargs: Additional parameters to initialize `shap.KernelExplainer`, e.g., ``nsamples``.
            Please refer to the doc of `shap.KernelExplainer`.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        if "nsamples" in kwargs:
            data = shap.sample(self.data, nsamples=kwargs["nsamples"])
        else:
            data = self.data

        self.explainer = shap.KernelExplainer(
            self.predict_fn, data, link="logit" if mode == "classification" else "identity", **kwargs
        )

    def explain(self, X, y=None, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`.
        :return: The feature-importance explanations for all the input instances.
        """
        X = self._to_tabular(X).remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X)
        shap_values = self.explainer.shap_values(instances, **kwargs)

        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(instances))]
                else:
                    assert len(instances) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of instances in X."
                    )
            else:
                prediction_scores = self.predict_fn(instances)
                y = np.argmax(prediction_scores, axis=1)
        else:
            y = None

        for i, instance in enumerate(instances):
            df = X.iloc(i).to_pd()
            feature_values = [df[self.feature_columns[feat]].values[0] for feat in range(len(self.feature_columns))]
            if self.mode == "classification":
                label = y[i]
                importance_scores = shap_values[label][i]
                explanations.add(
                    instance=df,
                    target_label=label,
                    feature_names=self.feature_columns,
                    feature_values=feature_values,
                    importance_scores=importance_scores,
                    sort=True,
                )
            else:
                explanations.add(
                    instance=df,
                    target_label=None,
                    feature_names=self.feature_columns,
                    feature_values=feature_values,
                    importance_scores=shap_values[i],
                    sort=True,
                )
        return explanations
