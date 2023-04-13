#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for tabular data.
"""
import shap
import numpy as np
from typing import Callable, List

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import \
    FeatureImportance


class ShapTabular(TabularExplainer):
    """
    The SHAP explainer for tabular data.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """
    explanation_type = "local"
    alias = ["shap"]

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
        self.link = kwargs.get("link", None)
        if self.link is None:
            self.link = "logit" if self.mode == "classification" else "identity"
        else:
            del kwargs["link"]

        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        if self.target_column is not None:
            assert self.target_column not in self.ignored_features, \
                f"The target column {self.target_column} cannot be in the ignored feature list."
        self.valid_indices = [i for i, f in enumerate(self.feature_columns) if f not in self.ignored_features]

        self.background_data = shap.sample(self.data, nsamples=kwargs.get("nsamples", 100))
        self.explainer = shap.KernelExplainer(self.predict_fn, self.background_data, link=self.link, **kwargs)

    def explain(self, X, y=None, **kwargs) -> FeatureImportance:
        """
        Generates the local SHAP explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`,
            e.g., ``nsamples`` -- the number of times to re-evaluate the model when explaining each prediction.
        :return: The feature importance explanations.
        """
        X = self._to_tabular(X).remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X)

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

        if len(self.ignored_features) == 0:
            shap_values = self.explainer.shap_values(instances, **kwargs)
            for i, instance in enumerate(instances):
                df = X.iloc(i).to_pd()
                feature_values = \
                    [df[self.feature_columns[feat]].values[0] for feat in range(len(self.feature_columns))]
                if self.mode == "classification":
                    label = y[i]
                    importance_scores = shap_values[label][i]
                else:
                    label = None
                    importance_scores = shap_values[i]
                explanations.add(
                    instance=df,
                    target_label=label,
                    feature_names=self.feature_columns,
                    feature_values=feature_values,
                    importance_scores=importance_scores,
                    sort=True,
                )
        else:
            for i, instance in enumerate(instances):
                def _predict(_x):
                    _y = np.tile(instance, (_x.shape[0], 1))
                    _y[:, self.valid_indices] = _x
                    return self.predict_fn(_y)

                predict_function = _predict
                test_x = instance[self.valid_indices]
                explainer = shap.KernelExplainer(
                    predict_function, self.background_data[:, self.valid_indices],
                    link=self.link, **kwargs
                )
                shap_values = explainer.shap_values(np.expand_dims(test_x, axis=0), **kwargs)

                df = X.iloc(i).to_pd()
                feature_values = [df[self.feature_columns[f]].values[0] for f in self.valid_indices]
                feature_names = [self.feature_columns[f] for f in self.valid_indices]
                if self.mode == "classification":
                    label = y[i]
                    importance_scores = shap_values[label][0]
                else:
                    label = None
                    importance_scores = shap_values[0]
                explanations.add(
                    instance=df,
                    target_label=label,
                    feature_names=feature_names,
                    feature_values=feature_values,
                    importance_scores=importance_scores,
                    sort=True,
                )
        return explanations

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
