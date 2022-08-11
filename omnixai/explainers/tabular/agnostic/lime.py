#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LIME explainer for tabular data.
"""
import warnings
import numpy as np
from lime import lime_tabular
from typing import Callable, List

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import FeatureImportance


class LimeTabular(TabularExplainer):
    """
    The LIME explainer for tabular data.
    If using this explainer, please cite the original work: https://github.com/marcotcr/lime.
    """

    explanation_type = "local"
    alias = ["lime"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            mode: str = "classification",
            ignored_features: List = None,
            **kwargs
    ):
        """
        :param training_data: The data used to train local explainers in LIME. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``training_data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param ignored_features: The features ignored in computing feature importance scores.
        :param kwargs: Additional parameters to initialize `lime_tabular.LimeTabularExplainer`,
            e.g., ``kernel_width`` and ``discretizer``. Please refer to the doc of
            `lime_tabular.LimeTabularExplainer`.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        if self.target_column is not None:
            assert self.target_column not in self.ignored_features, \
                f"The target column {self.target_column} cannot be in the ignored feature list."

        if len(self.ignored_features) == 0:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.data,
                mode=mode,
                feature_names=self.feature_columns,
                categorical_features=self.categorical_features,
                categorical_names=self.categorical_names,
                **kwargs,
            )
            self.valid_indices = list(range(len(self.feature_columns)))
        else:
            self.valid_indices = [i for i, f in enumerate(self.feature_columns) if f not in self.ignored_features]
            categorical_names = [f for f in self.categorical_columns if f not in self.ignored_features]
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.data[:, self.valid_indices],
                mode=mode,
                feature_names=[self.feature_columns[i] for i in self.valid_indices],
                categorical_features=list(range(len(categorical_names))),
                categorical_names=categorical_names,
                **kwargs,
            )

    def explain(self, X, y=None, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters used in `LimeTabularExplainer.explain_instance`,
            e.g., ``num_features``. Please refer to the doc of
            `LimeTabularExplainer.explain_instance`.
        :return: The feature-importance explanations for all the input instances.
        """
        if "labels" in kwargs:
            warnings.warn(
                "Argument `labels` is not used, "
                "please use `y` instead of `labels` to specify "
                "the labels you want to explain."
            )
            kwargs.pop("labels")

        X = self._to_tabular(X).remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X)

        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(instances))]
                else:
                    assert len(instances) == len(y), (
                        f"Parameter `y` is a {type(y)}, the length of y "
                        f"should be the same as the number of instances in X."
                    )
                if "top_labels" in kwargs:
                    kwargs.pop("top_labels")
            else:
                kwargs["top_labels"] = 1
        else:
            y = None

        for i, instance in enumerate(instances):
            if len(self.ignored_features) == 0:
                predict_function = self.predict_fn
                test_x = instance
            else:
                def _predict(_x):
                    _y = np.tile(instance, (_x.shape[0], 1))
                    _y[:, self.valid_indices] = _x
                    return self.predict_fn(_y)
                predict_function = _predict
                test_x = instance[self.valid_indices]

            if self.mode == "classification":
                e = self.explainer.explain_instance(
                    test_x, predict_fn=predict_function, labels=(y[i],) if y is not None else None, **kwargs
                )
                exp = e.as_map().items()
            else:
                e = self.explainer.explain_instance(
                    test_x, predict_fn=predict_function, labels=(1,), **kwargs)
                exp = [(None, e.as_map()[1])]

            for label, values in exp:
                df = X.iloc(i).to_pd()
                feature_values = [df[self.feature_columns[self.valid_indices[v[0]]]].values[0] for v in values]
                feature_names = [self.feature_columns[self.valid_indices[v[0]]] for v in values]
                explanations.add(
                    instance=df,
                    target_label=label,
                    feature_names=feature_names,
                    feature_values=feature_values,
                    importance_scores=[v[1] for v in values],
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
