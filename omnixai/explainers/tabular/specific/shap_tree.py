#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The tree-specific SHAP explainer for tabular data.
"""
import shap
import numpy as np
from typing import List, Any

from ..base import SklearnBase
from ....preprocessing.base import TransformBase, Identity
from ....preprocessing.encode import OneHot, LabelEncoder
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import FeatureImportance


class ShapTreeTabular(SklearnBase):
    """
    The tree-specific SHAP explainer for tabular data.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "local"
    alias = ["shap_tree"]

    def __init__(
        self,
        mode: str = "classification",
        model: Any = None,
        cate_encoder: TransformBase = OneHot(),
        cont_encoder: TransformBase = Identity(),
        target_encoder: TransformBase = LabelEncoder(),
        **kwargs,
    ):
        """
        :param mode: The task type, e.g. `classification` or `regression`.
        :param model: The tree-based models, e.g., scikit-learn decision trees, xgboost.
        :param cate_encoder: The encoder for categorical features, e.g.,
            `OneHot`, `Ordinal`.
        :param cont_encoder: The encoder for continuous-valued features,
            e.g., `Identity`, `Standard`, `MinMax`, `Scale`.
        :param target_encoder: The encoder for targets/labels, e.g.,
            `LabelEncoder` for classification.
        :param kwargs: Additional parameters.
        """
        super().__init__(
            mode=mode, cate_encoder=cate_encoder, cont_encoder=cont_encoder, target_encoder=target_encoder, **kwargs
        )
        assert model is not None, "Please specify the model."
        self.model = model
        self.explainer = None
        self.feature_names = None
        self.kwargs = kwargs

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y, **kwargs)

    def fit(self, training_data: Tabular, train_size: float = 0.8, **kwargs):
        """
        Trains the model with the training dataset.

        :param training_data: The training dataset.
        :param train_size: The proportion of the training samples used in train-test splitting.
        """
        super(ShapTreeTabular, self).fit(training_data=training_data, train_size=train_size, **kwargs)
        self.feature_names = self.transformer.get_feature_names()
        data = self.transformer.transform(training_data.remove_target_column())
        if "nsamples" in self.kwargs:
            data = shap.sample(data, nsamples=self.kwargs["nsamples"])
        self.explainer = shap.TreeExplainer(self.model, data, **kwargs)

    def explain(self, X: Tabular, y: List = None, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label for each input instance will be explained
            when `y = None`.
        :param kwargs: Not used.
        """
        X = X.remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X)
        shap_values = self.explainer.shap_values(instances, **kwargs)
        predict_fn = self.model.predict_proba if self.mode == "classification" \
            else self.model.predict

        # For xgboost.XGBClassifier, sometimes the output of shap tree doesn't
        # include the SHAP values for different labels (which may be a bug)
        multiple_outputs = False
        if isinstance(shap_values, list):
            multiple_outputs = True

        if self.mode == "classification":
            if y is not None and multiple_outputs:
                if type(y) == int:
                    y = [y for _ in range(len(instances))]
                else:
                    assert len(instances) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of instances in X."
                    )
            else:
                prediction_scores = predict_fn(instances)
                y = np.argmax(prediction_scores, axis=1)
        else:
            y = None

        for i, instance in enumerate(instances):
            label = int(y[i]) if y is not None else None
            if label is not None and multiple_outputs:
                importance_scores = shap_values[label][i]
            else:
                importance_scores = shap_values[i]
            assert len(importance_scores) == len(self.feature_names), "A bug occurs in ShapTreeTabular.explain."

            explanations.add(
                instance=X.iloc(i).to_pd(),
                target_label=label,
                feature_names=self.feature_names,
                feature_values=instance,
                importance_scores=importance_scores,
                sort=True,
            )
        return explanations
