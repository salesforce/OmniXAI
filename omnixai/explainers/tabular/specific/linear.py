#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The explainable linear models.
"""
import numpy as np
from typing import List
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression as LR

from ..base import SklearnBase
from ....data.tabular import Tabular
from ....preprocessing.base import TransformBase, Identity
from ....preprocessing.encode import OneHot, LabelEncoder
from ....preprocessing.normalize import Standard
from ....explanations.tabular.linear import LinearExplanation


class LinearBase(SklearnBase):
    """
    The base class for explainable linear models, e.g., linear regression and
    linear classification (logistic regression).
    """

    explanation_type = "both"

    def __init__(
        self,
        mode: str = "classification",
        cate_encoder: TransformBase = OneHot(drop="first"),
        cont_encoder: TransformBase = Standard(),
        target_encoder: TransformBase = LabelEncoder(),
        **kwargs
    ):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        :param cate_encoder: The encoder for categorical features, e.g.,
            `OneHot`, `Ordinal`.
        :param cont_encoder: The encoder for continuous-valued features,
            e.g., `Identity`, `Standard`, `MinMax`, `Scale`.
        :param target_encoder: The encoder for targets/labels, e.g.,
            `Identity` for regression, `LabelEncoder` for classification.
        """
        super().__init__(
            mode=mode, cate_encoder=cate_encoder, cont_encoder=cont_encoder, target_encoder=target_encoder, **kwargs
        )

    def fit(self, training_data: Tabular, train_size: float = 0.8, **kwargs) -> None:
        """
        Trains the model with the training dataset.

        :param training_data: The training dataset.
        :param train_size: The proportion of the training samples used in train-test splitting.
        """
        super(LinearBase, self).fit(training_data=training_data, train_size=train_size, **kwargs)

    def _local_scores(self, X: Tabular):
        """
        Computes the `local scores` given the input instances. The local scores
        are defined by `feature value * feature coefficient`.

        :param X: The input instances.
        :return: The local scores.
        :rtype: List
        """
        features = self.transformer.get_feature_names()
        X = self.transformer.transform(X.remove_target_column())
        assert len(features) == X.shape[1], "The number of features doesn't fit the input data."

        if self.mode == "classification":
            coefficients = self.model.coef_[0]
            intercept = self.model.intercept_[0]
        else:
            coefficients = self.model.coef_
            intercept = self.model.intercept_

        scores = []
        for i, x in enumerate(X):
            s = dict(zip(features, x * coefficients))
            s.update({"intercept": intercept})
            scores.append({k: v for k, v in s.items() if v != 0.0})
        return scores

    def _global_scores(self):
        """
        Computes the `global scores`. The global scores are defined by `linear model coefficients`.

        :return: The global scores.
        :rtype: Dict
        """
        features = self.transformer.get_feature_names()
        if self.mode == "classification":
            coefficients = self.model.coef_[0]
            intercept = self.model.intercept_[0]
        else:
            coefficients = self.model.coef_
            intercept = self.model.intercept_
        scores = dict(zip(features, coefficients))
        scores.update({"intercept": intercept})
        return scores

    def explain(self, X: Tabular, y: List = None, **kwargs):
        """
        Generates the explanations for the input instances. The explanations are either
        global or local. Global explanations are the linear coefficients. Local explanations
        are the feature importance scores of the input instances.

        :param X: A batch of input instances. Global explanations
            are generated if ``X`` is `None`.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when `y = None`.
        :param kwargs: Not used.
        :rtype: LinearExplanation
        """
        explanations = LinearExplanation(mode=self.mode)
        explanations.add(
            coefficients=self._global_scores(),
            importance_scores=self._local_scores(X) if X is not None else None,
            outputs=self.predict(X) if X is not None else None,
        )
        return explanations


class LinearRegression(LinearBase):
    """
    The linear regression model based on `Lasso`.
    """

    alias = ["linear_regression"]

    def __init__(
        self,
        cate_encoder: TransformBase = OneHot(drop="first"),
        cont_encoder: TransformBase = Standard(),
        target_encoder: TransformBase = Identity(),
        **kwargs
    ):
        """
        :param cate_encoder: The encoder for categorical features, e.g.,
            `OneHot`, `Ordinal`.
        :param cont_encoder: The encoder for continuous-valued features,
            e.g., `Identity`, `Standard`, `MinMax`, `Scale`.
        :param target_encoder: The encoder for targets/labels, e.g.,
            `Identity` for regression.
        """
        super().__init__(
            mode="regression",
            cate_encoder=cate_encoder,
            cont_encoder=cont_encoder,
            target_encoder=target_encoder,
            **kwargs
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = Lasso(
            alpha=kwargs.get("alpha", 1.0),
            fit_intercept=kwargs.get("fit_intercept", True),
            normalize=kwargs.get("normalize", False),
            precompute=kwargs.get("precompute", False),
            copy_X=kwargs.get("copy_X", True),
            max_iter=kwargs.get("max_iter", 1000),
            tol=kwargs.get("tol", 1e-4),
            warm_start=kwargs.get("warm_start", False),
            positive=kwargs.get("positive", False),
            random_state=kwargs.get("random_state", None),
            selection=kwargs.get("selection", "cyclic"),
        )
        self.model.fit(X, y, kwargs.get("sample_weight", None))


class LogisticRegression(LinearBase):
    """
    The logistic regression model.
    """

    alias = ["logistic_regression"]

    def __init__(
        self,
        cate_encoder: TransformBase = OneHot(drop="first"),
        cont_encoder: TransformBase = Standard(),
        target_encoder: TransformBase = LabelEncoder(),
        **kwargs
    ):
        """
        :param cate_encoder: The encoder for categorical features, e.g.,
            `OneHot`, `Ordinal`.
        :param cont_encoder: The encoder for continuous-valued features,
            e.g., `Identity`, `Standard`, `MinMax`, `Scale`.
        :param target_encoder: The encoder for targets/labels, e.g.,
            `LabelEncoder` for classification.
        """
        super().__init__(
            mode="classification",
            cate_encoder=cate_encoder,
            cont_encoder=cont_encoder,
            target_encoder=target_encoder,
            **kwargs
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = LR(
            penalty=kwargs.get("penalty", "l2"),
            dual=kwargs.get("dual", False),
            tol=kwargs.get("tol", 1e-4),
            C=kwargs.get("C", 1.0),
            fit_intercept=kwargs.get("fit_intercept", True),
            intercept_scaling=kwargs.get("intercept_scaling", 1),
            class_weight=kwargs.get("class_weight", None),
            random_state=kwargs.get("random_state", None),
            solver=kwargs.get("solver", "lbfgs"),
            max_iter=kwargs.get("max_iter", 500),
            multi_class=kwargs.get("multi_class", "auto"),
            verbose=kwargs.get("verbose", 0),
            warm_start=kwargs.get("warm_start", False),
            n_jobs=kwargs.get("n_jobs", None),
            l1_ratio=kwargs.get("l1_ratio", None),
        )
        self.model.fit(X, y, kwargs.get("sample_weight", None))
