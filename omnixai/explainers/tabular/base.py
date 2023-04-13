#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sklearn
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Callable
from sklearn.base import BaseEstimator

from ...data.tabular import Tabular
from ...preprocessing.base import TransformBase, Identity
from ...preprocessing.encode import Ordinal, OneHot, LabelEncoder
from ...preprocessing.normalize import Standard
from ...preprocessing.tabular import TabularTransform
from ..base import ExplainerBase


class TabularExplainerMixin:

    def _to_tabular(self, X):
        """
        Converts a pandas dataframe or a numpy array into a `Tabular` object.

        :param X: The data to convert.
        :return: A `Tabular` object.
        :rtype: Tabular
        """
        if isinstance(X, Tabular):
            pass
        elif isinstance(X, pd.DataFrame):
            target_column = self.target_column if self.target_column is not None and self.target_column in X \
                else None
            X = Tabular(
                data=X,
                categorical_columns=self.categorical_columns,
                target_column=target_column
            )
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)
            target_column = self.target_column
            feature_columns = self.original_feature_columns
            if self.target_column is not None:
                if X.shape[1] != len(self.original_feature_columns):
                    target_column = None
                    feature_columns = [c for c in self.original_feature_columns if c != self.target_column]
            X = Tabular(
                data=X,
                feature_columns=feature_columns,
                categorical_columns=self.categorical_columns,
                target_column=target_column,
            )
        else:
            raise ValueError(f"Unsupported data type for TabularExplainer: {type(X)}")
        return X

    def _to_numpy(self, X):
        """
        Converts a `Tabular` object into a numpy array.

        :param X: The data to convert.
        :return: A numpy array.
        :rtype: np.ndarray
        """
        if isinstance(X, Tabular):
            return X.to_numpy(copy=False)
        elif isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported data type for TabularExplainer: {type(X)}")


class TabularExplainer(ExplainerBase, TabularExplainerMixin):
    """
    The base class of model-agnostic explainers for tabular data.
    """

    def __init__(self, training_data: Tabular, predict_function: Callable, mode: str = "classification", **kwargs):
        """
        :param training_data: The data used to initialize the explainer.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param kwargs: Additional parameters.
        """
        super().__init__()
        assert isinstance(training_data, Tabular), "training_data should be an instance of Tabular."
        self.categorical_columns = training_data.categorical_columns
        self.continuous_columns = training_data.continuous_columns
        self.target_column = training_data.target_column
        # The feature columns for the original training data
        self.original_feature_columns = training_data.columns
        self.mode = mode

        if mode == "classification":
            self.transformer = TabularTransform(cate_transform=Ordinal(), cont_transform=Identity()).fit(training_data)
        elif mode == "regression":
            self.transformer = TabularTransform(
                cate_transform=Ordinal(), cont_transform=Identity(), target_transform=Identity()
            ).fit(training_data)
        else:
            raise ValueError(f"Unknown mode: {mode}, " f"please choose `classification` or `regression`")

        # The feature columns for the transformed data
        self.feature_columns = self.transformer.get_feature_names()
        self.categorical_features = list(range(len(self.categorical_columns)))
        try:
            categories = self.transformer.categories
            self.categorical_names = {feature: categories[feature] for feature in self.categorical_features}
        except AttributeError:
            self.categorical_names = None

        self.data = self.transformer.transform(training_data)
        if training_data.target_column is not None:
            self.data = self.data[:, :-1]
        self.dim = self.data.shape[1]

        if mode == "classification":
            self.predict_fn = lambda x: predict_function(self.transformer.invert(x))
        else:
            self.predict_fn = lambda x: predict_function(self.transformer.invert(x)).flatten()


class SklearnBase(ExplainerBase, BaseEstimator):
    """
    The base class of model-specific explainers for scikit-learn models.
    """

    def __init__(
        self,
        mode: str = "classification",
        cate_encoder: TransformBase = OneHot(),
        cont_encoder: TransformBase = Standard(),
        target_encoder: TransformBase = LabelEncoder(),
        **kwargs,
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
        super().__init__()
        assert mode in [
            "classification",
            "regression",
        ], f"Unknown mode: {mode}, please choose `classification` or `regression`"
        self.mode = mode
        self.cate_encoder = cate_encoder
        self.cont_encoder = cont_encoder
        # For regression tasks, `target_encoder` uses `Identity` only
        if mode == "classification":
            self.target_encoder = target_encoder
        else:
            self.target_encoder = Identity()

        self.transformer = None
        self.cate_features = []
        self.cont_features = []

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = None

    def fit(self, training_data: Tabular, train_size: float = 0.8, **kwargs):
        """
        Trains the model given the training dataset.

        :param training_data: The training dataset.
        :param train_size: Used in train-test splits, i.e., the proportion
            of the training samples.
        """
        assert isinstance(training_data, Tabular), (
            f"`training_data` should be an instance of `Tabular` " f"instead of `{type(training_data)}`."
        )
        assert training_data.target_column is not None, "`training_data` should have a label/target column."

        self.transformer = TabularTransform(
            cate_transform=self.cate_encoder, cont_transform=self.cont_encoder, target_transform=self.target_encoder
        ).fit(training_data)

        data = self.transformer.transform(training_data)
        if train_size < 1.0:
            train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
                data[:, :-1], data[:, -1], train_size=train_size
            )
        else:
            train_x, test_x, train_y, test_y = data[:, :-1], None, data[:, -1], None

        self.num_training_samples = train_x.shape[0]
        self._fit(train_x, train_y, **kwargs)

        if train_size < 1.0:
            pred_y = self.predict(self.transformer.invert(test_x))
            if self.mode == "classification":
                print(f"Validation accuracy: " f"{sklearn.metrics.accuracy_score(test_y, pred_y)}")
            else:
                print(f"Validation MSE: " f"{np.mean((pred_y - test_y) ** 2)}")

        if self.mode == "classification":
            self.predict_function = lambda z: self.model.predict_proba(self.transformer.transform(z))
        else:
            self.predict_function = lambda z: self.model.predict(self.transformer.transform(z))
        return self

    def predict(self, X: Tabular) -> np.ndarray:
        """
        Predicts targets or labels.

        :param X: The test dataset.
        :return: The predictions.
        :rtype: np.ndarray
        """
        assert self.transformer is not None, "The linear model is not trained, please run `fit` first."
        return self.model.predict(self.transformer.transform(X.remove_target_column()))

    def predict_proba(self, X: Tabular) -> np.ndarray:
        """
        Predicts class probabilities in classification.

        :param X: The test dataset.
        :return: The predicted class probabilities.
        :rtype: np.ndarray
        """
        assert self.transformer is not None, "The linear model is not trained, please run `fit` first."
        return self.model.predict_proba(self.transformer.transform(X.remove_target_column()))

    def class_names(self):
        """
        Returns the class names in classification.

        :return: The class names.
        :rtype: List
        """
        if self.mode == "classification":
            names = self.transformer.class_names
            if names is None:
                raise RuntimeError(
                    "No class_names found because the encoder " "for the target/label column is not `LabelEncoder`."
                )
            return names
        else:
            raise RuntimeError("No class_names found because it is a regression task.")
