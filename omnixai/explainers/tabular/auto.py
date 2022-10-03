#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd
from typing import Collection, Callable, Any, Dict

from ...data.tabular import Tabular
from ..base import AutoExplainerBase


class TabularExplainer(AutoExplainerBase):
    """
    The class derived from `AutoExplainerBase` for tabular data,
    allowing users to choose multiple explainers and generate
    different explanations at the same time.

    .. code-block:: python

        explainers = TabularExplainer(
            explainers=["lime", "shap", "mace", "pdp"],
            mode="classification",
            data=data,
            model=model,
            preprocess=preprocess_function,
            postprocess=None,
            params={
                "lime": {"kernel_width": 3},
                "shap": {"nsamples": 100},
                "mace": {"ignored_features": ["Sex", "Race", "Relationship", "Capital Loss"]}
            }
        )
        local_explanations = explainers.explain(x)
        global_explanations = explainers.explain_global()
    """

    _MODELS = AutoExplainerBase._EXPLAINERS[__name__.split(".")[2]]

    def __init__(
            self,
            explainers: Collection,
            mode: str,
            data: Tabular,
            model: Any,
            preprocess: Callable = None,
            postprocess: Callable = None,
            params: Dict = None,
    ):
        """
        :param explainers: The names or alias of the explainers to use.
        :param mode: The task type, e.g. `classification` or `regression`.
        :param data: The training data used to initialize explainers. ``data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, ``data`` can be its subset by applying
            `omnixai.sampler.tabular.Sampler.subsample`.
        :param model: The machine learning model to explain, which can be a scikit-learn model,
            a tensorflow model, a torch model, or a black-box prediction function.
        :param preprocess: The preprocessing function that converts the raw input features
            into the inputs of ``model``.
        :param postprocess: The postprocessing function that transforms the outputs of ``model``
            to a user-specific form, e.g., the predicted probability for each class.
        :param params: A dict containing the additional parameters for initializing each explainer,
            e.g., `params["lime"] = {"param_1": param_1, ...}`.
        """
        super().__init__(
            explainers=explainers,
            mode=mode,
            data=data,
            model=model,
            preprocess=preprocess,
            postprocess=postprocess,
            params=params,
        )
        if data is not None:
            self.data_info = {
                "categorical_columns": data.categorical_columns,
                "target_column": data.target_column,
                "feature_columns": data.columns
            }

    def _convert_data(self, X):
        if isinstance(X, Tabular):
            return X
        if len(self.data_info) == 0:
            raise TypeError(f"The input X is not a `Tabular` instance, "
                            f"please convert {type(X)} into `Tabular`")

        cate_columns = self.data_info["categorical_columns"]
        target_column = self.data_info["target_column"]
        feature_columns = self.data_info["feature_columns"]

        if isinstance(X, pd.DataFrame):
            target_column = target_column if target_column is not None and target_column in X \
                else None
            X = Tabular(
                data=X,
                categorical_columns=cate_columns,
                target_column=target_column
            )
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)
            if target_column is not None:
                if X.shape[1] != len(feature_columns):
                    feature_columns = [c for c in feature_columns if c != target_column]
                    target_column = None
            X = Tabular(
                data=X,
                feature_columns=feature_columns,
                categorical_columns=cate_columns,
                target_column=target_column
            )
        else:
            raise ValueError(f"Unsupported data type for TabularExplainer: {type(X)}")
        return X

    @staticmethod
    def list_explainers():
        """
        List the supported explainers.
        """
        from tabulate import tabulate
        lists = []
        for _class in TabularExplainer._MODELS:
            alias = _class.alias if hasattr(_class, "alias") else _class.__name__
            explanation_type = _class.explanation_type \
                if _class.explanation_type != "both" else "global & local"
            lists.append([_class.__module__, _class.__name__, alias, explanation_type])
        table = tabulate(
            lists,
            headers=["Package", "Explainer Class", "Alias", "Explanation Type"],
            tablefmt='orgtbl'
        )
        print(table)
