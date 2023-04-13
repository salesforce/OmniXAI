#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The explainable tree-based models.
"""
import numpy as np
from typing import List

from ..base import SklearnBase
from ....data.tabular import Tabular
from ....preprocessing.base import TransformBase, Identity
from ....preprocessing.encode import OneHot, LabelEncoder
from ....explanations.tabular.tree import TreeExplanation


class TreeBase(SklearnBase):
    """
    The base class for explainable tree models, e.g., decision trees, random forests, xgboost.
    """

    explanation_type = "both"

    def __init__(
        self,
        mode: str = "classification",
        cate_encoder: TransformBase = OneHot(),
        cont_encoder: TransformBase = Identity(),
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
        super(TreeBase, self).fit(training_data=training_data, train_size=train_size, **kwargs)

    def _local_explanations(self, X: Tabular):
        """
        Generates the local explanations given the input instances. The local explanations
        include decision paths.

        :param X: The input instances.
        :return: The local explanations.
        """
        paths = []
        instances = self.transformer.transform(X)
        node_indicator = self.model.decision_path(instances)
        leaf_id = self.model.apply(instances)
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        feature_names = self.transformer.get_feature_names()

        for sample_id in range(instances.shape[0]):
            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]

            path = []
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id[sample_id] == node_id:
                    continue
                # check if value of the split feature for sample 0 is below threshold
                if instances[sample_id, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                path.append(
                    {
                        "node": node_id,
                        "feature": feature_names[feature[node_id]],
                        "value": instances[sample_id, feature[node_id]],
                        "inequality": threshold_sign,
                        "threshold": threshold[node_id],
                        "text": "node {node} : {feature} = {value}) "
                        "{inequality} {threshold})".format(
                            node=node_id,
                            feature=feature_names[feature[node_id]],
                            value=instances[sample_id, feature[node_id]],
                            inequality=threshold_sign,
                            threshold=threshold[node_id],
                        ),
                    }
                )
            paths.append(path)

        return paths, node_indicator

    def explain(self, X: Tabular = None, y: List = None, **kwargs):
        """
        Generates the explanations for the input instances. The explanations are either
        global or local. Global explanations are the tree structure. Local explanations
        are the decision paths of the input instances.

        :param X: A batch of input instances. Global explanations
            are generated if ``X`` is `None`.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label for each input instance will be explained
            when `y = None`.
        :rtype: TreeExplanation
        """
        explanations = TreeExplanation()
        if X is None:
            explanations.add_global(
                model=self.model,
                feature_names=self.transformer.get_feature_names(),
                class_names=self.transformer.class_names,
            )
        else:
            paths, node_indicator = self._local_explanations(X.remove_target_column())
            explanations.add_local(
                model=self.model,
                decision_paths=paths,
                node_indicator=node_indicator,
                feature_names=self.transformer.get_feature_names(),
                class_names=self.transformer.class_names,
            )
        return explanations


class TreeRegressor(TreeBase):
    """
    The tree regressor based on `sklearn.tree.DecisionTreeRegressor`.
    """

    alias = ["tree_regressor"]

    def __init__(
        self,
        cate_encoder: TransformBase = OneHot(),
        cont_encoder: TransformBase = Identity(),
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
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor(**kwargs)
        self.model.fit(X, y, kwargs.get("sample_weight", None))


class TreeClassifier(TreeBase):
    """
    The tree classifier based on `sklearn.tree.DecisionTreeClassifier`.
    """

    alias = ["tree_classifier"]

    def __init__(
        self,
        cate_encoder: TransformBase = OneHot(),
        cont_encoder: TransformBase = Identity(),
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
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(**kwargs)
        self.model.fit(X, y, kwargs.get("sample_weight", None))
