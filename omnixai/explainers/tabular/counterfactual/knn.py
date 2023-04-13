#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The KNN-based counterfactual explainer for tabular data.
"""
import numpy as np
import pandas as pd
from typing import List, Callable, Union

from ..base import ExplainerBase
from ...tabular.base import TabularExplainerMixin
from ....data.tabular import Tabular
from ....explanations.tabular.counterfactual import CFExplanation

from .mace.retrieval import CFRetrieval
from .mace.diversify import DiversityModule


class KNNCounterfactualExplainer(ExplainerBase, TabularExplainerMixin):
    """
    The counterfactual explainer for tabular data based on KNN search. Given a query instance,
    it finds the instances in the training dataset that are close to the query with the desired label.
    """
    explanation_type = "local"
    alias = ["ce_knn", "knn_ce"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            mode: str = "classification",
            **kwargs,
    ):
        """
        :param training_data: The data used to initialize a MACE explainer. ``training_data``
            can be the training dataset for training the machine learning model.
        :param predict_function: The prediction function corresponding to the model to explain.
            The model should be a classifier, the outputs of the ``predict_function``
            are the class probabilities.
        :param mode: The task type can be `classification` only.
        """
        super().__init__()
        assert mode == "classification", "MACE supports classification tasks only."
        self.predict_function = predict_function

        self.categorical_columns = training_data.categorical_columns
        self.target_column = training_data.target_column
        self.original_feature_columns = training_data.columns

        self.recall = CFRetrieval(training_data, predict_function, None, **kwargs)
        self.diversity = DiversityModule(training_data)

    def explain(
            self,
            X: Tabular,
            y: Union[List, np.ndarray] = None,
            max_number_examples: int = 5,
            **kwargs
    ) -> CFExplanation:
        """
        Generates counterfactual explanations.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of the desired labels, which should be different from the predicted labels of ``X``.
            If ``y = None``, the desired labels will be the labels different from the predicted labels of ``X``.
        :param max_number_examples: The maximum number of the generated counterfactual
            examples per class for each input instance.
        :return: A CFExplanation object containing the generated explanations.
        """
        if y is not None:
            assert len(y) == X.shape[0], (
                f"The length of `y` should equal the number of instances in `X`, " f"got {len(y)} != {X.shape[0]}"
            )

        X = self._to_tabular(X).remove_target_column()
        scores = self.predict_function(X)
        labels = np.argmax(scores, axis=1)
        num_classes = scores.shape[1]

        explanations = CFExplanation()
        for i in range(X.shape[0]):
            x = X.iloc(i)
            label = int(labels[i])
            if y is None or y[i] == label:
                desired_labels = [z for z in range(num_classes) if z != label]
            else:
                desired_labels = [int(y[i])]

            all_cfs = []
            for desired_label in desired_labels:
                df, _ = self.recall.get_nn_samples(x, desired_label)
                examples = Tabular(df, categorical_columns=x.categorical_columns)
                cfs = self.diversity.get_diverse_cfs(
                    self.predict_function, x, examples,
                    oracle_function=lambda _s: int(desired_label == np.argmax(_s)),
                    desired_label=desired_label, k=max_number_examples
                )
                cfs_df = cfs.to_pd()
                if x.continuous_columns:
                    cfs_df = cfs_df.astype({c: float for c in x.continuous_columns})
                cfs_df["label"] = desired_label
                all_cfs.append(cfs_df)

            instance_df = x.to_pd()
            instance_df["label"] = label
            explanations.add(query=instance_df, cfs=pd.concat(all_cfs) if len(all_cfs) > 0 else None)
        return explanations
