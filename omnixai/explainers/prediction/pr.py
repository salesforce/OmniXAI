#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The precision recall curve for classification.
"""
import numpy as np
from typing import Callable, List
from sklearn.metrics import precision_recall_curve

from ..base import ExplainerBase
from ...explanations.prediction.pr import PrecisionRecallExplanation


class PrecisionRecall(ExplainerBase):
    """
    The precision recall curve for a classification model.
    """
    explanation_type = "prediction"
    alias = ["pr"]

    def __init__(
            self,
            predict_function: Callable,
            test_data,
            test_labels: List,
            mode: str = "classification"
    ):
        """
        :param predict_function: The prediction function corresponding to a classification model.
            The outputs of the ``predict_function`` are the class probabilities.
        :param test_data: The test data. ``test_data`` contains the raw features of the test instances.
            If ``test_data`` is a ``Tabular`` with a target/label column, this column is ignored
            (because the labels in this column are raw labels which are not processed by a LabelEncoder).
        :param test_labels: The test labels. The specified labels by ``test_labels`` will be used to
            compute precision and recall. Note that the labels in ``test_labels`` should be integers (processed
            by a LabelEncoder) and match the prediction probabilities computed by ``predict_function``.
        :param mode: The task type can be `classification` only.
        """
        super().__init__()
        assert mode == "classification", "`PrecisionRecall` only supports classification models."
        assert test_labels is not None, "Please set the test labels."
        assert len(test_labels) == len(test_data), \
            f"The length of `test_labels` is not equal to the number of examples in `test_data`, " \
            f"{len(test_labels)} != {len(test_data)}"

        self.predict_function = predict_function
        self.y_prob = predict_function(test_data)
        self.y_test = test_labels
        self.num_classes = self.y_prob.shape[1]

    def explain(self, **kwargs) -> PrecisionRecallExplanation:
        """
        Computes the precision recall curves.

        :return: The precision recall curves.
        """
        explanations = PrecisionRecallExplanation()
        y_true = np.zeros(self.y_prob.shape)
        for i, label in enumerate(self.y_test):
            y_true[i, label] = 1

        precisions, recalls = {}, {}
        for i in range(self.num_classes):
            precisions[i], recalls[i], _ = precision_recall_curve(y_true[:, i], self.y_prob[:, i])
        explanations.add(precisions, recalls)
        return explanations
