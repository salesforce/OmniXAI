#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The ROC curve for classification
"""
import numpy as np
from typing import Callable, List
from sklearn.metrics import roc_curve, auc

from ...data.tabular import Tabular
from ..base import ExplainerBase
from ...explanations.prediction.roc import ROCExplanation


class ROC(ExplainerBase):
    """
    The ROC curve for a classification model.
    """
    explanation_type = "global"
    alias = ["roc"]

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
            compute ROC and AUC. Note that the labels in ``test_labels`` should be integers (processed
            by a LabelEncoder) and match the prediction probabilities computed by ``predict_function``.
        :param mode: The task type can be `classification` only.
        """
        super().__init__()
        assert mode == "classification", "ROC only supports classification models."
        assert test_labels is not None, "Please set the test labels."
        assert len(test_labels) == len(test_data), \
            f"The length of `test_labels` is not equal to the number of examples in `test_data`, " \
            f"{len(test_labels)} != {len(test_data)}"

        self.predict_function = predict_function
        self.y_prob = predict_function(
            test_data.remove_target_column() if isinstance(test_data, Tabular) else test_data)
        self.y_test = test_labels
        self.num_classes = self.y_prob.shape[1]

    def explain(self, **kwargs) -> ROCExplanation:
        """
        Computes the micro-average ROC curve, macro-average ROC curve and ROC curves of all the classes.

        :return: All the ROC curves.
        """
        explanation = ROCExplanation()
        y_true = np.zeros(self.y_prob.shape)
        for i, label in enumerate(self.y_test):
            y_true[i, label] = 1

        # Class-specific ROC curve
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], self.y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), self.y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        explanation.add(fpr, tpr, roc_auc)
        return explanation
