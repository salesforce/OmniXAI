#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The prediction analysis for classification and regression.
"""
import numpy as np
from typing import Callable, List, Dict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score

from ..base import ExplainerBase
from ...explanations.prediction.roc import ROCExplanation
from ...explanations.prediction.pr import PrecisionRecallExplanation
from ...explanations.prediction.confusion import ConfusionMatrixExplanation
from ...explanations.prediction.cumulative import CumulativeGainExplanation
from ...explanations.prediction.metrics import MetricExplanation
from ...explanations.prediction.lift import LiftCurveExplanation


class PredictionAnalyzer(ExplainerBase):
    """
    The analysis for the prediction results of a classification or regression model.
    """
    explanation_type = "prediction"
    alias = ["prediction"]

    def __init__(
            self,
            predict_function: Callable,
            test_data,
            test_targets: List,
            mode: str = "classification"
    ):
        """
        :param predict_function: The prediction function corresponding to a classification model.
            The outputs of the ``predict_function`` are the class probabilities.
        :param test_data: The test data. ``test_data`` contains the raw features of the test instances.
            If ``test_data`` is a ``Tabular`` with a target/label column, this column is ignored
            (because the labels in this column are raw labels which are not processed by a LabelEncoder).
        :param test_targets: The test labels or targets. The specified targets by ``test_targets`` will be used to
            compute metrics and curves. For classification, ``test_targets`` should be integers (processed
            by a LabelEncoder) and match the prediction probabilities computed by ``predict_function``.
        :param mode: The task type, e.g., `classification` and `regression`.
        """
        super().__init__()
        assert mode in ["classification", "regression"], \
            "`PredictionAnalyzer` only supports classification and regression models."
        assert test_targets is not None, "Please set the test targets."
        assert len(test_targets) == len(test_data), \
            f"The length of `test_labels` is not equal to the number of examples in `test_data`, " \
            f"{len(test_targets)} != {len(test_data)}"

        self.mode = mode
        self.predict_function = predict_function
        self.y_test = test_targets
        self.y_prob = predict_function(test_data)
        if mode == "classification":
            self.num_classes = self.y_prob.shape[1]

    def _roc(self) -> ROCExplanation:
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

    def _precision_recall(self) -> PrecisionRecallExplanation:
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

    def _confusion_matrix(self) -> ConfusionMatrixExplanation:
        """
        Computes the confusion matrix given the model and the test dataset.

        :return: The confusion matrix.
        """
        y_pred = np.argmax(self.y_prob, axis=1)
        mat = confusion_matrix(self.y_test, y_pred)
        return ConfusionMatrixExplanation(mat)

    def _cumulative_gain(self) -> CumulativeGainExplanation:
        """
        Computes the cumulative gain.

        :return: The cumulative gain.
        """
        explanations = CumulativeGainExplanation()
        y_true = np.zeros(self.y_prob.shape)
        for i, label in enumerate(self.y_test):
            y_true[i, label] = 1

        percentages = np.arange(start=1, stop=y_true.shape[0] + 1)
        percentages = percentages / y_true.shape[0]
        percentages = np.insert(percentages, 0, [0])

        class_gains, class_trues = {}, {}
        for i in range(self.num_classes):
            true, score = y_true[:, i], self.y_prob[:, i]
            true = true[np.argsort(score)[::-1]]
            gains = np.cumsum(true)
            gains = gains / (np.sum(true) + 1e-8)
            gains = np.insert(gains, 0, [0])
            class_gains[i] = gains
            class_trues[i] = np.sum(true)

        explanations.add(class_gains, percentages, class_trues)
        return explanations

    def _lift_curve(self) -> LiftCurveExplanation:
        """
        Computes the cumulative lift curve.

        :return: The cumulative lift curve.
        """
        explanations = LiftCurveExplanation()
        class_gains = {}
        cg = self._cumulative_gain().get_explanations()
        percentages = cg["percentages"][1:]
        for i in range(self.num_classes):
            gains = cg["gains"][i][1:]
            class_gains[i] = gains / percentages
        explanations.add(class_gains, percentages)
        return explanations

    def _metric(self) -> MetricExplanation:
        metrics = {}
        if self.mode == "classification":
            y_pred = np.argmax(self.y_prob, axis=1)
            # Precision, recall and accuracy
            report = classification_report(self.y_test, y_pred, output_dict=True)
            for i in range(self.num_classes):
                metrics[i] = report[str(i)]
            metrics["macro"] = report["macro avg"]
            metrics["micro"] = report["weighted avg"]
            # AUC
            roc = self._roc().get_explanations()["auc"]
            for i in range(self.num_classes):
                metrics[i]["auc"] = roc[i]
            metrics["macro"]["auc"] = roc["macro"]
            metrics["micro"]["auc"] = roc["micro"]
        else:
            metrics["mse"] = mean_squared_error(self.y_test, self.y_prob)
            metrics["mae"] = mean_absolute_error(self.y_test, self.y_prob)
            metrics["mape"] = mean_absolute_percentage_error(self.y_test, self.y_prob)
            metrics["r-square"] = r2_score(self.y_test, self.y_prob)
        return MetricExplanation(metrics, self.mode)

    def _regression_predict(self):
        pass

    def _regression_residual(self):
        pass

    def explain(self, **kwargs) -> Dict:
        pass
