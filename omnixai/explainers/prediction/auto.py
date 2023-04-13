#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The prediction analysis for classification and regression.
"""
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score

from ..base import ExplainerBase
from ...utils.misc import build_predict_function
from ...explanations.prediction.roc import ROCExplanation
from ...explanations.prediction.pr import PrecisionRecallExplanation
from ...explanations.prediction.confusion import ConfusionMatrixExplanation
from ...explanations.prediction.cumulative import CumulativeGainExplanation
from ...explanations.prediction.metrics import MetricExplanation
from ...explanations.prediction.lift import LiftCurveExplanation
from ...explanations.prediction.residual import ResidualExplanation


class PredictionAnalyzer(ExplainerBase):
    """
    The analysis for the prediction results of a classification or regression model:

    .. code-block:: python

        analyzer = PredictionAnalyzer(
            mode="classification",            # The task type, e.g., "classification" or "regression"
            test_data=test_data,              # The test dataset (a `Tabular` instance)
            test_targets=test_labels,         # The test labels (a numpy array)
            model=model,                      # The ML model
            preprocess=preprocess_function    # Converts raw features into the model inputs
        )
        prediction_explanations = analyzer.explain()
    """
    explanation_type = "prediction"
    alias = ["prediction"]

    def __init__(
            self,
            mode: str,
            test_data,
            test_targets,
            model: Any = None,
            preprocess: Callable = None,
            postprocess: Callable = None,
            predict_function: Callable = None,
            **kwargs
    ):
        """
        :param mode: The task type, e.g., `classification` and `regression`.
        :param test_data: The test data. ``test_data`` contains the raw features of the test instances.
            If ``test_data`` is a ``Tabular`` with a target/label column, this column is ignored
            (because the labels in this column are raw labels which are not processed by a LabelEncoder).
        :param test_targets: The test labels or targets. The specified targets by ``test_targets`` will be used to
            compute metrics and curves. For classification, ``test_targets`` should be integers (processed
            by a LabelEncoder) and match the class probabilities returned by the ML model.
        :param model: The machine learning model to analyze, which can be a scikit-learn model,
            a tensorflow model, a torch model, or a black-box prediction function.
        :param preprocess: The preprocessing function that converts the raw input features
            into the inputs of ``model``.
        :param postprocess: The postprocessing function that transforms the outputs of ``model``
            to a user-specific form, e.g., the predicted class probabilities.
        :param predict_function: The prediction function corresponding to the ML model.
            The outputs of the ``predict_function`` are the class probabilities. If ``predict_function``
            is not None, ``PredictionAnalyzer`` will ignore ``model`` and use ``predict_function`` only
            to generate prediction results.
        """
        super().__init__()
        assert mode in ["classification", "regression"], \
            "`PredictionAnalyzer` only supports classification and regression models."
        assert test_targets is not None, "Please set the test targets."

        if isinstance(test_targets, (list, tuple)):
            test_targets = np.array(test_targets)
        elif isinstance(test_targets, pd.DataFrame):
            test_targets = test_targets.values
        elif isinstance(test_targets, np.ndarray):
            test_targets = test_targets
        else:
            raise ValueError(f"The type of `test_targets` is {type(test_targets)}, which is not supported."
                             f"`test_targets` should be a list, a numpy array or a pandas dataframe.")
        if test_targets.ndim > 1:
            test_targets = test_targets.flatten()

        assert len(test_targets) == len(test_data), \
            f"The length of `test_labels` is not equal to the number of examples in `test_data`, " \
            f"{len(test_targets)} != {len(test_data)}"
        assert model is not None or predict_function is not None, \
            "Both `model` and `predict_function` are None, please set either of them."

        if predict_function is None:
            self.predict_function = build_predict_function(
                model=model,
                preprocess=preprocess,
                postprocess=postprocess,
                mode=mode
            )
        else:
            self.predict_function = predict_function

        self.mode = mode
        self.y_test = test_targets.astype(int) if mode == "classification" else test_targets
        self.y_prob = self._predict(test_data, batch_size=kwargs.get("batch_size", 128))
        if mode == "classification":
            self.num_classes = self.y_prob.shape[1]

    def _predict(self, x, batch_size=128):
        n, predictions = x.shape[0], []
        for i in range(0, n, batch_size):
            predictions.append(self.predict_function(x[i: i + batch_size]))
        z = np.concatenate(predictions, axis=0)
        return z.flatten() if self.mode == "regression" else z

    def _roc(self, **kwargs) -> ROCExplanation:
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

    def _precision_recall(self, **kwargs) -> PrecisionRecallExplanation:
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

    def _confusion_matrix(self, **kwargs) -> ConfusionMatrixExplanation:
        """
        Computes the confusion matrix given the model and the test dataset.

        :return: The confusion matrix.
        """
        y_pred = np.argmax(self.y_prob, axis=1)
        mat = confusion_matrix(self.y_test, y_pred)
        return ConfusionMatrixExplanation(mat)

    def _cumulative_gain(self, **kwargs) -> CumulativeGainExplanation:
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

    def _lift_curve(self, **kwargs) -> LiftCurveExplanation:
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

    def _metric(self, **kwargs) -> MetricExplanation:
        metrics = {}
        if self.mode == "classification":
            y_pred = np.argmax(self.y_prob, axis=1)
            # Precision, recall and accuracy
            report = classification_report(self.y_test, y_pred, output_dict=True)
            metrics["macro"] = report["macro avg"]
            metrics["micro"] = report["weighted avg"]
            for key, value in report.items():
                try:
                    metrics[int(key)] = value
                except:
                    pass
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

    def _regression_residual(self, residual_type="diff", **kwargs) -> ResidualExplanation:
        if residual_type == "diff":
            r = self.y_test - self.y_prob
        elif residual_type == "ratio":
            r = np.abs(self.y_test) / (np.abs(self.y_prob) + 1e-6)
        elif residual_type == "log_ratio":
            r = np.log(np.maximum(np.abs(self.y_test) / (np.abs(self.y_prob) + 1e-6), 1e-3))
        else:
            raise ValueError(f"Unknown regression residual type: {residual_type}, "
                             f"please choose from 'diff', 'ratio' and 'log_ratio'.")
        return ResidualExplanation(self.y_prob, r, residual_type)

    def explain(self, **kwargs) -> Dict:
        results = {"metric": self._metric(**kwargs)}
        if self.mode == "classification":
            results["confusion_matrix"] = self._confusion_matrix(**kwargs)
            results["roc"] = self._roc(**kwargs)
            results["precision_recall"] = self._precision_recall(**kwargs)
            results["cumulative_gain"] = self._cumulative_gain(**kwargs)
            results["lift_curve"] = self._lift_curve(**kwargs)
        else:
            results["residual"] = self._regression_residual(**kwargs)
        return results
