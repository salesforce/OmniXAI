#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The metrics for classification and regression.
"""
import pandas as pd
from typing import Dict
from ..base import ExplanationBase, DashFigure


class MetricExplanation(ExplanationBase):
    """
    The class for plotting metrics. For classification, it stores the metrics
    in a dict, i.e., `{"macro": metrics, "micro": metrics, 0: metrics for class 0, ...}`, and
    each "metrics" is also a dict, i.e., `{"accuracy": accuracy, "precision": precision,
    "recall": recall, ...}`. For regression, the metrics are stored in the format
    `{"mse": mean squared error, "rmse": root mean square error, ...}`.
    """

    def __init__(self, metrics: Dict, mode: str):
        super().__init__()
        self.metrics = metrics
        self.mode = mode

    def get_explanations(self):
        """
        Gets the precision recall curves.

        :return: A Dict for the metrics. For classification, it stores the metrics
            in a dict, i.e., `{"macro": metrics, "micro": metrics, 0: metrics for class 0, ...}`, and
            each "metrics" is also a dict, i.e., `{"accuracy": accuracy, "precision": precision,
            "recall": recall, ...}`. For regression, the metrics are stored in the format
            `{"mse": mean squared error, "rmse": root mean square error, ...}`.
        """
        return self.metrics

    def _metrics_to_df(self, class_names):
        if self.mode == "classification":
            values = []
            for i in range(len(self.metrics) - 2):
                values.append(["{:.4f}".format(self.metrics[i][m])
                               for m in ["precision", "recall", "f1-score", "auc"]])
            values.append(["{:.4f}".format(self.metrics["macro"][m])
                           for m in ["precision", "recall", "f1-score", "auc"]])
            values.append(["{:.4f}".format(self.metrics["micro"][m])
                           for m in ["precision", "recall", "f1-score", "auc"]])
            columns = ["Precision", "Recall", "F1-score", "AUC"]
            if class_names is None:
                index = list(range(len(self.metrics) - 2)) + ["Macro", "Micro"]
            else:
                index = [class_names[i] for i in range(len(self.metrics) - 2)] + ["Macro", "Micro"]
            df = pd.DataFrame(values, columns=columns, index=index)
        else:
            columns = ["MSE", "MAE", "MAPE", "R-squared"]
            values = [["{:.4f}".format(self.metrics[m]) for m in ["mse", "mae", "mape", "r-square"]]]
            df = pd.DataFrame(values, columns=columns)
        return df

    def plot(self, class_names=None, **kwargs):
        """
        Plots the metrics.

        :param class_names: A list of the class names indexed by the labels for classification.
            For regression, this parameter is ignored.
        :return: A matplotlib figure plotting the metrics.
        """
        import matplotlib.pyplot as plt

        df = self._metrics_to_df(class_names)
        fig, ax = plt.subplots()
        if self.mode == "classification":
            ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index.values, loc="center")
        else:
            ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        ax.axis("off")
        ax.axis("tight")
        return fig

    def _plotly_figure(self, class_names, **kwargs):
        import plotly.graph_objects as go

        df = self._metrics_to_df(class_names)
        if self.mode == "classification":
            df = df.reset_index().rename(columns={"index": "Class"})
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), align="center"),
            cells=dict(values=df.values.T, align="center"))
        ])
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the metrics using Dash.

        :param class_names: A list of the class names indexed by the labels for classification.
            For regression, this parameter is ignored.
        :return: A plotly dash figure plotting the metrics.
        """
        return DashFigure(self._plotly_figure(class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the metrics in IPython.

        :param class_names: A list of the class names indexed by the labels for classification.
            For regression, this parameter is ignored.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        metrics = {}
        for key, value in d["metrics"].items():
            try:
                key = int(key)
            except:
                pass
            metrics[key] = value
        return MetricExplanation(metrics=metrics, mode=d["mode"])
