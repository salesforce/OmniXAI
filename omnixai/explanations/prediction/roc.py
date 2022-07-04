#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from typing import Dict
from ..base import ExplanationBase, DashFigure


class ROCExplanation(ExplanationBase):
    """
    The class for plotting ROC curves. It stores the false positive rates, true positive rates and AUCs
    in a dict, i.e., `{"fpr": the false positive rates, "tpr": the true positive rates, "auc": the ROC areas}`.
    Each of "fpr", "tpr" and "auc" is a dict with the following format `{"micro": micro-average roc,
    "macro": macro-average roc, 0: roc for label 0, ...}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, fpr: Dict, tpr: Dict, auc: Dict):
        """
        Adds the false positive rates, true positive rates and AUCs.

        :param fpr: The false positive rates. ``fpr`` is a dict with the following format
            `{"micro": micro-average roc, "macro": macro-average roc, 0: roc for label 0, ...}`.
        :param tpr: The true positive rates. ``tpr`` is a dict with the following format
            `{"micro": micro-average roc, "macro": macro-average roc, 0: roc for label 0, ...}`.
        :param auc: The ROC areas. ``auc`` is a dict with the following format
            `{"micro": micro-average roc, "macro": macro-average roc, 0: roc for label 0, ...}`.
        """
        self.explanations = {"fpr": fpr, "tpr": tpr, "auc": auc}

    def get_explanations(self):
        """
        Gets the ROC curves.

        :return: A Dict for the ROC curves, i.e.,
            `{"fpr": the false positive rates, "tpr": the true positive rates, "auc": the ROC areas}`.
            Each of "fpr", "tpr" and "auc" is a dict with the following format
            `{"micro": micro-average roc, "macro": macro-average roc, 0: roc for label 0, ...}`.
        """
        return self.explanations

    def plot(self, linewidth=2, **kwargs):
        """
        Plots the ROC curves.

        :return: A matplotlib figure plotting the ROC curves.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fpr = self.explanations["fpr"]
        tpr = self.explanations["tpr"]
        auc = self.explanations["auc"]
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig = plt.figure()
        plt.plot(
            fpr["micro"], tpr["micro"],
            label="Micro-average ROC curve (area = {0:0.2f})".format(auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=linewidth
        )
        plt.plot(
            fpr["macro"], tpr["macro"],
            label="Macro-average ROC curve (area = {0:0.2f})".format(auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=linewidth
        )
        for i in range(len(fpr) - 2):
            plt.plot(
                fpr[i], tpr[i],
                color=colors[i % len(colors)],
                linewidth=linewidth,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=linewidth)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        return fig

    def _plotly_figure(self, linewidth=2, **kwargs):
        import plotly.graph_objects as go

        fpr = self.explanations["fpr"]
        tpr = self.explanations["tpr"]
        auc = self.explanations["auc"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr["micro"],
            y=tpr["micro"],
            name="Micro-average ROC curve (area = {0:0.2f})".format(auc["micro"]),
            line=dict(width=linewidth),
        ))
        fig.add_trace(go.Scatter(
            x=fpr["macro"],
            y=tpr["macro"],
            name="Macro-average ROC curve (area = {0:0.2f})".format(auc["macro"]),
            line=dict(width=linewidth),
        ))
        for i in range(len(fpr) - 2):
            fig.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                name="ROC curve of class {0} (area = {1:0.2f})".format(i, auc[i]),
                line=dict(width=linewidth),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(color="black", dash="dash", width=linewidth),
            name="Baseline"
        ))
        return fig

    def plotly_plot(self, **kwargs):
        """
        Plots the ROC curves using Dash.

        :return: A plotly dash figure plotting the ROC curves.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Plots the ROC curves in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))
