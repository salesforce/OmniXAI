#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The precision recall curves.
"""
from typing import Dict
from ..base import ExplanationBase, DashFigure


class PrecisionRecallExplanation(ExplanationBase):
    """
    The class for plotting precision recall curves. It stores the precisions and recalls
    in a dict, i.e., `{"precision": the precision, "recall": the recall}`.
    Each of "precision" and "recall" is a dict with the following format `{0: precision or
    recall for class 0, ...}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, precision: Dict, recall: Dict):
        """
        Adds the precision and recall.

        :param precision: The precision. ``precision`` is a dict with the following format
            `{0: precision for class 0, ...}`.
        :param recall: The recall. ``recall`` is a dict with the following format
            `{0: recall for label 0, ...}`.
        """
        self.explanations = {"precision": precision, "recall": recall}

    def get_explanations(self):
        """
        Gets the precision recall curves.

        :return: A Dict for the precision recall curves, i.e.,
            `{"precision": the precision, "recall": the recall}`.
            Each of "precision" and "recall" is a dict with the following format `{0: precision or
            recall for class 0, ...}`.
        """
        return self.explanations

    def plot(self, class_names=None, linewidth=2, **kwargs):
        """
        Plots the precision recall curves.

        :param class_names: A list of the class names indexed by the labels.
        :param linewidth: The line width.
        :return: A matplotlib figure plotting the precision recall curves.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        precision = self.explanations["precision"]
        recall = self.explanations["recall"]
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig = plt.figure()
        for i in range(len(precision)):
            label = class_names[i] if class_names is not None else i
            plt.plot(
                recall[i], precision[i],
                color=colors[i % len(colors)],
                linewidth=linewidth,
                label="Class {}".format(label),
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curves")
        plt.legend(loc="lower left")
        plt.grid()
        return fig

    def _plotly_figure(self, class_names=None, linewidth=2, **kwargs):
        import plotly.graph_objects as go

        precision = self.explanations["precision"]
        recall = self.explanations["recall"]

        fig = go.Figure()
        for i in range(len(precision)):
            label = class_names[i] if class_names is not None else i
            fig.add_trace(go.Scatter(
                x=recall[i],
                y=precision[i],
                name="Class {}".format(label),
                line=dict(width=linewidth),
            ))
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            title={"text": "Precision Recall Curves"}
        )
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the precision recall curves using Dash.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure plotting the precision recall curves.
        """
        return DashFigure(self._plotly_figure(class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the precision recall curves in IPython.

        :param class_names: A list of the class names indexed by the labels.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        e = d["explanations"]
        for metric in ["precision", "recall"]:
            e[metric] = {int(key): value for key, value in e[metric].items()}
        exp = PrecisionRecallExplanation()
        exp.explanations = e
        return exp
