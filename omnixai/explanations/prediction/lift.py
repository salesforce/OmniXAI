#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The lift curves.
"""
import numpy as np
from typing import Dict
from ..base import ExplanationBase, DashFigure


class LiftCurveExplanation(ExplanationBase):
    """
    The class for plotting the lift curves. It stores the lift curves for each class
    in a dict, i.e., `{"gains": the lift gains, "percentages": the percentage of samples}`.
    The value of "gains" is a dict with the following format `{0: the lift gains for class 0, ...}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, gains: Dict, percentages: np.ndarray):
        """
        Adds the false positive rates, true positive rates and AUCs.

        :param gains: The lift gains. ``gains`` is a dict with the following format
            `{0: the lift gains for class 0, ...}`.
        :param percentages: The percentage of samples.
        """
        self.explanations = {"gains": gains, "percentages": percentages}

    def get_explanations(self):
        """
        Gets the lift gains.

        :return: A Dict for the lift gains, i.e.,
            `{"gains": the lift gains, "percentages": the percentage of samples}`.
            The value of "gains" is a dict with the following format `{0: the lift gains
            for class 0, ...}`.
        """
        return self.explanations

    def plot(self, class_names=None, linewidth=2, **kwargs):
        """
        Plots the lift curves.

        :param class_names: A list of the class names indexed by the labels.
        :param linewidth: The line width.
        :return: A matplotlib figure plotting the lift curves.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        percentages = self.explanations["percentages"]
        class_gains = self.explanations["gains"]
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig = plt.figure()
        for i in range(len(class_gains)):
            label = class_names[i] if class_names is not None else i
            plt.plot(
                percentages,
                class_gains[i],
                color=colors[i % len(colors)],
                linewidth=linewidth,
                label="Class {}".format(label),
            )

        plt.plot([0, 1], [1, 1], "k--", linewidth=linewidth, label="Baseline")
        plt.xlabel("Percentage of samples")
        plt.ylabel("Lift")
        plt.title("Lift Curves")
        plt.legend(loc="upper right")
        plt.grid()
        return fig

    def _plotly_figure(self, class_names=None, linewidth=2, **kwargs):
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors

        percentages = self.explanations["percentages"]
        class_gains = self.explanations["gains"]
        colors = list(mcolors.TABLEAU_COLORS.values())

        fig = go.Figure()
        for i in range(len(class_gains)):
            color = colors[i % len(colors)]
            label = class_names[i] if class_names is not None else i
            fig.add_trace(go.Scatter(
                x=percentages,
                y=class_gains[i],
                name="Class {}".format(label),
                line=dict(color=color, width=linewidth),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[1, 1],
            line=dict(color="black", dash="dash", width=linewidth),
            name="Baseline"
        ))
        fig.update_layout(
            xaxis_title="Percentage of samples",
            yaxis_title="Lift",
            title={"text": "Lift Curves"}
        )
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the lift curves using Dash.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure plotting the lift curves.
        """
        return DashFigure(self._plotly_figure(class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the lift curves in IPython.

        :param class_names: A list of the class names indexed by the labels.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        e = d["explanations"]
        e["gains"] = {int(key): np.array(value) for key, value in e["gains"].items()}
        e["percentages"] = np.array(e["percentages"])
        exp = LiftCurveExplanation()
        exp.explanations = e
        return exp
