#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The cumulative gains.
"""
import numpy as np
from typing import Dict
from ..base import ExplanationBase, DashFigure


class CumulativeGainExplanation(ExplanationBase):
    """
    The class for plotting the cumulative gains. It stores the cumulative gains for each class
    in a dict, i.e., `{"gains": the cumulative gains, "percentages": the percentage of samples,
    "num_samples": the number of samples in each class}`.
    The value of "gains" is a dict with the following format `{0: the cumulative gains for class 0, ...}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, gains: Dict, percentages: np.ndarray, num_samples: Dict):
        """
        Adds the false positive rates, true positive rates and AUCs.

        :param gains: The cumulative gains. ``gains`` is a dict with the following format
            `{0: the cumulative gains for class 0, ...}`.
        :param percentages: The percentage of samples.
        :param num_samples: The number of samples in each class.
        """
        self.explanations = {"gains": gains, "percentages": percentages, "num_samples": num_samples}

    def get_explanations(self):
        """
        Gets the cumulative gains.

        :return: A Dict for the cumulative gains, i.e.,
            `{"gains": the cumulative gains, "percentages": the percentage of samples,
            "num_samples": the number of samples in each class}`.
            The value of "gains" is a dict with the following format `{0: the cumulative gains
            for class 0, ...}`.
        """
        return self.explanations

    def plot(self, class_names=None, linewidth=2, **kwargs):
        """
        Plots the cumulative gains.

        :param class_names: A list of the class names indexed by the labels.
        :param linewidth: The line width.
        :return: A matplotlib figure plotting the cumulative gains.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        percentages = self.explanations["percentages"]
        class_gains = self.explanations["gains"]
        num_samples = self.explanations["num_samples"]
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
            plt.plot(
                [0, num_samples[i] / len(percentages), 1],
                [0, 1, 1],
                color=colors[i % len(colors)],
                linestyle="dashed",
                linewidth=linewidth,
                label="Class {} Best".format(label)
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=linewidth, label="Baseline")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Percentage of samples")
        plt.ylabel("Gains")
        plt.title("Cumulative Gain Curves")
        plt.legend(loc="lower right")
        plt.grid()
        return fig

    def _plotly_figure(self, class_names=None, linewidth=2, **kwargs):
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors

        percentages = self.explanations["percentages"]
        class_gains = self.explanations["gains"]
        num_samples = self.explanations["num_samples"]
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
                x=[0, num_samples[i] / len(percentages), 1],
                y=[0, 1, 1],
                name="Class {} Best".format(label),
                line=dict(color=color, width=linewidth, dash="dash"),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(color="black", dash="dash", width=linewidth),
            name="Baseline"
        ))
        fig.update_layout(
            xaxis_title="Percentage of samples",
            yaxis_title="Gains",
            title={"text": "Cumulative Gain Curves"}
        )
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the cumulative gains using Dash.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure plotting the cumulative gains.
        """
        return DashFigure(self._plotly_figure(class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the cumulative gains in IPython.

        :param class_names: A list of the class names indexed by the labels.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        e = d["explanations"]
        e["gains"] = {int(key): np.array(value) for key, value in e["gains"].items()}
        e["percentages"] = np.array(e["percentages"])
        e["num_samples"] = {int(key): value for key, value in e["num_samples"].items()}
        exp = CumulativeGainExplanation()
        exp.explanations = e
        return exp
