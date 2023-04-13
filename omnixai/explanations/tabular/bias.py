#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The bias analysis results for tabular data.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure
from collections import OrderedDict


class BiasExplanation(ExplanationBase):
    """
    The class for bias analysis results. The bias analysis results are stored in a dict.
    """

    def __init__(self, mode):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.mode = mode
        self.explanations = OrderedDict()

    def add(self, metric_name, metric_values):
        """
        Adds a new bias metric.

        :param metric_name: The bias metric name.
        :param metric_values: The bias metric values.
        """
        self.explanations[metric_name] = metric_values

    def get_explanations(self):
        """
        Gets the bias analysis results.

        :return: A dict containing the bias analysis results with the following format:
            `{metric_name: {"feature value or threshold": the metric value}, ...}`.
        """
        return self.explanations

    def _rearrange_metrics(self):
        metric_names = list(self.explanations.keys())
        labels = sorted(self.explanations[metric_names[0]].keys())
        label_metrics = [[self.explanations[metric][label] for metric in metric_names]
                         for label in labels]
        return metric_names, labels, label_metrics

    def plot(self, **kwargs):
        """
        Returns a matplotlib figure showing the bias analysis results.

        :return: A matplotlib figure plotting bias analysis results.
        """
        import matplotlib.pyplot as plt

        figures = []
        metric_names, labels, label_metrics = self._rearrange_metrics()
        for i, label in enumerate(labels):
            fig, axes = plt.subplots(1, 1)
            metric_scores = sorted(
                list(zip([f"{f}    " for f in metric_names], label_metrics[i])),
                key=lambda x: abs(x[1]),
            )
            fnames = [f for f, s in metric_scores]
            scores = [s for f, s in metric_scores]
            colors = ["green" if x > 0 else "red" for x in scores]
            positions = np.arange(len(scores)) + 0.5

            plt.sca(axes)
            plt.barh(positions, scores, align="center", color=colors)
            axes.yaxis.set_ticks_position("right")
            plt.yticks(positions, fnames, ha="right")
            plt.title(f"Label: {label}" if self.mode == "classification"
                      else f"Target threshold: {label}")
            plt.grid()
            figures.append(fig)
        return figures

    def _plotly_figure(self, **kwargs):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        metric_names, labels, label_metrics = self._rearrange_metrics()
        num_cols = min(2, len(labels))
        num_rows = int(np.ceil(len(labels) / num_cols))
        if self.mode == "classification":
            subplot_titles = [f"Label: {label}" for label in labels]
        else:
            subplot_titles = [f"Target threshold: {label}" for label in labels]
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)

        for i, label in enumerate(labels):
            row, col = divmod(i, num_cols)
            metric_scores = sorted(
                list(zip(metric_names, label_metrics[i])),
                key=lambda x: abs(x[1]), reverse=True
            )
            fnames = [f for f, s in metric_scores]
            scores = [s for f, s in metric_scores]
            colors = ["#008B8B" if s > 0 else "#DC143C" for s in scores]
            fig.add_trace(
                go.Bar(x=fnames, y=scores, marker_color=colors),
                row=row + 1, col=col + 1
            )
        if num_rows > 1:
            fig.update_layout(height=260 * num_rows)
        return fig

    def plotly_plot(self, **kwargs):
        """
        Returns a plotly dash figure showing the bias analysis results.

        :return: A plotly dash figure plotting bias analysis results.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Shows the bias analysis results in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = BiasExplanation(mode=d["mode"])
        exp.explanations = d["explanations"]
        return exp
