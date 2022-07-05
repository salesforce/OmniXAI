#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The metrics for classification and regression.
"""
from typing import Dict
from ..base import ExplanationBase, DashFigure


class MetricExplanation(ExplanationBase):
    """
    The class for plotting metrics. It stores the metrics
    in a dict, i.e., `{"precision": precision, "recall": recall, ...}`.
    """

    def __init__(self, metrics: Dict):
        super().__init__()
        self.metrics = metrics

    def get_explanations(self):
        """
        Gets the precision recall curves.

        :return: A Dict for the metrics, i.e., `{"precision": precision, "recall": recall, ...}`.
        """
        return self.metrics

    def plot(self, **kwargs):
        """
        Plots the metrics.

        :return: A matplotlib figure plotting the metrics.
        """
        import matplotlib.pyplot as plt

    def _plotly_figure(self, **kwargs):
        pass

    def plotly_plot(self, **kwargs):
        """
        Plots the metrics using Dash.

        :return: A plotly dash figure plotting the metrics.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the metrics in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))
