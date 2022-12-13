#
# Copyright (c) 2022 salesforce.com, inc.
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

    def plot(self, **kwargs):
        pass

    def _plotly_figure(self, **kwargs):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the bias analysis results.

        :return: A plotly dash figure plotting bias analysis results.
        """
        return DashFigure(self._plotly_figure(class_names=class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Shows the bias analysis results in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names=class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = BiasExplanation(mode=d["mode"])
        exp.explanations = d["explanations"]
        return exp
