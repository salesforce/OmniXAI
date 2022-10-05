#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature correlation analysis.
"""
import numpy as np
from omnixai.explanations.base import ExplanationBase, DashFigure


class CorrelationExplanation(ExplanationBase):
    """
    The class for correlation analysis results. The results are stored in a Dict, i.e.,
    `{"features": a list of feature names, "correlation": the correlation matrix}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, features, correlation):
        """
        Adds the count for a cross-feature.

        :param features: The feature names.
        :param correlation: The correlation matrix.
        """
        self.explanations = {"features": list(features), "correlation": correlation}

    def get_explanations(self):
        """
        Gets the correlation matrix.

        :return: A Dict for the feature names and the correlation matrix., i.e.,
            `{"features": a list of feature names, "correlation": the correlation matrix}`.
        """
        return self.explanations

    def plot(self, **kwargs):
        """
        Plots the correlation matrix.

        :return: A matplotlib figure plotting the correlation matrix.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        mat = ax.matshow(self.explanations["correlation"], cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(mat)

        features = self.explanations["features"]
        features = [self._s(f) for f in features]
        ticks = np.arange(len(features))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(features)
        ax.set_yticklabels(features)
        plt.xticks(rotation=45)
        return fig

    def _plotly_figure(self, **kwargs):
        import plotly.express as px

        features = self.explanations["features"]
        features = [self._s(f) for f in features]
        fig = px.imshow(self.explanations["correlation"], x=features, y=features, color_continuous_scale="RdBu_r")
        return fig

    def plotly_plot(self, **kwargs):
        """
        Plots the correlation matrix using Dash.

        :return: A plotly dash figure plotting the correlation matrix.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Plots the correlation matrix in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))

    @classmethod
    def from_dict(cls, d):
        e = d["explanations"]
        e["correlation"] = np.array(e["correlation"])
        exp = CorrelationExplanation()
        exp.explanations = e
        return exp
