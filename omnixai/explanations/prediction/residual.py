#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The regression residual plot.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure


class ResidualExplanation(ExplanationBase):
    """
    The class for plotting the regression residuals.
    """

    def __init__(self, predictions, residuals, residual_type):
        super().__init__()
        self.predictions = predictions
        self.residuals = residuals
        self.residual_type = residual_type

    def get_explanations(self):
        """
        Gets the residuals.

        :return: The regression residuals.
        """
        return {"prediction": self.predictions, "residual": self.residuals}

    def plot(self, markersize=5, linewidth=2, **kwargs):
        """
        Plots the regression residuals.

        :param markersize: The marker size.
        :param linewidth: The line width.
        :return: A matplotlib figure plotting the regression residuals.
        """
        import matplotlib.pyplot as plt

        indices = np.argsort(self.predictions)
        predictions = self.predictions[indices]
        residuals = self.residuals[indices]

        fig = plt.figure()
        plt.plot(
            predictions,
            residuals,
            "o",
            markersize=markersize,
            label=f"Residuals ({self.residual_type})",
        )
        if self.residual_type == "ratio":
            plt.plot(
                predictions,
                np.ones(predictions.shape),
                color="orange",
                linewidth=linewidth,
                label="Baseline"
            )
        else:
            plt.plot(
                predictions,
                np.zeros(predictions.shape),
                color="orange",
                linewidth=linewidth,
                label="Baseline"
            )
        plt.xlabel("Prediction")
        plt.ylabel("Residual")
        plt.title("Regression Residuals")
        plt.legend(loc="upper right")
        plt.grid()
        return fig

    def _plotly_figure(self, markersize=5, linewidth=2, **kwargs):
        import plotly.graph_objects as go

        indices = np.argsort(self.predictions)
        predictions = self.predictions[indices]
        residuals = self.residuals[indices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            mode="markers",
            x=predictions,
            y=residuals,
            marker=dict(color="#1f77b4", size=markersize),
            name=f"Residuals ({self.residual_type})"
        ))
        if self.residual_type == "ratio":
            fig.add_trace(go.Scatter(
                x=predictions,
                y=np.ones(predictions.shape),
                line=dict(color="#ff7f0e", width=linewidth),
                name="Baseline"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=predictions,
                y=np.zeros(predictions.shape),
                line=dict(color="#ff7f0e", width=linewidth),
                name="Baseline"
            ))
        fig.update_layout(
            xaxis_title="Prediction",
            yaxis_title="Residual",
            title={"text": "Regression Residuals"}
        )
        return fig

    def plotly_plot(self, **kwargs):
        """
        Plots the regression residuals using Dash.

        :return: A plotly dash figure plotting the regression residuals.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Plots the regression residuals in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))

    @classmethod
    def from_dict(cls, d):
        return ResidualExplanation(
            predictions=np.array(d["predictions"]),
            residuals=np.array(d["residuals"]),
            residual_type=d["residual_type"]
        )
