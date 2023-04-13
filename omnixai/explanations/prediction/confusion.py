#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The confusion matrix.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure


class ConfusionMatrixExplanation(ExplanationBase):
    """
    The class for the confusion matrix.
    """

    def __init__(self, confusion_matrix):
        """
        :param confusion_matrix: The confusion matrix.
        """
        super().__init__()
        self.confusion_matrix = confusion_matrix

    def get_explanations(self):
        """
        Get the confusion matrix.

        :return: The confusion matrix.
        :rtype: np.ndarray
        """
        return self.confusion_matrix

    def plot(self, class_names=None, fontsize=15, **kwargs):
        """
        Plots the confusion matrix.

        :param class_names: A list of the class names indexed by the labels.
        :param fontsize: The font size for title, xlabel and ylabel.
        :return: A matplotlib figure plotting the confusion matrix.
        """
        import matplotlib.pyplot as plt
        if class_names is None:
            labels = [str(i) for i in range(self.confusion_matrix.shape[0])]
        else:
            labels = [class_names[i] for i in range(self.confusion_matrix.shape[0])]

        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix, cmap=plt.cm.Blues, alpha=0.8)
        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                ax.text(x=j, y=i, s=self.confusion_matrix[i, j], va="center", ha="center")
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)

        fig.colorbar(cax)
        plt.xlabel("Predictions", fontsize=fontsize)
        plt.ylabel("GroundTruth", fontsize=fontsize)
        plt.title("Confusion Matrix", fontsize=fontsize)
        return fig

    def _plotly_figure(self, class_names=None, **kwargs):
        import plotly.graph_objects as go
        if class_names is None:
            labels = [str(i) for i in range(self.confusion_matrix.shape[0])]
        else:
            labels = [class_names[i] for i in range(self.confusion_matrix.shape[0])]

        annotations = []
        total_counts = np.sum(self.confusion_matrix)
        for i, row in enumerate(self.confusion_matrix):
            for j, value in enumerate(row):
                annotations.append({
                    "x": labels[j],
                    "y": labels[i],
                    "font": {"color": "black"},
                    "text": "{} ({:.1f}%)".format(value, value / total_counts * 100),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                })
        layout = {
            "xaxis": {"title": "Predictions"},
            "yaxis": {"title": "GroundTruth"},
            "annotations": annotations,
            "title": {"text": "Confusion Matrix"}
        }
        fig = go.Figure(
            data=go.Heatmap(
                z=self.confusion_matrix,
                y=labels,
                x=labels,
                colorscale="blues"
            ),
            layout=layout
        )
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Plots the confusion matrix using Dash.

        :param class_names: A list of the class names indexed by the labels.
        :return: A plotly dash figure plotting the confusion matrix.
        """
        return DashFigure(self._plotly_figure(class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Plots the confusion matrix in IPython.

        :param class_names: A list of the class names indexed by the labels.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        return ConfusionMatrixExplanation(confusion_matrix=np.array(d["confusion_matrix"]))
