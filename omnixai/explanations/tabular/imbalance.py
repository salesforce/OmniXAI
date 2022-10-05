#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature imbalance plots.
"""
import numpy as np
import pandas as pd
from omnixai.explanations.base import ExplanationBase, DashFigure


class ImbalanceExplanation(ExplanationBase):
    """
    The class for feature imbalance plots. It uses a list to store the feature values
    and their counts (numbers of appearances) in each class. Each item in the list
    is a dict, i.e., `{"feature": feature value, "count": {class label 1: count 1,
    class label 2: count 2, ...}}`. If there are no class labels, the dict will be
    `{"feature": feature value, "count": count}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def add(self, feature, count):
        """
        Adds the count for a cross-feature.

        :param feature: A cross-feature (a list of feature values).
        :param count: The number of appearances.
        """
        self.explanations.append({"feature": feature, "count": count})

    def get_explanations(self):
        """
        Gets the imbalance analysis results.

        :return: A Dict storing the counts for all the cross-features,
            i.e., `{"feature": feature value, "count": {class label 1: count 1,
            class label 2: count 2, ...}}`. If there are no class labels, the dict will be
            `{"feature": feature value, "count": count}`.
        """
        return self.explanations

    def plot(self, **kwargs):
        """
        Shows the imbalance plot.

        :return: A matplotlib figure plotting the feature counts.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1)
        positions = np.arange(len(self.explanations)) + 0.5
        fnames = [", ".join(str(self._s(s)) for s in e["feature"]) + "    " for e in self.explanations]

        if not isinstance(self.explanations[0]["count"], dict):
            counts = [e["count"] for e in self.explanations]
            plt.barh(positions, counts, align="center")
        else:
            bottom = [0] * len(self.explanations)
            labels = sorted(self.explanations[0]["count"].keys())
            for label in labels:
                counts = [e["count"][label] for e in self.explanations]
                plt.barh(positions, counts, align="center", left=bottom, label=str(label))
                bottom = counts
            plt.legend(loc="upper left")

        axes.yaxis.set_ticks_position("right")
        plt.yticks(positions, fnames, ha="right")
        plt.title("Imbalance Plot")
        return fig

    def _plotly_figure(self, **kwargs):
        import plotly.express as px

        fnames = [", ".join(str(self._s(s)) for s in e["feature"]) for e in self.explanations]
        if not isinstance(self.explanations[0]["count"], dict):
            counts = [e["count"] for e in self.explanations]
            fig = px.bar(
                y=fnames, x=counts, orientation="h",
                labels={"x": "Counts", "y": "Features"}, title="Imbalance Plot"
            )
        else:
            df = pd.DataFrame(fnames, columns=["Features"])
            labels = sorted(self.explanations[0]["count"].keys())
            for label in labels:
                df[str(label)] = [e["count"][label] for e in self.explanations]
            fig = px.bar(df, y="Features", x=[str(label) for label in labels],
                         orientation="h", title="Imbalance Plot")
        return fig

    def plotly_plot(self, **kwargs):
        """
        Shows the imbalance plot.

        :return: A plotly dash figure plotting the feature counts.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Shows the imbalance plot in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = ImbalanceExplanation()
        exp.explanations = d["explanations"]
        return exp
