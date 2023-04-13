#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Accumulated local effects plots.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure
from collections import OrderedDict


class ALEExplanation(ExplanationBase):
    """
    The class for ALE explanation results. The ALE explanation results are stored in a dict.
    """

    def __init__(self, mode):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.mode = mode
        self.explanations = OrderedDict()

    def add(self, feature_name, values, scores, sampled_scores=None):
        """
        Adds the raw values of the accumulated local effects
        corresponding to one specific feature.

        :param feature_name: The feature column name.
        :param values: The feature values.
        :param scores: The ALE scores corresponding to the values.
        :param sampled_scores: The ALE scores computed with Monte-Carlo samples.
        """
        self.explanations[feature_name] = \
            {"values": values, "scores": scores, "sampled_scores": sampled_scores}

    def get_explanations(self):
        """
        Gets the accumulated local effects.

        :return: A dict containing the accumulated local effects of all the studied features
            with the following format: `{feature_name: {"values": the feature values, "scores": the ALE scores}}`.
        """
        return self.explanations

    def plot(self, class_names=None, **kwargs):
        """
        Returns a matplotlib figure showing the ALE explanations.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A matplotlib figure plotting ALE explanations.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations()
        features = list(explanations.keys())

        figures = []
        for i, feature in enumerate(features):
            fig, axes = plt.subplots(1, 1, squeeze=False)
            exp = explanations[feature]
            plt.sca(axes[0, 0])
            values = [self._s(v, max_len=15) for v in exp["values"]]
            plt.plot(values, exp["scores"])
            # Rotate xticks if it is a categorical feature
            if isinstance(values[0], str):
                plt.xticks(rotation=45)
            plt.ylabel("Accumulated local effects")
            plt.title(feature)
            if class_names is not None:
                plt.legend(class_names)
            else:
                if self.mode == "classification":
                    plt.legend([f"Class {i}" for i in range(exp["scores"].shape[1])])
                else:
                    plt.legend(["Target"])
            plt.grid()

            if exp["sampled_scores"] is not None:
                for scores in exp["sampled_scores"]:
                    plt.plot(values, scores, color="#808080", alpha=0.1)
            figures.append(fig)
        return figures

    def _plotly_figure(self, class_names=None, **kwargs):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        explanations = self.get_explanations()
        features = list(explanations.keys())
        num_cols = min(2, len(features))
        num_rows = int(np.ceil(len(features) / num_cols))
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=features)
        for i, feature in enumerate(features):
            e = explanations[feature]
            row, col = divmod(i, num_cols)
            values = [self._s(v, max_len=15) for v in e["values"]]
            if self.mode == "classification":
                for k in range(e["scores"].shape[1]):
                    label = class_names[k] if class_names is not None else f"Class {k}"
                    fig.add_trace(go.Scatter(x=values, y=e["scores"][:, k], name=self._s(str(feature), 10),
                                             legendgroup=label,
                                             legendgrouptitle_text=label),
                                  row=row + 1, col=col + 1)
            else:
                fig.add_trace(go.Scatter(x=values, y=e["scores"].flatten(), name=self._s(str(feature), 10),
                                         legendgroup="Target"),
                              row=row + 1, col=col + 1)

            if e["sampled_scores"] is not None:
                for scores in e["sampled_scores"]:
                    if self.mode == "classification":
                        for k in range(scores.shape[1]):
                            label = class_names[k] if class_names is not None else f"Class {k}"
                            fig.add_trace(go.Scatter(x=values, y=scores[:, k],
                                                     opacity=0.1, mode="lines", showlegend=False,
                                                     line=dict(color="#808080"),
                                                     legendgroup=label),
                                          row=row + 1, col=col + 1)
                    else:
                        fig.add_trace(go.Scatter(x=values, y=scores.flatten(),
                                                 opacity=0.1, mode="lines", showlegend=False,
                                                 line=dict(color="#808080"),
                                                 legendgroup="Target"),
                                      row=row + 1, col=col + 1)
        if num_rows > 1:
            fig.update_layout(height=260 * num_rows)
        return fig

    def plotly_plot(self, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the ALE explanations.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting ALE explanations.
        """
        return DashFigure(self._plotly_figure(class_names=class_names, **kwargs))

    def ipython_plot(self, class_names=None, **kwargs):
        """
        Shows the accumulated local effects plots in IPython.

        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(class_names=class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        explanations = {}
        for name, e in d["explanations"].items():
            e["values"] = np.array(e["values"])
            e["scores"] = np.array(e["scores"])
            e["sampled_scores"] = np.array(e["sampled_scores"]) \
                if e["sampled_scores"] is not None else None
            explanations[name] = e
        exp = ALEExplanation(mode=d["mode"])
        exp.explanations = explanations
        return exp
