#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Explanations for linear models.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure


class LinearExplanation(ExplanationBase):
    """
    The class for explanation results for linear models. The results are stored
    in a dict with the following format: `{"coefficients": the linear coefficients,
    "scores": the feature importance scores of a batch of instances, "outputs": the
    predicted values of a batch of instances}`. The value of "scores" is a dict whose
    keys are feature names and values are feature importance scores.
    """

    def __init__(self, mode):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.mode = mode
        self.explanations = {}

    def __repr__(self):
        return repr(self.explanations)

    def add(self, coefficients, importance_scores, outputs):
        """
        Adds the generated explanation corresponding to one instance.

        :param coefficients: Linear coefficients.
        :param importance_scores: Feature importance scores, e.g., `feature value * coefficient`.
        :param outputs: The predictions.
        """
        self.explanations["coefficients"] = coefficients
        self.explanations["scores"] = importance_scores
        self.explanations["outputs"] = outputs

    def get_explanations(self):
        """
        Gets the generated explanations.

        :return: A dict containing the global explanation, i.e., the linear coefficients,
            and the local explanations for all the instances, i.e., feature importance scores,
            with the following format: `{"coefficients": the linear coefficients,
            "scores": the feature importance scores of a batch of instances, "outputs": the
            predicted values of a batch of instances}`. The value of "scores"
            is a dict whose keys are feature names and values are feature importance scores.
        """
        return self.explanations

    def _plot(self, plt, ax, feat_scores, title, font_size=None):
        plt.sca(ax)
        feat_scores = sorted(feat_scores.items(), key=lambda x: abs(x[1]))
        fnames = [f"{self._s(f, max_len=30)}" + " " * 5 for f, s in feat_scores if s != 0.0]
        scores = [s for f, s in feat_scores if s != 0.0]
        colors = ["green" if x > 0 else "red" for x in scores]
        positions = np.arange(len(scores)) + 0.5
        plt.barh(positions, scores, align="center", color=colors)
        ax.yaxis.set_ticks_position("right")
        plt.yticks(positions, fnames, ha="right", fontsize=font_size)
        plt.title(title)

    def plot(self, plot_coefficients=False, class_names=None, max_num_subplots=9, font_size=None, **kwargs):
        """
        Returns a list of matplotlib figures showing the global and local explanations.

        :param plot_coefficients: Whether to plot linear coefficients.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param max_num_subplots: The maximum number of subplots in the figure.
        :param font_size: The font size of ticks.
        :return: A list of matplotlib figures plotting linear coefficients and
            feature importance scores.
        """
        import matplotlib.pyplot as plt

        figures = []
        if plot_coefficients:
            fig, ax = plt.subplots(1, 1)
            coefficients = self.explanations["coefficients"]
            self._plot(plt, ax, coefficients, "coefficients", font_size)
            figures.append(fig)

        if self.explanations["scores"] is not None:
            scores = self.explanations["scores"]
            outputs = self.explanations["outputs"]
            num_figures = len(scores)
            if max_num_subplots is not None:
                num_figures = min(num_figures, max_num_subplots)
            num_rows = int(np.round(np.sqrt(num_figures)))
            num_cols = int(np.ceil(num_figures / num_rows))

            fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)
            for i, score in enumerate(scores):
                row, col = divmod(i, num_cols)
                if self.mode == "classification":
                    output = int(outputs[i]) if class_names is None else class_names[int(outputs[i])]
                    title = f"Explanation for class {output}"
                else:
                    title = f"Explanation for output {round(outputs[i], 4)}"
                self._plot(plt, axes[row, col], score, f"Instance {i}: {title}", font_size)
            figures.append(fig)
        return figures

    def _plotly_figure(self, index, class_names=None, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        if self.explanations["scores"] is not None:
            assert index < len(self.explanations["scores"]), "`index` is out of bound."
            output = self.explanations["outputs"][index]
            if self.mode == "classification":
                class_name = int(output) if class_names is None else class_names[int(output)]
                title = f"Instance {index}: Class {class_name}"
            else:
                title = f"Instance {index}: Output {output}"
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Coefficients", title])
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=["Coefficients"])

        # Plot coefficients
        coefficients = self.explanations["coefficients"]
        feat_scores = sorted(coefficients.items(), key=lambda x: abs(x[1]))
        fnames = [self._s(f, max_len=30) for f, s in feat_scores if s != 0.0]
        scores = [s for f, s in feat_scores if s != 0.0]
        coefficient_fig = px.bar(
            y=fnames,
            x=scores,
            orientation="h",
            color=[s > 0 for s in scores],
            labels={"color": "Positive", "x": "Coefficient", "y": "Features"},
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        fig.add_trace(coefficient_fig.data[0], row=1, col=1)

        # Plot feature importance scores
        if self.explanations["scores"] is not None:
            scores = self.explanations["scores"]
            feat_scores = sorted(scores[index].items(), key=lambda x: abs(x[1]))
            fnames = [self._s(f, max_len=30) for f, s in feat_scores if s != 0.0]
            scores = [s for f, s in feat_scores if s != 0.0]
            score_fig = px.bar(
                y=fnames,
                x=scores,
                orientation="h",
                color=[s > 0 for s in scores],
                labels={"color": "Positive", "x": "Importance scores", "y": "Features"},
                color_discrete_map={True: "#008B8B", False: "#DC143C"},
            )
            fig.add_trace(score_fig.data[0], row=1, col=2)
        return fig

    def plotly_plot(self, index=0, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the linear coefficients and feature
        importance scores for one specific instance.

        :param index: The index of the instance which cannot be None, e.g.,
            it will plot the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting linear coefficients and feature importance scores.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, **kwargs))

    def ipython_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots the linear coefficients and feature importance scores in IPython.

        :param index: The index of the instance which cannot be None, e.g.,
            it will plot the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = LinearExplanation(mode=d["mode"])
        e = d["explanations"]
        e["outputs"] = np.array(e["outputs"])
        exp.explanations = e
        return exp
