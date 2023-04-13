#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature importance explanations.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure


class FeatureImportance(ExplanationBase):
    """
    The class for feature importance explanations. It uses a list to store
    the feature importance explanations of the input instances. Each item in the list
    is a dict with the following format `{"instance": the input instance, "features": a list of feature names,
    "values": a list of feature values, "scores": a list of feature importance scores}`.
    If the task is `classification`, the dict has an additional entry `{"target_label":
    the predicted label of the input instance}`.
    """

    def __init__(self, mode, explanations=None):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        :param explanations: The explanation results for initializing ``FeatureImportance``,
            which is optional.
        """
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return FeatureImportance(mode=self.mode, explanations=[self.explanations[i]])

    def add(self, instance, target_label, feature_names, feature_values, importance_scores, sort=False, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param instance: The instance to explain.
        :param target_label: The label to explain, which is ignored for regression.
        :param feature_names: The list of the feature column names.
        :param feature_values: The list of the feature values.
        :param importance_scores: The list of the feature importance scores.
        :param sort: `True` if the features are sorted based on the importance scores.
        """
        scores = list(zip(feature_names, feature_values, importance_scores))
        if sort:
            scores = sorted(scores, key=lambda x: abs(x[-1]), reverse=True)
        e = {
            "instance": instance,
            "features": [s[0] for s in scores],
            "values": [s[1] for s in scores],
            "scores": [s[2] for s in scores],
        }
        e.update(kwargs)
        if self.mode == "classification":
            e["target_label"] = target_label
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``FeatureImportance``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"instance": the input instance,
            "features": a list of feature names, "values": a list of feature values,
            "scores": a list of feature importance scores}`. If the task is `classification`,
            the dict has an additional entry `{"target_label": the predicted label
            of the input instance}`.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, class_names=None, num_features=20, max_num_subplots=4, **kwargs):
        """
        Plots feature importance scores.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it shows a figure with ``max_num_subplots`` subplots
            where each subplot plots the feature importance scores for one instance.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        :param max_num_subplots: The maximum number of subplots in the figure.
        :return: A matplotlib figure plotting feature importance scores.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if max_num_subplots is not None:
            indices = indices[:max_num_subplots]

        num_rows = int(np.round(np.sqrt(len(indices))))
        num_cols = int(np.ceil(len(indices) / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i, index in enumerate(indices):
            exp = explanations[index]
            feat_scores = sorted(
                list(zip([f"{self._s(f)} = {self._s(v)}    "
                          for f, v in zip(exp["features"], exp["values"])], exp["scores"])),
                key=lambda x: abs(x[1]),
            )
            if num_features is not None:
                feat_scores = feat_scores[-num_features:]
            # Ignore those features with importance_score = 0
            fnames = [f for f, s in feat_scores if s != 0.0]
            scores = [s for f, s in feat_scores if s != 0.0]
            colors = ["green" if x > 0 else "red" for x in scores]
            positions = np.arange(len(scores)) + 0.5

            row, col = divmod(i, num_cols)
            plt.sca(axes[row, col])
            plt.barh(positions, scores, align="center", color=colors)
            axes[row, col].yaxis.set_ticks_position("right")
            plt.yticks(positions, fnames, ha="right")
            if self.mode == "classification":
                class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
                plt.title(f"Instance {index}: Class {class_name}")
            else:
                plt.title(f"Instance {index}")
        return fig

    def _plotly_figure(self, index, class_names=None, num_features=20, **kwargs):
        import plotly.express as px

        exp = self.explanations[index]
        if self.mode == "classification":
            class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
            title = f"Label: Class {class_name}"
        else:
            title = ""

        feat_scores = sorted(
            list(zip([f"{self._s(f)} = {self._s(v)}"
                      for f, v in zip(exp["features"], exp["values"])], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores if s != 0.0]
        scores = [s for f, s in feat_scores if s != 0.0]

        fig = px.bar(
            y=fnames,
            x=scores,
            orientation="h",
            color=[s > 0 for s in scores],
            labels={"color": "Positive", "x": "Importance scores", "y": "Features"},
            title=title,
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        return fig

    def plotly_plot(self, index=0, class_names=None, num_features=20, **kwargs):
        """
        Plots feature importance scores for one specific instance using Dash.

        :param index: The index of an explanation result stored in ``FeatureImportance``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        :return: A plotly dash figure plotting feature importance scores.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, num_features=num_features, **kwargs))

    def ipython_plot(self, index=0, class_names=None, num_features=20, **kwargs):
        """
        Plots the feature importance scores in IPython.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, num_features=num_features, **kwargs))

    @classmethod
    def from_dict(cls, d):
        import pandas as pd
        explanations = []
        for e in d["explanations"]:
            e["instance"] = pd.DataFrame.from_dict(e["instance"])
            explanations.append(e)
        return FeatureImportance(mode=d["mode"], explanations=explanations)


class GlobalFeatureImportance(ExplanationBase):
    """
    The class for global feature importance scores. It uses a dict to store
    the feature importance scores with the following format `{"features": a list of feature names,
    "scores": a list of feature importance scores}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, feature_names, importance_scores, sort=False, **kwargs):
        """
        Adds the generated feature importance scores.

        :param feature_names: The list of the feature column names.
        :param importance_scores: The list of the feature importance scores.
        :param sort: `True` if the features are sorted based on the importance scores.
        """
        scores = list(zip(feature_names, importance_scores))
        if sort:
            scores = sorted(scores, key=lambda x: abs(x[-1]), reverse=True)
        self.explanations = {
            "features": [s[0] for s in scores],
            "scores": [s[1] for s in scores],
        }

    def get_explanations(self):
        """
        Gets the generated explanations.

        :return: The feature importance scores.
            The returned dict has the following format: `{"features": a list of feature names,
            "scores": a list of feature importance scores}`.
        :rtype: Dict
        """
        return self.explanations

    def plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots feature importance scores.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        :return: A matplotlib figure plotting feature importance scores.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1)
        exp = self.get_explanations()
        feat_scores = sorted(
            list(zip([f"{self._s(f) if truncate_long_features else f}    " for f in exp["features"]], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores]
        scores = [s for f, s in feat_scores]
        colors = ["green" if x > 0 else "red" for x in scores]
        positions = np.arange(len(scores)) + 0.5

        plt.sca(axes)
        plt.barh(positions, scores, align="center", color=colors)
        axes.yaxis.set_ticks_position("right")
        plt.yticks(positions, fnames, ha="right")
        plt.title(f"Global Feature Importance")
        return fig

    def _plotly_figure(self, num_features=20, truncate_long_features=True, **kwargs):
        import plotly.express as px

        exp = self.explanations
        title = f"Global Feature Importance"
        feat_scores = sorted(
            list(zip([f"{self._s(f) if truncate_long_features else f}" for f in exp["features"]], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores]
        scores = [s for f, s in feat_scores]

        fig = px.bar(
            y=fnames,
            x=scores,
            orientation="h",
            labels={"x": "Importance scores", "y": "Features"},
            title=title,
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        return fig

    def plotly_plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots feature importance scores for one specific instance using Dash.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        :return: A plotly dash figure plotting feature importance scores.
        """
        return DashFigure(self._plotly_figure(num_features=num_features,
                                              truncate_long_features=truncate_long_features,
                                              **kwargs))

    def ipython_plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots the feature importance scores in IPython.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        """
        import plotly
        plotly.offline.iplot(self._plotly_figure(num_features=num_features,
                                                 truncate_long_features=truncate_long_features,
                                                 **kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = GlobalFeatureImportance()
        exp.explanations = d["explanations"]
        return exp
