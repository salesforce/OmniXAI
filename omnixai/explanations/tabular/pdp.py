#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Partial dependence plots.
"""
import warnings
import numpy as np
from ..base import ExplanationBase, DashFigure
from collections import defaultdict


class PDPExplanation(ExplanationBase):
    """
    The class for PDP explanation results. The PDP explanation results are stored in a dict.
    The key in the dict is "global" indicating PDP is a global explanation method.
    The value in the dict is another dict with the following format:
    `{feature_name: {"values": the PDP grid values, "scores": the average PDP scores,
    "stds": the standard deviation of the PDP scores}}`.
    """

    def __init__(self, mode):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        self.mode = mode
        self.explanations = defaultdict(dict)

    def __repr__(self):
        return repr(self.explanations)

    def add(self, index, feature_name, values, pdp_mean, pdp_std):
        """
        Adds the raw values of the partial dependence function
        corresponding to one specific feature.

        :param index: `index = global` for global explanations.
        :param feature_name: The feature column name.
        :param values: The grid values when generating PDP.
        :param pdp_mean: The average PDP scores corresponding to the values.
        :param pdp_std: The standard deviation of the PDP scores corresponding to the values.
        """
        self.explanations[index][feature_name] = {"values": values, "scores": pdp_mean, "stds": pdp_std}

    def get_explanations(self):
        """
        Gets the raw values of the partial dependence function.

        :return: A dict containing the raw values of the partial
            dependence function for all the features with the following format:
            `{feature_name: {"values": the PDP grid values, "scores": the average PDP scores,
            "stds": the standard deviation of the PDP scores}}`.
        """
        keys = sorted(self.explanations.keys())
        explanations = [self.explanations[k] for k in keys]
        return explanations[0] if len(explanations) == 1 else explanations

    def plot(self, features, class_names=None, plot_std=False, **kwargs):
        """
        Returns a matplotlib figure showing the PDP explanations.

        :param features: A list of features to be shown in the figure.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param plot_std: Whether to plot the standard deviation of the PDP scores.
        :return: A matplotlib figure plotting PDP explanations.
        """
        import matplotlib.pyplot as plt

        assert features is not None, "Please choose the features to plot."
        if isinstance(features, str):
            features = [features]

        explanations = self.get_explanations()
        if isinstance(explanations, dict):
            explanations = [explanations]
        if len(explanations) == 0:
            return None
        assert all(
            f in explanations[0] for f in features
        ), "Some of the specified features are not included in the explanations."
        if len(explanations) > 5:
            warnings.warn(
                f"There are too many instances ({len(explanations)} > 5), "
                f"so only the first 5 instances are plotted."
            )
            explanations = explanations[:5]

        figures = []
        for exps in explanations:
            num_rows = int(np.round(np.sqrt(len(features))))
            num_cols = int(np.ceil(len(features) / num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

            for i, feature in enumerate(features):
                exp = exps[feature]
                row, col = divmod(i, num_cols)
                plt.sca(axes[row, col])
                # Plot partial dependence
                if plot_std:
                    plt.errorbar(exp["values"], exp["scores"], exp["stds"])
                else:
                    plt.plot(exp["values"], exp["scores"])
                # Rotate xticks if it is a categorical feature
                if isinstance(exp["values"][0], str):
                    plt.xticks(rotation=45)
                plt.ylabel("Partial dependence")
                plt.title(feature)
                if class_names is not None:
                    plt.legend(class_names)
                plt.grid()
            figures.append(fig)

        return figures

    def _plotly_figure(self, index, features, class_names=None, **kwargs):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        assert features is not None, "Please choose the features to plot."
        if isinstance(features, str):
            features = [features]

        explanations = self.get_explanations()
        if isinstance(explanations, dict):
            explanations = [explanations]
        if len(explanations) == 0:
            return None
        if "global" not in self.explanations:
            assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
            exp = explanations[index]
        else:
            exp = explanations[0]

        num_cols = 2
        num_rows = int(np.ceil(len(features) / num_cols))
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=features)
        for i, feature in enumerate(features):
            e = exp[feature]
            row, col = divmod(i, num_cols)
            if self.mode == "classification":
                for k in range(e["scores"].shape[1]):
                    label = class_names[k] if class_names is not None else f"Label {k}"
                    fig.add_trace(go.Scatter(x=e["values"], y=e["scores"][:, k], name=label), row=row + 1, col=col + 1)
            else:
                fig.add_trace(go.Scatter(x=e["values"], y=e["scores"], name="Value"), row=row + 1, col=col + 1)
        fig.update_layout(height=200 * num_rows)
        return fig

    def plotly_plot(self, features, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the PDP explanations.

        :param features: A list of features to be shown in the figure.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting PDP explanations.
        """
        if "index" in kwargs:
            kwargs.pop("index")
        return DashFigure(self._plotly_figure(index=None, features=features, class_names=class_names, **kwargs))

    def ipython_plot(self, features, class_names=None, **kwargs):
        """
        Shows the partial dependence plots in IPython.

        :param features: A list of features to be shown in the figure.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        if "index" in kwargs:
            kwargs.pop("index")
        plotly.offline.iplot(self._plotly_figure(index=None, features=features, class_names=class_names, **kwargs))
