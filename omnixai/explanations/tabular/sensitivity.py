#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Morris sensitivity analysis.
"""
from collections import defaultdict
from ..base import ExplanationBase, DashFigure


class SensitivityExplanation(ExplanationBase):
    """
    The class for sensitivity analysis results. The results are stored in
    a dict with the following format: `{feature_name: {"mu": Morris mu,
    "mu_star": Morris mu_star, "sigma": Morris sigma, "mu_star_conf":
    Morris mu_star_conf}}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = defaultdict(dict)

    def add(self, feature_name, mu, mu_star, sigma, mu_star_conf):
        """
        Adds the sensitivity analysis result of a specific feature.

        :param feature_name: The feature column name.
        :param mu: `mu`.
        :param mu_star: `mu_star`.
        :param sigma: `sigma`.
        :param mu_star_conf: `mu_star_conf`.
        """
        self.explanations[feature_name] = {"mu": mu, "mu_star": mu_star, "sigma": sigma, "mu_star_conf": mu_star_conf}

    def get_explanations(self):
        """
        Gets the Morris sensitivity analysis results.

        :return: A dict containing the raw values of the partial
            dependence function for all the features with the following format:
            `{feature_name: {"mu": Morris mu,
            "mu_star": Morris mu_star, "sigma": Morris sigma, "mu_star_conf":
            Morris mu_star_conf}}`.
        :rtype: Dict
        """
        return self.explanations

    def plot(self, **kwargs):
        """
        Returns a matplotlib figure showing the sensitivity analysis results.

        :return: A matplotlib figure.
        """
        import matplotlib.pyplot as plt

        features = list(self.explanations.keys())
        results = {
            "mu": [self.explanations[f]["mu"] for f in features],
            "mu_star": [self.explanations[f]["mu_star"] for f in features],
            "sigma": [self.explanations[f]["sigma"] for f in features],
            "mu_star_conf": [self.explanations[f]["mu_star_conf"] for f in features],
        }
        fig, axes = plt.subplots(2, 2, squeeze=False)
        for i, name in enumerate(["mu", "mu_star", "sigma", "mu_star_conf"]):
            plt.sca(axes[i // 2, i % 2])
            plt.barh([self._s(f, max_len=10) for f in features], results[name])
            plt.ylabel(name)
            plt.xlabel("Sensitivity")
            plt.grid()
        return fig

    def _plotly_figure(self, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        features = list(self.explanations.keys())
        results = {
            "mu": [self.explanations[f]["mu"] for f in features],
            "mu_star": [self.explanations[f]["mu_star"] for f in features],
            "sigma": [self.explanations[f]["sigma"] for f in features],
            "mu_star_conf": [self.explanations[f]["mu_star_conf"] for f in features],
        }
        fig = make_subplots(rows=2, cols=2, subplot_titles=["mu", "mu_star", "sigma", "mu_star_conf"])
        for i, name in enumerate(["mu", "mu_star", "sigma", "mu_star_conf"]):
            r, c = i // 2, i % 2
            _fig = px.bar(y=[self._s(f, max_len=10) for f in features], x=results[name],
                          orientation="h", labels={"x": "Sensitivity", "y": name})
            fig.add_trace(_fig.data[0], row=r + 1, col=c + 1)
        return fig

    def plotly_plot(self, **kwargs):
        """
        Returns a plotly dash figure showing sensitivity analysis results.
        """
        return DashFigure(self._plotly_figure(**kwargs))

    def ipython_plot(self, **kwargs):
        """
        Plots sensitivity analysis results in IPython.
        """
        import plotly

        plotly.offline.iplot(self._plotly_figure(**kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = SensitivityExplanation()
        exp.explanations = d["explanations"]
        return exp
