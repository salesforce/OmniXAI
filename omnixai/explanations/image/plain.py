#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Plain image explanations for vision tasks.
"""
from ..base import ExplanationBase, DashFigure


class PlainExplanation(ExplanationBase):
    """
    The class for plain image explanation. It stores a batch of images and the corresponding
    names. Each image represents a plain explanation.
    """

    def __init__(self):
        super().__init__()
        self.explanations = None

    def __repr__(self):
        return repr(self.explanations)

    def add(self, images, names):
        self.explanations = {"image": images, "name": names}

    def get_explanations(self):
        """
        Gets the generated explanations.
        """
        return self.explanations

    def plot(self, num_figures_per_row=2, **kwargs):
        """
        Returns a matplotlib figure plotting the stored images.

        :param num_figures_per_row: The number of figures for each row.
        :return: A matplotlib figure plotting the stored images.
        """
        import matplotlib.pyplot as plt
        if self.explanations is None:
            return

        names = self.explanations["name"]
        images = self.explanations["image"]
        num_cols = num_figures_per_row
        num_rows = len(images) // num_cols
        if num_rows * num_cols != len(images):
            num_rows += 1
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i in range(len(images)):
            r, c = divmod(i, num_cols)
            plt.sca(axes[r, c])
            plt.imshow(images[i].to_pil())
            plt.title(names[i])
            plt.xticks([])
            plt.yticks([])
        return fig

    def _plotly_figure(self, num_figures_per_row=2, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        names = self.explanations["name"]
        images = self.explanations["image"]
        num_cols = num_figures_per_row
        num_rows = len(images) // num_cols
        if num_rows * num_cols != len(images):
            num_rows += 1

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[name for name in names],
        )
        for i in range(len(images)):
            r, c = divmod(i, num_cols)
            img_figure = px.imshow(images[i].to_pil())
            fig.add_trace(img_figure.data[0], row=r + 1, col=c + 1)

        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig

    def plotly_plot(self, num_figures_per_row=2, **kwargs):
        """
        Returns a plotly dash figure plotting the stored images.

        :param num_figures_per_row: The number of figures for each row.
        :return: A plotly dash figure plotting the stored images.
        """
        return DashFigure(self._plotly_figure(
            num_figures_per_row=num_figures_per_row, **kwargs))

    def ipython_plot(self, num_figures_per_row=2, **kwargs):
        """
        Plots the stored images in IPython.

        :param num_figures_per_row: The number of figures for each row.
        """
        import plotly

        return plotly.offline.iplot(self._plotly_figure(
            num_figures_per_row=num_figures_per_row, **kwargs))
