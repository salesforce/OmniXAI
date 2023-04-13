#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Plain image explanations for vision tasks.
"""
import warnings
from ..base import ExplanationBase, DashFigure


class PlainExplanation(ExplanationBase):
    """
    The class for plain image explanation. It stores a batch of images and the corresponding
    names. Each image represents a plain explanation.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def __repr__(self):
        return repr(self.explanations)

    def add(self, images, names=None):
        self.explanations.append({"image": images, "name": names})

    def get_explanations(self):
        """
        Gets the generated explanations.
        """
        return self.explanations if len(self.explanations) > 1 else self.explanations[0]

    def _estimate_num_per_row(self, index=0, t=8):
        n = len(self.explanations[index]["image"])
        return 1 if n == 1 else min(max((n + t - 1) // t, 2), 8)

    def plot(self, index=None, num_figures_per_row=None, **kwargs):
        """
        Returns a matplotlib figure plotting the stored images.

        :param index: The index of the stored results.
        :param num_figures_per_row: The number of figures for each row.
        :return: A matplotlib figure plotting the stored images.
        """
        import matplotlib.pyplot as plt
        if num_figures_per_row is None:
            num_figures_per_row = self._estimate_num_per_row()
        index = index if index is not None else 0

        exp = self.explanations[index]
        names, images = exp["name"], exp["image"]
        if len(images) < num_figures_per_row:
            warnings.warn("`num_figures_per_row` is greater than the number of images.")
        num_cols = num_figures_per_row
        num_rows = len(images) // num_cols
        if num_rows * num_cols != len(images):
            num_rows += 1
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i in range(len(images)):
            r, c = divmod(i, num_cols)
            plt.sca(axes[r, c])
            plt.imshow(images[i])
            if names is not None:
                plt.title(names[i])
            plt.xticks([])
            plt.yticks([])
        return fig

    def _plotly_figure(self, index, num_figures_per_row, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots
        if num_figures_per_row is None:
            num_figures_per_row = self._estimate_num_per_row()

        exp = self.explanations[index]
        names, images = exp["name"], exp["image"]
        if len(images) < num_figures_per_row:
            warnings.warn("`num_figures_per_row` is greater than the number of images.")
        num_cols = num_figures_per_row
        num_rows = len(images) // num_cols
        if num_rows * num_cols != len(images):
            num_rows += 1

        if num_rows == 1 and num_cols == 1:
            title = None if names is None else names[0]
            fig = px.imshow(images[0], title=title)
            max_height = images[0].size[1]
        else:
            fig = make_subplots(
                rows=num_rows,
                cols=num_cols,
                subplot_titles=[name for name in names] if names is not None else None,
            )
            max_height = 0
            for i in range(len(images)):
                r, c = divmod(i, num_cols)
                img_figure = px.imshow(images[i])
                fig.add_trace(img_figure.data[0], row=r + 1, col=c + 1)
                max_height = max(max_height, images[i].size[1])

        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_layout(height=max(max_height * num_rows, 300 * num_rows))
        return fig

    def plotly_plot(self, index=None, num_figures_per_row=None, **kwargs):
        """
        Returns a plotly dash figure plotting the stored images.

        :param index: The index of the stored results.
        :param num_figures_per_row: The number of figures for each row.
        :return: A plotly dash figure plotting the stored images.
        """
        index = index if index is not None else 0
        return DashFigure(self._plotly_figure(
            index=index, num_figures_per_row=num_figures_per_row, **kwargs))

    def ipython_plot(self, index=None, num_figures_per_row=None, **kwargs):
        """
        Plots the stored images in IPython.

        :param index: The index of the stored results.
        :param num_figures_per_row: The number of figures for each row.
        """
        import plotly

        index = index if index is not None else 0
        return plotly.offline.iplot(self._plotly_figure(
            index=index, num_figures_per_row=num_figures_per_row, **kwargs))

    @classmethod
    def from_dict(cls, d):
        import numpy as np
        from PIL import Image as PilImage
        explanations = []
        for e in d["explanations"]:
            e["image"] = [PilImage.fromarray(np.array(img).astype(np.uint8))
                          for img in e["image"]]
            explanations.append(e)
        exp = PlainExplanation()
        exp.explanations = explanations
        return exp
