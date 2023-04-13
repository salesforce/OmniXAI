#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Image mask explanations for vision tasks.
"""
import warnings
import numpy as np
from skimage.segmentation import mark_boundaries

from ..base import ExplanationBase, DashFigure


class MaskExplanation(ExplanationBase):
    """
    The class for image mask explanations used by LIME. It uses a list to store image mask
    explanations of the input instances. Each item in the list is a dict with the following format:
    `{"images": the input images, "labels": the predicted labels, "masks": the masks returned by
    the explainer, e.g., LIME, "boundary": the boundaries extracted from "masks"}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def add(self, labels, images, masks):
        """
        Adds the generated explanation of one image.

        :param labels: The predicted labels.
        :param images: The input images.
        :param masks: The mask explanation results.
        """
        self.explanations.append(
            {
                "labels": labels,
                "images": images,
                "masks": masks,
                "boundary": [mark_boundaries(image / 255.0, mask) for image, mask in zip(images, masks)],
            }
        )

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index or name of the instance. When ``index`` is None,
            this method return all the explanations.
        :return: The explanation for one specific instance (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"images": the input images,
            "labels": the predicted labels, "masks": the masks returned by the explainer,
            e.g., LIME, "boundary": the boundaries extracted from "masks"}`.
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, class_names=None, **kwargs):
        """
        Returns a matplotlib figure showing the explanations.

        :param index: The index of the instance, e.g., it will plot the first explanation
            result when ``index = 0``. When ``index`` is None, it returns a figure showing
            the mask explanations for the first 10 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A matplotlib figure plotting mask explanations.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if len(indices) > 10:
            warnings.warn(
                f"There are too many instances ({len(indices)} > 10), " f"so only the first 10 instances are plotted."
            )
            indices = indices[:10]
        if len(indices) == 0:
            return

        num_rows = len(indices)
        num_cols = len(self.explanations[0]["labels"])
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i, index in enumerate(indices):
            exp = explanations[index]
            for j, label in enumerate(exp["labels"]):
                plt.sca(axes[i, j])
                plt.imshow(exp["boundary"][j])
                class_name = label if class_names is None else class_names[label]
                plt.title(f"{class_name}")
                plt.xticks([])
                plt.yticks([])
        return fig

    def _plotly_figure(self, index, class_names=None, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        exp = self.explanations[index]
        num_cols = 2
        num_rows = int(np.ceil(len(exp["labels"]) / num_cols))
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[
                f"Class: {label}" if class_names is None else f"Class: {class_names[label]}" for label in exp["labels"]
            ],
        )
        for i, label in enumerate(exp["labels"]):
            row, col = divmod(i, num_cols)
            img = exp["boundary"][i]
            img_figure = px.imshow(img.squeeze())
            fig.add_trace(img_figure.data[0], row=row + 1, col=col + 1)

        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        return fig

    def plotly_plot(self, index=0, class_names=None, **kwargs):
        """
        Returns a plotly dash figure showing the explanations for one specific instance.

        :param index: The index of the instance which cannot be None, e.g., it will plot
            the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure plotting mask explanations.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, **kwargs))

    def ipython_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots mask explanations in IPython.

        :param index: The index of the instance which cannot be None, e.g., it will plot
            the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        return plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, **kwargs))

    @classmethod
    def from_dict(cls, d):
        explanations = []
        for e in d["explanations"]:
            e["images"] = np.array(e["images"])
            e["masks"] = np.array(e["masks"])
            e["boundary"] = [np.array(b) for b in e["boundary"]]
            explanations.append(e)
        exp = MaskExplanation()
        exp.explanations = explanations
        return exp
