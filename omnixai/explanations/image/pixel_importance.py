#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Pixel importance explanations for vision tasks.
"""
import warnings
import numpy as np
from ..base import ExplanationBase, DashFigure


class PixelImportance(ExplanationBase):
    """
    The class for pixel importance explanations. The pixel importance scores
    of the input instances are stored in a list. Each item in the list is a dict with
    the following format: `{"image": the input image, "scores": the pixel importance scores}`.
    If the task is `classification`, the dict has an additional entry `{"target_label":
    the predicted label of the input instance}`.
    """

    def __init__(self, mode, explanations=None, use_heatmap=False):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        :param explanations: The explanation results for initializing ``PixelImportance``,
            which is optional.
        :param use_heatmap: `True` if plot heatmaps (for Grad-CAM methods)
            or `False` if plot raw values (for integrated gradients and SHAP).
        """
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations
        self.use_heatmap = use_heatmap

    def __repr__(self):
        return repr(self.explanations)

    def add(self, image, target_label, importance_scores, **kwargs):
        """
        Adds the generated explanation of one image.

        :param image: The input image.
        :param target_label: The label to be explained, which is ignored for regression.
        :param importance_scores: The list of the corresponding pixel importance scores.
        """
        e = {"image": image, "scores": importance_scores}
        e.update(kwargs)
        if self.mode == "classification":
            e["target_label"] = target_label
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``PixelImportance``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific image (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"image": the input image, "scores": the pixel
            importance scores}`. If the task is `classification`, the dict has an additional
            entry `{"target_label": the predicted label of the input instance}`.
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, class_names=None, max_num_figures=20, **kwargs):
        """
        Returns matplotlib figures plotting pixel importance scores.

        :param index: The index of an explanation result stored in ``PixelImportance``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it returns a figure plotting the pixel importance scores
            for the first 10 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param max_num_figures: The maximum number of figures to plot.
        :return: A list of matplotlib figures plotting pixel importance scores.
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

        figures = []
        for index in indices:
            exp = explanations[index]
            all_scores = exp["scores"]
            if not isinstance(all_scores, (list, tuple)):
                all_scores = [all_scores]
            if "labels" in exp:
                labels = exp["labels"]
                if not isinstance(labels, (list, tuple)):
                    labels = [labels]
                assert len(all_scores) == len(labels)
            image = np.transpose(np.stack([exp["image"]] * 3), (1, 2, 0)) \
                if exp["image"].ndim == 2 else exp["image"]

            for i, scores in enumerate(all_scores):
                fig, axes = plt.subplots(1, 2, squeeze=False)
                importance_scores = np.expand_dims(scores, axis=-1) \
                    if scores.ndim == 2 else scores

                # Image and importance scores
                if not self.use_heatmap:
                    scores = _plot_pixel_importance(importance_scores, image, overlay=True)
                else:
                    scores = _plot_pixel_importance_heatmap(importance_scores, image, overlay=True)
                plt.sca(axes[0, 0])
                plt.imshow(scores)
                plt.xticks([])
                plt.yticks([])
                if "labels" in exp:
                    plt.title(f"{labels[i]}")
                elif "target_label" in exp:
                    label = exp["target_label"]
                    class_name = label if class_names is None else class_names[label]
                    plt.title(f"{class_name}")

                # Positive pixel importance scores
                if not self.use_heatmap:
                    scores = _plot_pixel_importance(importance_scores, image, polarity="positive")
                else:
                    scores = _plot_pixel_importance_heatmap(importance_scores, image, overlay=False)
                plt.sca(axes[0, 1])
                plt.imshow(scores)
                plt.xticks([])
                plt.yticks([])
                plt.title("Score")

                figures.append(fig)
                if len(figures) >= max_num_figures:
                    return figures
        return figures

    def _plotly_figure(self, index, class_names=None, max_num_figures=20, **kwargs):
        import plotly.express as px
        from plotly.subplots import make_subplots

        exp = self.explanations[index]
        all_scores = exp["scores"]
        if not isinstance(all_scores, (list, tuple)):
            all_scores = [all_scores]
            all_scores = all_scores[:max_num_figures]
        if "labels" in exp:
            labels = exp["labels"]
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            labels = labels[:max_num_figures]
            assert len(all_scores) == len(labels)
        image = np.transpose(np.stack([exp["image"]] * 3), (1, 2, 0)) \
            if exp["image"].ndim == 2 else exp["image"]

        # Subtitles for the plots
        if len(all_scores) == 1:
            subplot_titles = ["Overlay", "Score"]
        else:
            subplot_titles = []
            if "labels" in exp:
                for i in range(len(all_scores)):
                    subplot_titles += [str(labels[i]), "Score"]
            else:
                for i in range(len(all_scores)):
                    subplot_titles += ["Overlay", "Score"]

        fig = make_subplots(rows=len(all_scores), cols=2, subplot_titles=subplot_titles)
        for i, scores in enumerate(all_scores):
            importance_scores = np.expand_dims(scores, axis=-1) \
                if scores.ndim == 2 else scores
            # Image and importance scores
            if not self.use_heatmap:
                img = _plot_pixel_importance(importance_scores, image, overlay=True)
            else:
                img = _plot_pixel_importance_heatmap(importance_scores, image, overlay=True)
            img_figure = px.imshow(img.squeeze().astype(np.uint8))
            fig.add_trace(img_figure.data[0], row=i + 1, col=1)
            # Positive pixel importance scores
            if not self.use_heatmap:
                img = _plot_pixel_importance(importance_scores, image, polarity="positive")
            else:
                img = _plot_pixel_importance_heatmap(importance_scores, image, overlay=False)
            img_figure = px.imshow(img.squeeze().astype(np.uint8))
            fig.add_trace(img_figure.data[0], row=i + 1, col=2)

        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_yaxes(visible=False, showticklabels=False)
        if len(all_scores) > 1:
            fig.update_layout(height=260 * len(all_scores))
        return fig

    def plotly_plot(self, index=0, class_names=None, max_num_figures=20, **kwargs):
        """
        Returns a plotly dash figure plotting the pixel importance scores for one
        specific instance.

        :param index: The index of the instance which cannot be None, e.g., it will
            plot the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param max_num_figures: The maximum number of figures to plot.
        :return: A plotly dash figure plotting the pixel importance scores.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(
            index, class_names=class_names, max_num_figures=max_num_figures, **kwargs))

    def ipython_plot(self, index=0, class_names=None, max_num_figures=20, **kwargs):
        """
        Plots the pixel importance scores in IPython.

        :param index: The index of the instance which cannot be None, e.g., it will
            plot the first explanation result when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param max_num_figures: The maximum number of figures to plot.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        return plotly.offline.iplot(self._plotly_figure(
            index, class_names=class_names, max_num_figures=max_num_figures, **kwargs))

    @classmethod
    def from_dict(cls, d):
        explanations = []
        for e in d["explanations"]:
            e["image"] = np.array(e["image"])
            e["scores"] = np.array(e["scores"])
            explanations.append(e)
        return PixelImportance(
            mode=d["mode"],
            explanations=explanations,
            use_heatmap=d["use_heatmap"]
        )


def _plot_pixel_importance_heatmap(importance_scores, image, overlay=True):
    import cv2

    heatmap = cv2.applyColorMap(np.uint8(255 * importance_scores), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    if np.max(image) > 1:
        image = image / 255.0
    if overlay:
        heatmap = heatmap + image
    heatmap = heatmap / np.max(heatmap)
    return np.uint8(255 * heatmap)


def _compute_threshold_by_top_percentage(attributions, percentage=60.0):
    """
    Compute the threshold value that maps to the top percentage of values.
    This function takes the cumulative sum of attributions and computes the set
    of top attributions that contribute to the given percentage of the total sum.
    The lowest value of this given set is returned.
    Adapted from:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py

    Args:
      attributions: (numpy.array) The provided attributions.
      percentage: (float) Specified percentage by which to threshold.
    Returns:
      (float) The threshold value.
    Raises:
      ValueError: if percentage is not in [0, 100].
    """
    if percentage < 0 or percentage > 100:
        raise ValueError("percentage must be in [0, 100]")

    # For percentage equal to 100, this should in theory return the lowest
    # value as the threshold. However, due to precision errors in numpy's cumsum,
    # the last value won't sum to 100%. Thus, in this special case, we force the
    # threshold to equal the min value.
    if percentage == 100:
        return np.min(attributions)

    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    # Sort the attributions from largest to smallest.
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    # Compute a normalized cumulative sum, so that each attribution is mapped to
    # the percentage of the total sum that it and all values above it contribute.
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    return threshold


def _linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2):
    """
    Transform the attributions by a linear function.
    Adapted from:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
    """
    if clip_above_percentile < 0 or clip_above_percentile > 100:
        raise ValueError("clip_above_percentile must be in [0, 100]")
    if clip_below_percentile < 0 or clip_below_percentile > 100:
        raise ValueError("clip_below_percentile must be in [0, 100]")
    if low < 0 or low > 1:
        raise ValueError("low must be in [0, 1]")

    m = _compute_threshold_by_top_percentage(attributions, percentage=100 - clip_above_percentile)
    e = _compute_threshold_by_top_percentage(attributions, percentage=100 - clip_below_percentile)
    # Transform the attributions by a linear function f(x) = a*x + b such that
    # f(m) = 1.0 and f(e) = low. Derivation:
    #   a*m + b = 1, a*e + b = low  ==>  a = (1 - low) / (m - e)
    #                               ==>  b = low - (1 - low) * e / (m - e)
    #                               ==>  f(x) = (1 - low) (x - e) / (m - e) + low
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    # Recover the original sign of the attributions.
    transformed *= np.sign(attributions)
    # Map values below low to 0.
    transformed *= transformed >= low
    # Clip values above and below.
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def _plot_pixel_importance(
    attributions,
    image,
    polarity="positive",
    clip_above_percentile=99.0,
    clip_below_percentile=0,
    outlines_component_percentage=90,
    use_linear_transform=True,
    overlay=False,
):
    """
    Plots pixel importance scores.
    Adapted from:
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
    """
    if polarity == "both":
        pos_attributions = _plot_pixel_importance(
            attributions,
            image,
            polarity="positive",
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            outlines_component_percentage=outlines_component_percentage,
            overlay=False,
        )
        neg_attributions = _plot_pixel_importance(
            attributions,
            image,
            polarity="negative",
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            outlines_component_percentage=outlines_component_percentage,
            overlay=False,
        )
        attributions = pos_attributions + neg_attributions
        if overlay:
            attributions = np.clip(0.7 * image + 0.5 * attributions, 0, 255)
        return attributions

    elif polarity == "positive":
        attributions = np.clip(attributions, 0, 1)
        channel = [0, 255, 0]

    elif polarity == "negative":
        attributions = np.abs(np.clip(attributions, -1, 0))
        channel = [255, 0, 0]

    attributions = np.mean(attributions, axis=2)
    if use_linear_transform:
        attributions = _linear_transform(attributions, clip_above_percentile, clip_below_percentile, 0.0)
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        attributions = np.clip(0.7 * image + 0.5 * attributions, 0, 255)
    return attributions.astype(int)
