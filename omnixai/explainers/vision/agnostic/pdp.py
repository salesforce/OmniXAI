#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The partial dependence plots for vision tasks.
"""
import numpy as np
from collections import defaultdict

from ...base import ExplainerBase
from ....data.image import Image
from ....explanations.image.pixel_importance import PixelImportance


class PartialDependenceImage(ExplainerBase):
    """
    The partial dependence plots for vision tasks. The input image is segmented by a particular
    segmentation method, e.g., "quickshift". For each segment, its importance score is measured
    by the average change of the predicted value when the segment is replaced by new segments constructed
    in the grid search.
    """

    explanation_type = "local"
    alias = ["pdp", "partial_dependence"]

    def __init__(self, predict_function, mode="classification", **kwargs):
        """
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        """
        super().__init__()
        assert mode in [
            "classification",
            "regression",
        ], f"Unknown mode: {mode}, please choose `classification` or `regression`"
        self.predict_function = predict_function
        self.mode = mode

    @staticmethod
    def _extract_segments(mask):
        """
        Extracts image segments given a mask image.

        :param mask: A mask image.
        :return: Image segments.
        :rtype: np.ndarray
        """
        segments = defaultdict(list)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                segments[mask[i, j]].append((i, j))
        segments = [segments[k] for k in sorted(segments.keys())]
        masks = np.zeros((len(segments), mask.shape[0], mask.shape[1]))
        for i in range(len(segments)):
            m = np.zeros((mask.shape[0], mask.shape[1]))
            for a, b in segments[i]:
                m[a, b] = 1
            masks[i] = m
        return masks

    def _compute_pdp(self, image, segments, label, grid_resolution):
        """
        Computes partial dependence scores.

        :param image: The input image.
        :param segments: The masks for the image segments.
        :param label: The labels to explain.
        :param grid_resolution: The number of candidate colors for each segment.
        :return: The importance scores for all the segments.
        :rtype: np.ndarray
        """
        candidates = np.linspace(50, 200, num=grid_resolution)
        scores = np.zeros((len(segments), len(candidates)))
        for i, mask in enumerate(segments):
            if image.ndim == 3:
                mask = np.concatenate([np.expand_dims(mask, axis=-1)] * image.shape[-1], axis=-1)
            ims = Image(
                data=np.stack([mask * value + (1 - mask) * image for value in candidates]),
                batched=True,
                channel_last=True,
            )
            if label is not None:
                scores[i] = np.array([self.predict_function(im)[0, label] for im in ims])
            else:
                scores[i] = np.array([self.predict_function(im)[0] for im in ims])
        return scores

    def explain(self, X: Image, y=None, **kwargs) -> PixelImportance:
        """
        Generates PDP explanations.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each input instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters in the PDP explainer, e.g., ``grid_resolution`` --
            the resolution in the grid search, and ``n_segments`` -- the number of image segments used
            by image segmentation methods.
        :return: The generated explanations, e.g., the importance scores for image segments.
        """
        from ....utils.segmentation import image_segmentation

        grid_resolution = kwargs.get("grid_resolution", 10)
        explanations = PixelImportance(mode=self.mode)

        if self.mode == "classification":
            outputs = self.predict_function(X)
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(X))]
                else:
                    assert len(X) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of images in X."
                    )
            else:
                y = np.argmax(outputs, axis=1).astype(int)
            prediction_scores = [outputs[i, y[i]] for i in range(len(y))]
        else:
            y = None
            prediction_scores = self.predict_function(X)

        for i in range(X.shape[0]):
            image = X[i].to_numpy()[0]
            label = y[i] if y is not None else None
            score = prediction_scores[i]
            # Image segmentation
            mask = image_segmentation(image, method="slic", n_segments=kwargs.get("n_segments", 20), **kwargs)
            segments = self._extract_segments(mask)
            # Compute PDP scores
            pdp_scores = self._compute_pdp(image, segments, label, grid_resolution)
            pdp_scores = np.mean(score - pdp_scores, axis=1)
            importance = segments * pdp_scores.reshape((pdp_scores.shape[0], 1, 1))
            importance = np.sum(importance, axis=0)
            if image.ndim == 3:
                importance = np.concatenate([np.expand_dims(importance, axis=-1)] * image.shape[-1], axis=-1)
            # Add explanation
            explanations.add(image=image, target_label=label, importance_scores=importance)
        return explanations
