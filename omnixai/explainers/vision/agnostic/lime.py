#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LIME explainer for image classification.
"""
from lime import lime_image
from typing import Callable

from ...base import ExplainerBase
from ....data.image import Image
from ....preprocessing.image import Resize
from ....explanations.image.mask import MaskExplanation


class LimeImage(ExplainerBase):
    """
    The LIME explainer for image classification.
    If using this explainer, please cite the original work: https://github.com/marcotcr/lime.
    This explainer only supports image classification.
    """

    explanation_type = "local"
    alias = ["lime"]

    def __init__(self, predict_function: Callable, mode: str = "classification", **kwargs):
        """
        :param predict_function: The prediction function corresponding to the machine learning
            model to explain. For classification, the outputs of the ``predict_function``
            are the class probabilities.
        :param mode: The task type can be `classification` only.
        """
        super().__init__()
        assert mode == "classification", "Only supports classification tasks for image data."
        self.mode = mode
        self.predict_fn = lambda x: predict_function(Image(x, batched=True))

    def explain(self, X, **kwargs) -> MaskExplanation:
        """
        Generates the explanations for the input instances.

        :param X: A batch of input instances.
        :param kwargs: Additional parameters, e.g., ``top_labels`` -- the number
            of the top labels to explain. Please refer to the doc of
            `LimeImageExplainer.explain_instance`.
        :return: The explanations for all the input instances.
        """
        if "top_labels" not in kwargs:
            kwargs["top_labels"] = 2
        explanations = MaskExplanation()
        # Reduce the size of the images
        if max(X.shape[1], X.shape[2]) > 256:
            X = Resize(256).transform(X)

        for i in range(X.shape[0]):
            explanation = lime_image.LimeImageExplainer().explain_instance(
                image=X[i].to_numpy()[0],
                classifier_fn=self.predict_fn,
                hide_color=kwargs.get("hide_color", None),
                top_labels=kwargs.get("top_labels", 2),
                num_features=kwargs.get("num_features", 100000),
                num_samples=kwargs.get("num_samples", 1000),
                batch_size=kwargs.get("batch_size", 10),
                segmentation_fn=kwargs.get("segmentation_fn", None),
                random_seed=kwargs.get("random_seed", None),
            )
            images, masks = [], []
            for label in explanation.top_labels:
                image, mask = explanation.get_image_and_mask(
                    label=label,
                    positive_only=kwargs.get("positive_only", False),
                    num_features=kwargs.get("num_features", 5),
                    hide_rest=kwargs.get("hide_rest", False),
                )
                images.append(image)
                masks.append(mask)
            explanations.add(explanation.top_labels, images, masks)
        return explanations
