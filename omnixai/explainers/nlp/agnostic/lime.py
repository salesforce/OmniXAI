#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LIME explainer for text classification.
"""
import warnings
import numpy as np
from lime import lime_text
from typing import Callable

from ...base import ExplainerBase
from ....data.text import Text
from ....explanations.text.word_importance import WordImportance


class LimeText(ExplainerBase):
    """
    The LIME explainer for text classification.
    If using this explainer, please cite the original work: https://github.com/marcotcr/lime.
    This explainer only supports text classification tasks.
    """

    explanation_type = "local"
    alias = ["lime"]

    def __init__(self, predict_function: Callable, mode: str = "classification", **kwargs):
        """
        :param predict_function: The prediction function corresponding to the machine learning
            model to explain. When the task is `classification`, the outputs of the ``predict_function``
            are the class probabilities.
        :param mode: The task type can be `classification` only.
        :param kwargs: Additional parameters for `lime_text.LimeTextExplainer`. Please refer to the doc of
            `lime_text.LimeTextExplainer`.
        """
        super().__init__()
        assert mode == "classification", "Only supports classification tasks for text data."
        if "training_data" in kwargs:
            kwargs.pop("training_data")
        self.mode = mode
        self.predict_fn = lambda x: predict_function(Text(x))
        self.explainer = lime_text.LimeTextExplainer(**kwargs)

    def explain(self, X: Text, y=None, **kwargs) -> WordImportance:
        """
        Generates the word/token-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For classification, the top predicted label
            of each input instance will be explained when `y = None`.
        :param kwargs: Additional parameters for `LimeTextExplainer.explain_instance`.
        :return: The explanations for all the input instances.
        """
        if "labels" in kwargs:
            warnings.warn(
                "Argument `labels` is not used, "
                "please use `y` instead of `labels` to specify "
                "the labels you want to explain."
            )
            kwargs.pop("labels")
        if "top_labels" in kwargs:
            warnings.warn("Argument `top_labels` is not used.")
            kwargs.pop("top_labels")
        explanations = WordImportance(mode=self.mode)

        if y is not None:
            if type(y) == int:
                y = [y for _ in range(len(X))]
            else:
                assert len(X) == len(y), (
                    f"Parameter `y` is a {type(y)}, the length of y "
                    f"should be the same as the number of instances in X."
                )
        else:
            scores = self.predict_fn(X.to_str())
            y = np.argmax(scores, axis=1).astype(int)

        for i in range(len(X)):
            e = self.explainer.explain_instance(X[i].to_str(), classifier_fn=self.predict_fn, labels=(y[i],), **kwargs)
            exp = e.as_list(label=y[i])
            explanations.add(
                instance=X[i].to_str(),
                target_label=y[i] if y is not None else None,
                tokens=[e[0] for e in exp],
                importance_scores=[e[1] for e in exp],
            )
        return explanations
