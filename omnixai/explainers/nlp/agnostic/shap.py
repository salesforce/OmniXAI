#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for text classification.
"""
import sys
import shap
import numpy as np
from typing import Callable

from ...base import ExplainerBase
from ....data.text import Text
from ....explanations.text.word_importance import WordImportance


class ShapText(ExplainerBase):
    """
    The SHAP explainer for text classification.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    This explainer only supports `TextClassificationPipeline` in the `transformer` library.
    """

    explanation_type = "local"
    alias = ["shap"]

    def __init__(self, model: Callable, mode: str = "classification", **kwargs):
        """
        :param model: The model with type `transformers.pipelines.text_classification.TextClassificationPipeline`.
        :param mode: The task type can be `classification` only.
        :param kwargs: Additional parameters for `shap.Explainer`. Please refer to the doc of
            `shap.Explainer`.
        """
        from ....utils.misc import is_torch_available, is_transformers_available

        if not is_torch_available():
            raise EnvironmentError("`torch` is not installed, please install `torch`.")
        if not is_transformers_available():
            raise EnvironmentError("`transformers` is not installed, please install `transformers`.")
        module = sys.modules["transformers.pipelines.text_classification"]
        if not isinstance(model, getattr(module, "TextClassificationPipeline", None)):
            raise TypeError(
                "`predict_function` should be an instance of "
                "`transformers.pipelines.text_classification.TextClassificationPipeline`"
            )

        super().__init__()
        self.mode = mode
        self.model = model
        self.explainer = shap.Explainer(model, **kwargs)

    def explain(self, X: Text, y=None, **kwargs) -> WordImportance:
        """
        Generates the word/token-importance explanations for the input instances.

        :param X: A batch of input instances.
        :param y: A batch of labels to explain. For classification, the top predicted label
            of each input instance will be explained when `y = None`.
        :param kwargs: Additional parameters for `shap.Explainer`.
        :return: The explanations for all the instances.
        """
        explanations = WordImportance(mode=self.mode)
        class_names = None

        if y is not None:
            if type(y) == int:
                y = [y for _ in range(len(X))]
            else:
                assert len(X) == len(y), (
                    f"Parameter ``y`` is a {type(y)}, the length of y "
                    f"should be the same as the number of images in X."
                )
        else:
            scores = self.model(X.values)
            prediction_scores = [[s["score"] for s in ss] for ss in scores]
            class_names = [[s["label"] for s in ss] for ss in scores]
            y = np.argmax(prediction_scores, axis=1).astype(int)

        shap_values = self.explainer(X.values, **kwargs)
        for i in range(len(X)):
            scores = shap_values.values[i][:, y[i]]
            tokens = shap_values.data[i].tolist()
            explanations.add(
                instance=X[i].to_str(),
                target_label=y[i] if class_names is None else class_names[i][y[i]],
                tokens=tokens,
                importance_scores=scores,
            )
        return explanations
