#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The NLP counterfactual explainer based on the Polyjuice model.
"""
import numpy as np
import pandas as pd
from typing import Callable
from collections import Counter

from omnixai.explainers.base import ExplainerBase
from omnixai.data.text import Text
from omnixai.explanations.tabular.counterfactual import CFExplanation


class Polyjuice(ExplainerBase):
    """
    The counterfactual explainer for NLP tasks.
    The method implemented here is based on the model developed by Wu et al.,
    please cite the work: https://github.com/tongshuangwu/polyjuice if using this explainer.
    """

    explanation_type = "local"
    alias = ["polyjuice"]

    def __init__(self, predict_function: Callable, mode: str = "classification", **kwargs):
        """
        :param predict_function: The prediction function corresponding to the machine learning
            model to explain. When the task is `classification`, the outputs of the ``predict_function``
            are the class probabilities.
        :param mode: The task type which can be `classification` only.
        :param kwargs: Additional parameters, e.g., `model_path` and `cuda`.
        """
        super().__init__()
        assert mode == "classification", "Only supports classification tasks for text data."
        self.mode = mode
        self.predict_function = predict_function

        from polyjuice import Polyjuice
        self.explainer = Polyjuice(
            model_path=kwargs.get("model_path", "uw-hai/polyjuice"),
            is_cuda=kwargs.get("cuda", True)
        )

    def _predict(self, X: Text):
        scores = self.predict_function(X)
        return scores, np.argmax(scores, axis=1)

    def explain(self, X: Text, max_number_examples: int = 5, **kwargs):
        """
        Generates the counterfactual explanations for the input instances.

        :param X: A batch of input instances.
        :param max_number_examples: The maximum number of the generated counterfactual
            examples for each input instance.
        :param kwargs: Additional parameters for `polyjuice.Polyjuice`.
        :return: The explanations for all the input instances.
        :rtype: NLPCounterfactualExplanation
        """
        from polyjuice.generations import ALL_CTRL_CODES

        explanations = CFExplanation()
        predictions, labels = self._predict(X)

        for idx, text in enumerate(X.values):
            original_label = labels[idx]
            perturb_texts = self.explainer.perturb(
                text.lower(),
                ctrl_code=ALL_CTRL_CODES,
                num_perturbations=kwargs.get("num_perturbations", None),
                perplex_thred=kwargs.get("perplex_thred", 10)
            )
            perturb_texts = list(set([t.lower() for t in perturb_texts]))
            perturb_predictions, perturb_labels = self._predict(Text(perturb_texts))

            # Only keep the generated texts with different predicted labels
            cf_texts, cf_predictions, cf_labels = [], [], []
            for t, p, label in zip(perturb_texts, perturb_predictions, perturb_labels):
                if label != original_label:
                    cf_texts.append(t)
                    cf_predictions.append(p)
                    cf_labels.append(label)
            # Cannot find counterfactual examples
            if len(cf_texts) == 0:
                explanations.add(
                    query=pd.DataFrame([[text, original_label]], columns=["text", "label"]),
                    cfs=None
                )
            else:
                examples = []
                original_tokens = Text(text).to_tokens()[0]
                original_token_counts = Counter(original_tokens)

                for t, p, label in zip(cf_texts, cf_predictions, cf_labels):
                    perturb_tokens = Text(t).to_tokens()[0]
                    perturb_token_counts = Counter(perturb_tokens)

                    a, b = 0, 0
                    for key, value in original_token_counts.items():
                        a += max(0, value - perturb_token_counts.get(key, 0))
                    for key, value in perturb_token_counts.items():
                        b += max(0, value - original_token_counts.get(key, 0))
                    distance = max(a, b)

                    score = p[original_label] + (distance / len(original_tokens))
                    examples.append((t, score, label))
                examples = sorted(examples, key=lambda x: x[1])[:max_number_examples]
                explanations.add(
                    query=pd.DataFrame([[text, original_label]], columns=["text", "label"]),
                    cfs=pd.DataFrame([(e[0], e[2]) for e in examples], columns=["text", "label"])
                )
        return explanations
