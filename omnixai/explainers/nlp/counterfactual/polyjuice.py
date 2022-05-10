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
            are the class probabilities. When the task is `qa` (question-answering), the outputs are
            the generated answers.
        :param mode: The task type, e.g., `classification` pr `qa` (question-answering).
        :param kwargs: Additional parameters, e.g., `model_path` and `cuda`.
        """
        super().__init__()
        assert mode in ["classification", "qa"], \
            "Only supports classification and question-answering (qa) tasks for text data."
        self.mode = mode
        self.predict_function = predict_function

        from polyjuice import Polyjuice
        self.explainer = Polyjuice(
            model_path=kwargs.get("model_path", "uw-hai/polyjuice"),
            is_cuda=kwargs.get("cuda", True)
        )

    def _predict(self, x: Text):
        scores = self.predict_function(x)
        return scores, np.argmax(scores, axis=1)

    @staticmethod
    def _distance(a, b):
        if isinstance(a, Text):
            a, b = Counter(a.to_tokens()[0]), Counter(b.to_tokens()[0])
        elif isinstance(a, list):
            a, b = Counter(a), Counter(b)

        x, y = 0, 0
        for key, value in a.items():
            x += max(0, value - b.get(key, 0))
        for key, value in b.items():
            y += max(0, value - a.get(key, 0))
        distance = max(x, y)
        return distance

    def _perturb(self, text, **kwargs):
        ce_type = kwargs.get("ce_type", "perturb")
        if ce_type == "perturb":
            perturb_texts = self.explainer.perturb(
                text,
                num_perturbations=kwargs.get("num_perturbations", 10),
                perplex_thred=kwargs.get("perplex_thred", 10)
            )
        elif ce_type == "blank":
            perturb_texts = self.explainer.get_random_blanked_sentences(
                sentence=text,
                max_blank_sent_count=kwargs.get("num_perturbations", 10),
                is_token_only=True,
                max_blank_block=1
            )
        else:
            raise ValueError(f"Unknown `ce_type`: {ce_type}. Please choose 'perturb' or 'blank'.")
        return perturb_texts

    def _explain_classification(self, X: Text, max_number_examples: int = 5, **kwargs):
        explanations = CFExplanation()
        predictions, labels = self._predict(X)
        tokenizer = X.tokenizer

        for idx, text in enumerate(X.values):
            original_label = labels[idx]
            perturb_texts = self._perturb(text.lower(), **kwargs)
            perturb_texts = list(set([t.lower() for t in perturb_texts]))
            perturb_predictions, perturb_labels = self._predict(
                Text(perturb_texts, tokenizer=tokenizer))

            flips, non_flips = [], []
            original_tokens = Text(text, tokenizer=tokenizer).to_tokens()[0]
            original_token_counts = Counter(original_tokens)

            for t, p, label in zip(perturb_texts, perturb_predictions, perturb_labels):
                perturb_tokens = Text(t, tokenizer=tokenizer).to_tokens()[0]
                perturb_token_counts = Counter(perturb_tokens)
                distance = self._distance(original_token_counts, perturb_token_counts)
                score = p[original_label] + (distance / len(original_tokens))
                if label != original_label:
                    flips.append((t, score, label))
                else:
                    non_flips.append((t, score, label))

            examples = sorted(flips, key=lambda x: x[1]) + sorted(non_flips, key=lambda x: x[1])
            explanations.add(
                query=pd.DataFrame([[text, original_label]], columns=["text", "label"]),
                cfs=pd.DataFrame([(e[0], e[2]) for e in examples[:max_number_examples]], columns=["text", "label"])
            )
        return explanations

    def _explain_question_answering(self, X: Text, max_number_examples: int = 5, **kwargs):
        sep = kwargs.get("sep", "[SEP]")
        explanations = CFExplanation()
        tokenizer = X.tokenizer

        for x in X:
            # The answer obtained by the model
            res = self.predict_function(x)
            if not isinstance(res, str):
                res = res[0]
            # Get the context and question
            context, question = x.split(sep, 1)[0]
            perturb_questions = self._perturb(question, **kwargs)
            # Perturb the question
            flips, non_flips = [], []
            for perturbation in perturb_questions:
                if perturbation == question:
                    continue
                ce_res = self.predict_function(Text(f"{context}{sep}{perturbation}", tokenizer=tokenizer))
                if not isinstance(ce_res, str):
                    ce_res = ce_res[0]
                # Sort the results by a simple distance function
                distance = self._distance(
                    Text(question, tokenizer=tokenizer), Text(perturbation, tokenizer=tokenizer))
                if ce_res == res:
                    non_flips.append((perturbation, ce_res, distance))
                else:
                    flips.append((perturbation, ce_res, distance))

            examples = sorted(flips, key=lambda z: z[-1]) + sorted(non_flips, key=lambda z: z[-1])
            cfs = [[q, r] for q, r, _ in examples[:max_number_examples]]
            explanations.add(
                query=pd.DataFrame([[question, res]], columns=["question", "answer"]),
                cfs=pd.DataFrame(cfs, columns=["question", "answer"]) if cfs else None
            )
        return explanations

    def explain(self, X: Text, max_number_examples: int = 5, **kwargs) -> CFExplanation:
        """
        Generates the counterfactual explanations for the input instances.

        :param X: A batch of input instances. For question-answering tasks, each instance in ``X``
            has format `[context] [SEP] [question]`, i.e., concatenating the context and question
            with seperator `[SEP]`.
        :param max_number_examples: The maximum number of the generated counterfactual
            examples for each input instance.
        :param kwargs: Additional parameters for `polyjuice.Polyjuice`, e.g., "ce_type" - the perturb type
            ("perturb" or "blank").
        :return: The explanations for all the input instances.
        """
        if self.mode == "classification":
            return self._explain_classification(X=X, max_number_examples=max_number_examples, **kwargs)
        else:
            return self._explain_question_answering(X=X, max_number_examples=max_number_examples, **kwargs)
