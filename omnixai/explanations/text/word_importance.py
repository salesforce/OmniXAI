#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Word/token importance explanations for NLP tasks.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure


class WordImportance(ExplanationBase):
    """
    The class for word/token importance explanations. It uses a list to store
    the word/token importance scores of the input instances. Each item in the list
    is a dict with the following format `{"instance": the input instance, "tokens":
    a list of tokens, "scores": a list of feature importance scores}`.
    If the task is `classification`, the dict has an additional entry
    `{"target_label": the predicted label of the input instance}`.
    """

    def __init__(self, mode, explanations=None):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        :param explanations: The explanation results for initializing `WordImportance`,
            which is optional.
        """
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def add(self, instance, target_label, tokens, importance_scores, sort=False, **kwargs):
        """
        Adds the generated explanation of one instance.

        :param instance: The instance to be explained.
        :param target_label: The label to be explained, which is ignored for regression.
        :param tokens: The list of the words/tokens in the explanation.
        :param importance_scores: The list of the corresponding word/token importance scores.
        :param sort: Sort the features based on the importance scores if it is True.
        """
        scores = list(zip(tokens, importance_scores))
        if sort:
            scores = sorted(scores, key=lambda x: abs(x[-1]), reverse=True)
        e = {"instance": instance, "tokens": [s[0] for s in scores], "scores": [s[1] for s in scores]}
        e.update(kwargs)
        if self.mode == "classification":
            e["target_label"] = target_label
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``WordImportance``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"instance": the input instance, "tokens":
            a list of tokens, "scores": a list of feature importance scores}`.
            If the task is `classification`, the dict has an additional entry
            `{"target_label": the predicted label of the input instance}`.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, class_names=None, num_tokens_per_class=5, max_num_subplots=4, **kwargs):
        """
        Returns a matplotlib figure showing the explanations.

        :param index: The index of an explanation result stored in ``WordImportance``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it shows a figure with ``max_num_subplots`` subplots
            where each subplot plots the word/token importance scores for one instance.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_tokens_per_class: The maximum number of tokens shown in the figure for each class.
        :param max_num_subplots: The maximum number of subplots in the figure.
        :return: A matplotlib figure plotting word/token importance scores.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if max_num_subplots is not None:
            indices = indices[:max_num_subplots]

        num_rows = int(np.round(np.sqrt(len(indices))))
        num_cols = int(np.ceil(len(indices) / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        for i, index in enumerate(indices):
            exp = explanations[index]
            feat_scores = sorted(list(zip(exp["tokens"], exp["scores"])), key=lambda x: abs(x[1]))
            sorted_scores = sorted([s for f, s in feat_scores])
            n = min(len(sorted_scores) - 1, num_tokens_per_class + 1)
            pos_threshold = max(sorted_scores[-n], 0)
            neg_threshold = min(sorted_scores[n], 0)
            fnames = [f for f, s in feat_scores if s > pos_threshold or s < neg_threshold]
            scores = [s for f, s in feat_scores if s > pos_threshold or s < neg_threshold]
            colors = ["green" if x > 0 else "red" for x in scores]
            positions = np.arange(len(scores)) + 0.5

            row, col = divmod(i, num_cols)
            plt.sca(axes[row, col])
            plt.barh(positions, scores, align="center", color=colors)
            plt.yticks(positions, fnames)
            if self.mode == "classification":
                class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
                plt.title(f"Instance {index}: Class {class_name}")
            else:
                plt.title(f"Instance {index}")
            plt.grid(axis="x")
        return fig

    def plotly_plot(self, index=None, class_names=None, num_tokens_per_class=5, max_length=512, **kwargs):
        """
        Returns a plotly dash figure plotting the explanations.

        :param index: The index of the instance, e.g., it will plot the first
            explanation result when ``index = 0``.. For plotting all the results,
            set ``index`` to `None`.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_tokens_per_class: The maximum number of tokens shown in the figure for each class.
        :param max_length: The maximum number of tokens to show.
        :return: A plotly dash figure plotting word/token importance scores.
        """
        import plotly.express as px
        from dash import html, dcc

        if index is not None:
            indices = [index]
        else:
            indices = range(len(self.explanations))

        figures = []
        for index in indices:
            exp = self.explanations[index]
            if self.mode == "classification":
                class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
                title = f"Instance {index}: Class {class_name}"
            else:
                title = f"Instance {index}"

            feat_scores = sorted(list(zip(exp["tokens"], exp["scores"])), key=lambda x: abs(x[1]))
            sorted_scores = sorted([s for f, s in feat_scores])
            n = min(len(sorted_scores) - 1, num_tokens_per_class + 1)
            pos_threshold = max(sorted_scores[-n], 0)
            neg_threshold = min(sorted_scores[n], 0)
            fnames = [f for f, s in feat_scores if s > pos_threshold or s < neg_threshold]
            scores = [s for f, s in feat_scores if s > pos_threshold or s < neg_threshold]
            # A plotly figure shows the top five features
            fig = px.bar(
                y=fnames,
                x=scores,
                orientation="h",
                color=[s > 0 for s in scores],
                labels={"color": "Positive", "x": "Importance scores", "y": "Features"},
                title=title,
                color_discrete_map={True: "#008B8B", False: "#DC143C"},
            )

            # Highlight texts
            tokens, scores = exp["tokens"], np.array(exp["scores"])
            bound = max(abs(np.max(scores)), abs(np.min(scores)))
            scores = scores / (bound + 1e-8)

            n = len(tokens)
            children = []
            for i, (token, s) in enumerate(zip(tokens, scores)):
                if i >= max_length:
                    children.append(html.Span("..."))
                    break
                if s > 0:
                    r, g, b = 128 - int(64 * s), int(128 * s) + 127, 128 - int(64 * s)
                else:
                    r, g, b = int(-128 * s) + 127, 128 + int(64 * s), 128 + int(64 * s)
                token = str(token).strip()
                if i < n - 1:
                    token += " "
                children.append(html.Span(token, style={"color": f"rgb({r},{g},{b})"}))
            figures.append(html.Div([dcc.Graph(figure=fig), html.Div(children=children)]))
        return DashFigure(figures)

    def ipython_plot(self, index=None, class_names=None, max_length=512, **kwargs):
        """
        Plots word/token importance scores in IPython.

        :param index: The index of the instance, e.g., it will plot the first
            explanation result when ``index = 0``.. For plotting all the results,
            set ``index`` to `None`.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param max_length: The maximum number of tokens to show.
        """
        from IPython.display import HTML, display

        if index is not None:
            indices = [index]
        else:
            indices = range(len(self.explanations))

        content = []
        for index in indices:
            exp = self.explanations[index]
            if self.mode == "classification":
                class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
                title = f"Instance {index}: Class {class_name}"
            else:
                title = f"Instance {index}"
            content.append(f"<div>{title}</div>")

            tokens, scores = exp["tokens"], np.array(exp["scores"])
            bound = max(abs(np.max(scores)), abs(np.min(scores)))
            scores = scores / (bound + 1e-8)

            n = len(tokens)
            html_text = ""
            for i, (token, s) in enumerate(zip(tokens, scores)):
                if i >= max_length:
                    html_text += f"<span>...</span>"
                    break
                if s > 0:
                    r, g, b = 128 - int(64 * s), int(128 * s) + 127, 128 - int(64 * s)
                else:
                    r, g, b = int(-128 * s) + 127, 128 + int(64 * s), 128 + int(64 * s)
                token = str(token).strip()
                if i < n - 1:
                    token += " "
                html_text += f"<span style='color:rgb({r},{g},{b})'>{token}</span>"
            content.append(f"<div>{html_text}</div><br>")
        display(HTML("\n".join(content)))

    @classmethod
    def from_dict(cls, d):
        return WordImportance(mode=d["mode"], explanations=d["explanations"])
