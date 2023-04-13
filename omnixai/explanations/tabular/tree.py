#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Explanations for tree-based models.
"""
import warnings
import pandas as pd
from sklearn.tree import plot_tree, _tree
from ..base import ExplanationBase, DashFigure


class TreeExplanation(ExplanationBase):
    """
    The class for explanation results for tree-based models. The results are
    stored in a dict with the following format: `{"model": the trained tree model,
    "tree": the binary tree extracted from the model, "feature_names": A list of
    feature names, "class_names": A list of class names, "path": The decision paths
    for a batch of instances}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = None
        self.is_local = False

    def __repr__(self):
        return repr(self.explanations)

    def add_global(self, model, feature_names, class_names):
        """
        Adds the global explanations, i.e., the whole tree structure.

        :param model: The tree model.
        :param feature_names: The feature column names.
        :param class_names: The class names.
        """
        self.explanations = {
            "model": model,
            "tree": _export_dict(model, feature_names),
            "feature_names": feature_names,
            "class_names": class_names,
        }
        self.is_local = False

    def add_local(self, model, decision_paths, node_indicator, feature_names, class_names):
        """
        Adds the local explanations, i.e., decision paths.

        :param model: The tree model.
        :param decision_paths: The decision paths for one instance.
        :param node_indicator: The node indicator.
        :param feature_names: The feature column names.
        :param class_names: The class names.
        """
        self.explanations = {
            "model": model,
            "path": decision_paths,
            "node_indicator": node_indicator,
            "feature_names": feature_names,
            "class_names": class_names,
        }
        self.is_local = True

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of the instance, e.g., it will plot the
            first explanation result when ``index = 0``.When it is None,
            this method return all the explanations.
        :return: The explanations for one specific instance
            or all the explanations for all the instances.
        """
        if index is None or not self.is_local:
            return self.explanations
        else:
            return self.explanations["path"][index]

    def plot(self, index=None, figsize=(15, 10), fontsize=10, **kwargs):
        """
        Returns a matplotlib figure showing the explanations.

        :param index: The index of an explanation result stored in ``TreeExplanation``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots the explanations of the first 5 instances.
        :param figsize: The figure size.
        :param fontsize: The font size of texts.
        :return: A list of matplotlib figures plotting the tree and the decision paths.
        """
        import matplotlib.pyplot as plt

        if not self.is_local:
            fig = plt.figure(figsize=figsize)
            plot_tree(
                decision_tree=self.explanations["model"],
                feature_names=self.explanations["feature_names"],
                class_names=self.explanations["class_names"],
                proportion=True,
                rounded=True,
                filled=True,
                fontsize=fontsize,
            )
            return fig
        else:
            if index is None:
                node_indicator = self.explanations["node_indicator"]
            else:
                node_indicator = self.explanations["node_indicator"][index : index + 1]
            if node_indicator.shape[0] > 5:
                warnings.warn(
                    f"There are too many instances ({node_indicator.shape[0]} > 5), "
                    f"so only the first 5 instances are plotted."
                )
                node_indicator = node_indicator[:5]

            figures = []
            for i in range(node_indicator.shape[0]):
                fig = plt.figure(figsize=figsize)
                tree = plot_tree(
                    decision_tree=self.explanations["model"],
                    feature_names=self.explanations["feature_names"],
                    class_names=self.explanations["class_names"],
                    proportion=True,
                    rounded=True,
                    filled=True,
                    fontsize=fontsize,
                )
                for k in range(len(tree)):
                    if k not in node_indicator[i].indices:
                        plt.setp(tree[k], visible=False)
                figures.append(fig)
            return figures

    def _path_df(self, index, **kwargs):
        assert index < self.explanations["node_indicator"].shape[0], "`index` is out of bound."
        node_indicator = self.explanations["node_indicator"][index]
        tree = plot_tree(
            decision_tree=self.explanations["model"],
            feature_names=self.explanations["feature_names"],
            class_names=self.explanations["class_names"],
        )
        path = []
        for k in node_indicator.indices:
            path.append(tree[k]._text)
        df = pd.DataFrame([path], columns=[f"Node {i + 1}" for i in range(len(path))])
        return df

    def plotly_plot(self, index=0, **kwargs):
        """
        Returns a plotly dash figure showing decision paths.

        :param index: The index of the instance which cannot be None, e.g.,
            it will plot the first explanation result when ``index = 0``.
        :return: A plotly dash figure plotting decision paths.
        """
        from dash import dash_table

        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        df = self._path_df(index, **kwargs)
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i, "editable": False} for i in df.columns],
            data=df.to_dict("records"),
            style_cell={"whiteSpace": "pre-line"},
        )
        return DashFigure(table)

    def ipython_plot(self, index=0, figsize=(15, 10), fontsize=10, **kwargs):
        """
        Plots decision paths in IPython.

        :param index: The index of the instance which cannot be None, e.g.,
            it will plot the first explanation result when ``index = 0``.
        :param figsize: The figure size.
        :param fontsize: The font size of texts.
        """
        import matplotlib.pyplot as plt

        node_indicator = self.explanations["node_indicator"][index]

        fig = plt.figure(figsize=figsize)
        tree = plot_tree(
            decision_tree=self.explanations["model"],
            feature_names=self.explanations["feature_names"],
            class_names=self.explanations["class_names"],
            proportion=True,
            rounded=True,
            filled=True,
            fontsize=fontsize,
        )
        for k in range(len(tree)):
            if k not in node_indicator.indices:
                plt.setp(tree[k], visible=False)

    def to_json(self):
        raise RuntimeError("`TreeExplanation` cannot be converted into JSON format.")

    @classmethod
    def from_dict(cls, d):
        raise RuntimeError("`TreeExplanation` does not support `from_dict`.")


def _export_dict(tree, feature_names=None, max_depth=None):
    """
    https://github.com/scikit-learn/scikit-learn/blob/79bdc8f711d0af225ed6be9fdb708cea9f98a910/sklearn/tree/export.py
    Export a decision tree in dict format.

    :param tree: A decision tree classifier
    :param feature_names: Names of each of the features.
    :param max_depth: The maximum depth of the representation. If None, the tree is fully generated.
    :return: A dictionary of the tree structure.
    """
    tree_ = tree.tree_

    # i is the element in the tree_ to create a dict for
    def recur(i, depth=0):
        if max_depth is not None and depth > max_depth:
            return None
        if i == _tree.TREE_LEAF:
            return None

        feature = int(tree_.feature[i])
        threshold = float(tree_.threshold[i])
        if feature == _tree.TREE_UNDEFINED:
            feature = None
            threshold = None
            value = [map(int, l) for l in tree_.value[i].tolist()]
        else:
            value = None
            if feature_names:
                feature = feature_names[feature]

        return {
            "feature": feature,
            "threshold": threshold,
            "impurity": float(tree_.impurity[i]),
            "n_node_samples": int(tree_.n_node_samples[i]),
            "left": recur(tree_.children_left[i], depth + 1),
            "right": recur(tree_.children_right[i], depth + 1),
            "value": value,
        }

    return recur(0)
