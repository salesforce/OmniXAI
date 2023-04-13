#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Counterfactual explanations.
"""
import numpy as np
import pandas as pd
from ..base import ExplanationBase, DashFigure


class CFExplanation(ExplanationBase):
    """
    The class for counterfactual explanation results. It uses a list to store
    the generated counterfactual examples. Each item in the list is a dict with
    the following format: `{"query": the original input instance, "counterfactual":
    the generated counterfactual examples}`. Both "query" and "counterfactual" are
    pandas dataframes with an additional column "label" which stores the predicted
    labels of these instances.
    """

    def __init__(self, explanations=None):
        super().__init__()
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return CFExplanation(explanations=[self.explanations[i]])

    def add(self, query, cfs, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param query: The instance to explain.
        :param cfs: The generated counterfactual examples.
        :param kwargs: Additional information to store.
        """
        e = {"query": query, "counterfactual": cfs}
        e.update(kwargs)
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated counterfactual explanations.

        :param index: The index of an explanation result stored in ``CFExplanation``.
            When it is None, it returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or all the explanations for all the instances (a list). Each dict has
            the following format: `{"query": the original input instance, "counterfactual":
            the generated counterfactual examples}`. Both "query" and "counterfactual" are
            pandas dataframes with an additional column "label" which stores the predicted
            labels of these instances.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    @staticmethod
    def _get_changed_columns(query, cfs):
        """
        Gets the differences between the instance and the generated counterfactual examples.

        :param query: The input instance.
        :param cfs: The counterfactual examples.
        :return: The feature columns that have been changed in ``cfs``.
        :rtype: List
        """
        columns = []
        for col in query.columns:
            u = query[[col]].values[0]
            for val in cfs[[col]].values:
                if val != u:
                    columns.append(col)
                    break
        return columns

    @staticmethod
    def _plot(plt, index, query, cfs, context=None, font_size=10, bar_width=0.4):
        """
        Plots a table showing the generated counterfactual examples.
        """
        df = pd.concat([query, cfs], axis=0)
        rows = [f"Instance {index}"] + [f"CF {k}" for k in range(1, df.shape[0])]
        counts = np.zeros(len(df.columns))
        for i in range(df.shape[1] - 1):
            for j in range(1, df.shape[0]):
                counts[i] += int(df.values[0, i] != df.values[j, i])
        # Context
        if context is not None:
            df = pd.concat([context, df], axis=0)
            rows = [f"Context {k + 1}" for k in range(context.shape[0])] + rows

        plt.bar(np.arange(len(df.columns)) + 0.5, counts, bar_width)
        table = plt.table(cellText=df.values, rowLabels=rows, colLabels=df.columns, loc="bottom")
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.ylabel("The number of feature changes")
        plt.yticks(np.arange(max(counts)))
        plt.xticks([])
        plt.title(f"Counterfactual Examples")
        plt.grid()

        # Highlight the differences between the query and the CF examples
        for k in range(df.shape[1]):
            table[(0, k)].set_facecolor("#C5C5C5")
            for i in range(1, df.shape[0] - cfs.shape[0] + 1):
                table[(i, k)].set_facecolor("#E2DED0")
        for j in range(df.shape[0] - cfs.shape[0], df.shape[0]):
            for k in range(df.shape[1] - 1):
                if query.values[0][k] != df.values[j][k]:
                    table[(j + 1, k)].set_facecolor("#56b5fd")

        # Change the font size if `font_size` is set
        if font_size is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

    def plot(self, index=None, class_names=None, font_size=10, **kwargs):
        """
        Returns a list of matplotlib figures showing the explanations of
        one or the first 5 instances.

        :param index: The index of an explanation result stored in ``CFExplanation``. For
            example, it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots the explanations of the first 5 instances.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param font_size: The font size of table entries.
        :return: A list of matplotlib figures plotting counterfactual examples.
        """
        import warnings
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if len(indices) > 5:
            warnings.warn(
                f"There are too many instances ({len(indices)} > 5), " f"so only the first 5 instances are plotted."
            )
            indices = indices[:5]

        figures = []
        for i, index in enumerate(indices):
            fig = plt.figure()
            figures.append(fig)
            exp = explanations[index]
            if exp["counterfactual"] is None:
                continue
            if len(exp["query"].columns) > 5:
                columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
            else:
                columns = exp["query"].columns
            query, cfs = exp["query"][columns], exp["counterfactual"][columns]
            context = exp["context"][columns] if "context" in exp else None

            dfs = [query, cfs, context]
            if class_names is not None:
                for df in dfs:
                    if df is not None:
                        df["label"] = [class_names[label] for label in df["label"].values]
            self._plot(plt, index, query, cfs, context, font_size)
        return figures

    def plotly_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots the generated counterfactual examples in Dash.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :return: A plotly dash figure showing the counterfactual examples.
        """
        assert index is not None, \
            "`index` cannot be None for `plotly_plot`. Please specify the instance index."

        exp = self.explanations[index]
        context = exp["context"] if "context" in exp else None
        if exp["counterfactual"] is None:
            return DashFigure(self._plotly_table(exp["query"], None, context))

        if len(exp["query"].columns) > 5 and not kwargs.get("show_all_columns", False):
            columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
        else:
            columns = exp["query"].columns
        query, cfs = exp["query"][columns], exp["counterfactual"][columns]
        context = context[columns] if context is not None else None
        dfs = [query, cfs, context]

        if class_names is not None:
            for df in dfs:
                if df is not None:
                    df["label"] = [class_names[label] for label in df["label"].values]
        return DashFigure(self._plotly_table(query, cfs, context))

    def ipython_plot(self, index=0, class_names=None, **kwargs):
        """
        Plots the generated counterfactual examples in IPython.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        """
        assert index is not None, \
            "`index` cannot be None for `ipython_plot`. Please specify the instance index."
        import plotly
        import plotly.figure_factory as ff

        exp = self.explanations[index]
        if exp["counterfactual"] is None:
            return None
        if len(exp["query"].columns) > 5 and not kwargs.get("show_all_columns", False):
            columns = self._get_changed_columns(exp["query"], exp["counterfactual"])
        else:
            columns = exp["query"].columns
        query, cfs = exp["query"][columns], exp["counterfactual"][columns]
        context = exp["context"][columns] if "context" in exp else None

        names, dfs = [], []
        if context is not None:
            dfs.append(context)
            names += [f"Context" for _ in range(context.shape[0])]
        dfs.append(query)
        names.append("Query")
        if cfs is not None:
            dfs.append(cfs)
            names += [f"CF {i + 1}" for i in range(cfs.shape[0])]

        df = pd.concat(dfs, axis=0)
        if class_names is not None:
            df["label"] = [class_names[label] for label in df["label"].values]
        df.insert(loc=0, column="#", value=names)
        plotly.offline.iplot(ff.create_table(df.round(4)))

    @staticmethod
    def _plotly_table(query, cfs, context):
        """
        Plots a table showing the generated counterfactual examples.
        """
        from dash import dash_table
        feature_columns = query.columns
        columns = [{"name": "#", "id": "#"}] + [{"name": c, "id": c} for c in feature_columns]
        context_size = context.shape[0] if context is not None else 0
        highlight_row_offset = query.shape[0] + context_size + 1

        highlights = []
        query = query.values
        if cfs is not None:
            cfs = cfs.values
            for i, cf in enumerate(cfs):
                for j in range(len(cf) - 1):
                    if query[0][j] != cf[j]:
                        highlights.append((i, j))

        data = []
        # Context row
        if context is not None:
            for x in context.values:
                row = {"#": "Context"}
                row.update({c: d for c, d in zip(feature_columns, x)})
                data.append(row)
        # Query row
        for x in query:
            row = {"#": "Query"}
            row.update({c: d for c, d in zip(feature_columns, x)})
            data.append(row)
        # Separator
        row = {"#": "-"}
        row.update({c: "-" for c in feature_columns})
        data.append(row)
        # CF example row
        if cfs is not None:
            for i, x in enumerate(cfs):
                row = {"#": f"CF {i + 1}"}
                row.update({c: d for c, d in zip(feature_columns, x)})
                data.append(row)

        style_data_conditional = [{"if": {"row_index": 0}, "backgroundColor": "rgb(240, 240, 240)"}]
        for i, j in highlights:
            c = feature_columns[j]
            cond = {
                "if": {"filter_query": "{{{0}}} != ''".format(c),
                       "column_id": c, "row_index": i + highlight_row_offset},
                "backgroundColor": "dodgerblue",
            }
            style_data_conditional.append(cond)

        table = dash_table.DataTable(
            id="table",
            columns=columns,
            data=data,
            style_header_conditional=[{"textAlign": "center"}],
            style_cell_conditional=[{"textAlign": "center"}],
            style_data_conditional=style_data_conditional,
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_table={"overflowX": "scroll", "overflowY": "auto", "height": "260px"},
        )
        return table

    @classmethod
    def from_dict(cls, d):
        explanations = []
        for e in d["explanations"]:
            e["query"] = pd.DataFrame.from_dict(e["query"])
            e["counterfactual"] = pd.DataFrame.from_dict(e["counterfactual"])
            explanations.append(e)
        exp = CFExplanation()
        exp.explanations = explanations
        return exp
