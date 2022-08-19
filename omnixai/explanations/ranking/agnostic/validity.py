#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Counterfactual explanations.
"""
import numpy as np
from ...base import ExplanationBase, DashFigure


class ValidExplanation(ExplanationBase):
    """
    The class for ranking explanation results.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def __repr__(self):
        return repr(self.explanations)

    def set(self, query, df, top_features, validity, **kwargs):
        """
        Sets the generated explanation corresponding to one instance.

        :param query: The instance to explain.
        :param df: The dataframe of input query document features.
        :param top_features: The features that explain the ranking
        :param validity: The validity metric for the top features.
        :param kwargs: Additional information to store.
        """
        e = {
            "query": query,
             "docs": df,
             "top_features": top_features,
             "validity": validity
        }
        e.update(kwargs)
        self.explanations = e

    def get_explanations(self):
        """
        Gets the generated counterfactual explanations.

        :return: The explanation for one specific instance (a dict)
            or all the explanations for all the instances (a list). Each dict has
            the following format: `{"query": the original input instance, "counterfactual":
            the generated counterfactual examples}`. Both "query" and "counterfactual" are
            pandas dataframes with an additional column "label" which stores the predicted
            labels of these instances.
        :rtype: Union[Dict, List]
        """
        return self.explanations

    @staticmethod
    def _plot(plt, df, font_size, bar_width=0.4):
        """
        Plots a table showing the generated counterfactual examples.
        """

        counts = np.zeros(len(df.columns))
        for i in range(df.shape[1] - 1):
            for j in range(1, df.shape[0]):
                counts[i] += int(df.values[0, i] != df.values[j, i])

        plt.bar(np.arange(len(df.columns)) + 0.5, counts, bar_width)
        table = plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc="bottom")
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.ylabel("The number of feature changes")
        plt.yticks(np.arange(max(counts)))
        plt.xticks([])
        plt.grid()

        # Highlight the differences between the query and the CF examples
        for k in range(df.shape[1]):
            table[(0, k)].set_facecolor("#C5C5C5")
            table[(1, k)].set_facecolor("#E2DED0")
        for j in range(1, df.shape[0]):
            for k in range(df.shape[1] - 1):
                if df.values[0][k] != df.values[j][k]:
                    table[(j + 1, k)].set_facecolor("#56b5fd")

        # Change the font size if `font_size` is set
        if font_size is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

    def plot(self, font_size=10, **kwargs):
        """
        Returns a list of matplotlib figures showing the explanations of
        one or the first 5 instances.

        :param font_size: The font size of table entries.
        :return: Matplotlib figure plotting the most important features followed by remaning features
        """
        import warnings
        import matplotlib.pyplot as plt

        explanations = self.get_explanations()

        fig = plt.figure()

        self._plot(plt, explanations["docs"], font_size)
        return fig

    def plotly_plot(self, **kwargs):
        """
        Plots the document features and explainable features in Dash.
        :return: A plotly dash figure showing the important features followed by remaining features
        """

        df = self.explanations["docs"]
        top_features = self.explanations["top_features"].keys()
        query = self.explanations["query"]
        validity = self.explanations["validity"]
        return DashFigure(self._plotly_table(df, top_features, query, validity))

    def ipython_fig(self):
        """
            Returns the ipython figure
        """

        import plotly.figure_factory as ff

        exp = self.explanations

        df = exp["docs"]
        df["#Rank"] = exp["validity"]["Ranks"]
        top_features = exp["top_features"].keys()
        feature_columns = self.rearrange_columns(df, top_features)
        opacity = 1 / (len(top_features) + 1)
        a = 0

        fig = ff.create_table(df[feature_columns].round(4), colorscale='blues', font_colors=['#000000'])

        colorscale = []
        for i in range(0, len(top_features)):
            colorscale.append(a)
            a += opacity

        z = fig['data'][0]['z']
        for i in range(len(feature_columns)):
            for j in range(len(z)):
                z[j][i] = 0
        for i in range(1, len(top_features) + 1):
            for j in range(len(z)):
                z[j][i] = colorscale[i - 1]

        return fig

    def ipython_plot(self, **kwargs):
        """
        Plots a table for ipython showing the important features followed by the remaining features.
        """
        import plotly

        fig = self.ipython_fig()
        plotly.offline.iplot(fig)

    def _plotly_table(self, df, top_features, query, validity):
        """
        Plots a dash table showing the important features followed by the remaining features.
        """
        from dash import dash_table
        df["#Rank"] = validity["Ranks"]
        feature_columns = self.rearrange_columns(df, top_features)
        columns = [{"name": c, "id": c} for c in feature_columns]
        if query:
            columns = [{"name": query, "id": query}] + columns

        data = []
        for idx, row in df.iterrows():
            data.append({c: row[c] for c in feature_columns})

        style_data_conditional = [{"if": {"row_index": 0}, "backgroundColor": "rgb(240, 240, 240)"}]
        opacity = 1/(len(top_features)+1)
        a = 0
        for f in top_features:
            cond = {
                "if": {"column_id": f},
                "backgroundColor": f"rgba(13, 146, 238, {abs(1 - a)})"
            }
            a += opacity
            style_data_conditional.append(cond)

        table = dash_table.DataTable(
            id="table",
            columns=columns,
            data=data,
            style_header_conditional=[{"textAlign": "center"}],
            style_cell_conditional=[{"textAlign": "center"}],
            style_data_conditional=style_data_conditional,
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_table={"overflowX": "scroll"},
        )
        return table

    @staticmethod
    def rearrange_columns(df, top_features):
        return ["#Rank"] + list(top_features) + \
               [c for c in df.columns if c not in top_features and c != "#Rank"]
