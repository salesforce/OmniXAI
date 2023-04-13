#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Validity ranking explanation explanations.
"""
import numpy as np
from ...explanations.base import ExplanationBase, DashFigure


class ValidityRankingExplanation(ExplanationBase):
    """
    The class for validity ranking explanation results.
    """

    def __init__(self, explanations=None):
        super().__init__()
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return ValidityRankingExplanation(explanations=[self.explanations[i]])

    def add(self, query, df, top_features, validity, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param query: The instance to explain.
        :param df: The dataframe of input query item features.
        :param top_features: The features that explain the ranking
        :param validity: The validity metrics for the top features.
        :param kwargs: Additional information to store.
        """
        e = {
            "query": query,
            "item": df,
            "top_features": top_features,
            "validity": validity
        }
        e.update(kwargs)
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``ValidityRankingExplanation``.
            When it is None, it returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or all the explanations for all the instances (a list). Each dict has
            the following format: `{"query": the original input instance, "item":
            The dataframe of input query item features, "top_features": The top features that
            explain the ranking, "validity": The validity metrics for the top features.}`.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    @staticmethod
    def _plot(plt, df, top_features, validity, font_size, bar_width=0.4):
        df["#Rank"] = validity["Ranks"]
        columns = ValidityRankingExplanation.rearrange_columns(df, top_features)
        df = df[columns]

        counts = np.zeros(len(df.columns))
        for i, f in enumerate(df.columns):
            if f in top_features:
                counts[i] = top_features[f]

        plt.bar(np.arange(len(df.columns)) + 0.5, counts, bar_width)
        table = plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc="bottom")
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.ylabel("The validity metrics")
        plt.yticks(np.arange(max(counts)))
        plt.xticks([])
        plt.grid()

        for k in range(df.shape[1]):
            table[(0, k)].set_facecolor("#C5C5C5")
        for j in range(df.shape[0]):
            for k, f in enumerate(df.columns):
                if f in top_features:
                    table[(j + 1, k)].set_facecolor("#56b5fd")

        # Change the font size if `font_size` is set
        if font_size is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)

    def plot(self, index=0, font_size=8, **kwargs):
        """
        Returns a matplotlib figure showing the explanations.

        :param index: The index of an explanation result stored in ``ValidityRankingExplanation``.
        :param font_size: The font size of table entries.
        :return: Matplotlib figure plotting the most important features followed by remaining features.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        fig = plt.figure()
        self._plot(
            plt,
            df=explanations["item"],
            top_features=explanations["top_features"],
            validity=explanations["validity"],
            font_size=font_size
        )
        return fig

    def plotly_plot(self, index=0, **kwargs):
        """
        Plots the document features and explainable features in Dash.

        :param index: The index of an explanation result stored in ``ValidityRankingExplanation``.
        :return: A plotly dash figure showing the important features followed by remaining features
        """
        explanations = self.get_explanations(index)
        df = explanations["item"]
        top_features = explanations["top_features"].keys()
        validity = explanations["validity"]
        return DashFigure(self._plotly_table(df, top_features, validity))

    def ipython_plot(self, index=0, **kwargs):
        """
        Plots a table for ipython showing the important features followed by the remaining features.
        """
        import plotly

        fig = self._ipython_figure(index)
        plotly.offline.iplot(fig)

    def _ipython_figure(self, index):
        import plotly.figure_factory as ff

        exp = self.get_explanations(index)
        df = exp["item"]
        df["#Rank"] = exp["validity"]["Ranks"]
        top_features = exp["top_features"].keys()
        feature_columns = self.rearrange_columns(df, top_features)
        opacity = 1 / (len(top_features) + 1)
        a = 0

        fig = ff.create_table(
            df[feature_columns].round(4),
            colorscale='blues',
            font_colors=['#000000']
        )
        colorscale = []
        for i in range(0, len(top_features)):
            colorscale.append(1 - a)
            a += opacity

        z = fig['data'][0]['z']
        for i in range(len(feature_columns)):
            for j in range(len(z)):
                z[j][i] = 0
        for i in range(1, len(top_features) + 1):
            for j in range(len(z)):
                z[j][i] = colorscale[i - 1]
        return fig

    def _plotly_table(self, df, top_features, validity):
        """
        Plots a dash table showing the important features followed by the remaining features.
        """
        from dash import dash_table
        df["#Rank"] = validity["Ranks"]
        feature_columns = self.rearrange_columns(df, top_features)
        data = [{c: row[c] for c in feature_columns} for idx, row in df.iterrows()]

        style_data_conditional = [{"if": {"row_index": 0}, "backgroundColor": "rgb(240, 240, 240)"}]
        opacity = 1 / (len(top_features) + 1)
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
            columns=[{"name": c, "id": c} for c in feature_columns],
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

    @classmethod
    def from_dict(cls, d):
        import pandas as pd
        explanations = []
        for e in d["explanations"]:
            e["item"] = pd.DataFrame.from_dict(e["item"])
            explanations.append(e)
        exp = ValidityRankingExplanation()
        exp.explanations = explanations
        return exp
