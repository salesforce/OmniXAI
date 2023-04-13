#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Counterfactual explanations for time series tasks.
"""
import warnings
import numpy as np
from ..base import ExplanationBase, DashFigure


class CFExplanation(ExplanationBase):
    """
    The class for counterfactual explanations for time series tasks. It uses a list to store
    the counterfactual examples of the input instances. Each item in the list
    is a dict with the following format `{"query": the input instance, "counterfactual":
    the generated counterfactual example}`. Both "query" and "counterfactual" are
    pandas dataframes.
    """

    def __init__(self):
        super().__init__()
        self.explanations = []

    def __repr__(self):
        return repr(self.explanations)

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
            pandas dataframes.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, max_num_variables_to_plot=25, **kwargs):
        """
        Plots counterfactual examples for time series data.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots all the explanations.
        :param max_num_variables_to_plot: The maximum number of variables to plot in the figure.
        :return: A matplotlib figure plotting counterfactual examples.
        """
        import matplotlib.pyplot as plt

        if len(self.explanations) == 0:
            return None
        figures = []
        explanations = [self.explanations[index]] if index is not None \
            else self.explanations

        for exp in explanations:
            ts, cf = exp["query"], exp["counterfactual"]
            num_variables = ts.shape[1]
            if num_variables > max_num_variables_to_plot:
                warnings.warn("The number of variables in the time series exceeds "
                              "the maximum number of variables to plot.")

            n = min(num_variables, max_num_variables_to_plot)
            num_rows = int(np.round(np.sqrt(n)))
            num_cols = int(np.ceil(n / num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

            for i in range(n):
                row, col = divmod(i, num_cols)
                plt.sca(axes[row, col])
                # Plot the original time series
                left_ax = axes[row, col]
                ts_a = ts[[ts.columns[i]]]
                timestamps = [str(v) for v in ts_a.index.values]
                left_ax.plot(timestamps, ts_a.values.flatten(), color='k')
                left_ax.set_xticklabels(left_ax.get_xticks(), rotation=45)
                # Plot the counterfactual example
                if cf is not None:
                    right_ax = axes[row, col].twinx()
                    ts_b = cf[[cf.columns[i]]]
                    right_ax.plot(timestamps, ts_b.values.flatten(), color='r', label="cf")
                plt.title(f"{ts.columns[i]}")
                plt.legend()
                plt.grid()
            figures.append(fig)
        return figures

    def _plotly_figure(self, index, **kwargs):
        import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        exp = self.explanations[index]
        traces = []
        color_list = plotly.colors.qualitative.Dark24
        ts, cf = exp["query"], exp["counterfactual"]
        for i in range(ts.shape[1]):
            v = ts[[ts.columns[i]]]
            color = color_list[i % len(color_list)]
            # Original time series data
            traces.append(go.Scatter(
                name=ts.columns[i],
                x=v.index,
                y=v.values.flatten(),
                mode="lines",
                line=dict(color=color)
            ))
            # Counterfactual examples
            if cf is not None:
                v = cf[[ts.columns[i]]]
                color = color_list[i % len(color_list)]
                traces.append(go.Scatter(
                    name=f"{cf.columns[i]}_cf",
                    x=v.index,
                    y=v.values.flatten(),
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                ))

        layout = dict(
            showlegend=True,
            xaxis=dict(
                title="Time",
                type="date",
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                )
            ),
        )
        fig = make_subplots(figure=go.Figure(layout=layout))
        fig.update_yaxes(title_text="Timeseries")
        for trace in traces:
            fig.add_trace(trace)
        return fig

    def plotly_plot(self, index=0, **kwargs):
        """
        Plots counterfactual examples for one specific instance using Dash.

        :param index: The index of an explanation result stored in ``CFExplanation``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :return: A plotly dash figure plotting counterfactual examples.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, **kwargs))

    def ipython_plot(self, index=0, **kwargs):
        """
        Plots counterfactual examples in IPython.

        :param index: The index of an explanation result stored in ``CFExplanation``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        plotly.offline.iplot(self._plotly_figure(index, **kwargs))

    def to_json(self):
        raise RuntimeError("`CFExplanation` for timeseries cannot be converted into JSON format.")

    @classmethod
    def from_dict(cls, d):
        raise RuntimeError("`CFExplanation` for timeseries does not support `from_dict`.")
