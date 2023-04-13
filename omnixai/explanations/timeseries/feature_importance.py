#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature importance explanations for time series tasks.
"""
import warnings
import numpy as np
from ..base import ExplanationBase, DashFigure


class FeatureImportance(ExplanationBase):
    """
    The class for feature importance explanations for time series tasks. It uses a list to store
    the feature importance explanations of the input instances. Each item in the list
    is a dict with the following format `{"instance": the input instance, "scores": feature importance scores}`,
    where both "instance" and "scores" are pandas dataframes.
    """

    def __init__(self, mode, explanations=None):
        """
        :param mode: The task type, e.g., `anomaly_detection` or `forecasting`.
        :param explanations: The explanation results for initializing ``FeatureImportance``,
            which is optional.
        """
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def add(self, instance, importance_scores, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param instance: The instance to explain.
        :param importance_scores: The feature importance scores.
        """
        e = {"instance": instance, "scores": importance_scores}
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``FeatureImportance``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"instance": the input instance,
            "scores": feature importance scores}`, where both "instance" and "scores" are
            pandas dataframes.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, index=None, figure_type=None, max_num_variables_to_plot=25, **kwargs):
        """
        Plots importance scores for time series data.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it plots all the explanations.
        :param figure_type: The figure type, e.g., plotting importance scores in a `timeseries` or a `bar`.
        :param max_num_variables_to_plot: The maximum number of variables to plot in the figure.
        :return: A matplotlib figure plotting feature importance scores.
        """
        import matplotlib.pyplot as plt

        if len(self.explanations) == 0:
            return None
        if figure_type is not None:
            assert figure_type in ["timeseries", "bar"], \
                "`figure_type` can only be `timeseries` or `bar`."
        else:
            ts = self.explanations[0]["instance"]
            if ts.shape[1] == 1:
                # Univariate time series
                figure_type = "timeseries"
            else:
                # Multivariate time series
                figure_type = "bar" if ts.shape[0] == 1 else "timeseries"

        figures = []
        explanations = [self.explanations[index]] if index is not None \
            else self.explanations

        if figure_type == "timeseries":
            for exp in explanations:
                ts, scores = exp["instance"], exp["scores"]
                num_variables = max(ts.shape[1], scores.shape[1])
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
                    if i < ts.shape[1]:
                        left_ax = axes[row, col]
                        ts_a = ts[[ts.columns[i]]]
                        timestamps = [str(v) for v in ts_a.index.values]
                        left_ax.plot(timestamps, ts_a.values.flatten(), color='k')
                        left_ax.set_xticklabels(left_ax.get_xticks(), rotation=45)
                    # Plot the importance scores
                    right_ax = axes[row, col].twinx()
                    ts_b = scores[[scores.columns[i]]]
                    right_ax.plot(timestamps, ts_b.values.flatten(), color='r', label="score")
                    plt.title(f"{scores.columns[i]}")
                    plt.grid()
                figures.append(fig)
        else:
            for exp in explanations:
                scores = exp["scores"]
                min_values = np.min(scores.values, axis=0)
                max_values = np.max(scores.values, axis=0)
                flag = (np.abs(min_values) > np.abs(max_values)).astype(int)
                values = min_values * flag + max_values * (1 - flag)
                fnames = [f"{c}    " for c in scores.columns]

                fig, ax = plt.subplots(1, 1)
                plt.sca(ax)
                positions = np.arange(len(values)) + 0.5
                colors = ["green" if x > 0 else "red" for x in values]
                plt.barh(positions, values, align="center", color=colors)
                ax.yaxis.set_ticks_position("right")
                plt.yticks(positions, fnames, ha="right")
                figures.append(fig)
        return figures

    def _plotly_figure(self, index, **kwargs):
        import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        exp = self.explanations[index]
        traces, score_traces = [], []
        color_list = plotly.colors.qualitative.Dark24
        ts, scores = exp["instance"], exp["scores"]
        # Original time series data
        for i in range(ts.shape[1]):
            v = ts[[ts.columns[i]]]
            color = color_list[i % len(color_list)]
            traces.append(go.Scatter(
                name=ts.columns[i],
                x=v.index,
                y=v.values.flatten(),
                mode="lines",
                line=dict(color=color)
            ))
        # Feature importance scores
        for i in range(ts.shape[1]):
            v = scores[[ts.columns[i]]]
            color = color_list[i % len(color_list)]
            score_traces.append(go.Scatter(
                name=f"{scores.columns[i]}_score",
                x=v.index,
                y=v.values.flatten(),
                mode="lines",
                line=dict(color=color, dash="dash"),
            ))

        if "@timestamp" in scores:
            v = scores[["@timestamp"]]
            score_traces.append(go.Scatter(
                name="timestamp_score",
                x=v.index,
                y=v.values.flatten(),
                mode="lines",
                line=dict(color="black", dash="dash"),
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
        fig = make_subplots(
            figure=go.Figure(layout=layout),
            specs=[[{"secondary_y": True}]]
        )
        fig.update_yaxes(title_text="Timeseries")
        fig.update_yaxes(title_text="Importance Score", secondary_y=True)
        for trace_a, trace_b in zip(traces, score_traces):
            fig.add_trace(trace_a)
            fig.add_trace(trace_b, secondary_y=True)
        if len(score_traces) > len(traces):
            for trace_b in score_traces[len(traces):]:
                fig.add_trace(trace_b, secondary_y=True)
        return fig

    def plotly_plot(self, index=0, **kwargs):
        """
        Plots feature importance scores for one specific instance using Dash.

        :param index: The index of an explanation result stored in ``FeatureImportance``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :return: A plotly dash figure plotting feature importance scores.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, **kwargs))

    def ipython_plot(self, index=0, **kwargs):
        """
        Plots the feature importance scores in IPython.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        plotly.offline.iplot(self._plotly_figure(index, **kwargs))

    def to_json(self):
        raise RuntimeError("`FeatureImportance` for timeseries cannot be converted into JSON format.")

    @classmethod
    def from_dict(cls, d):
        raise RuntimeError("`FeatureImportance` for timeseries does not support `from_dict`.")
