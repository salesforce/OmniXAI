#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import base64
from io import BytesIO
from dash import dash_table, html, dcc

from ..data.tabular import Tabular
from ..data.image import Image
from ..data.text import Text
from ..data.timeseries import Timeseries
from ..preprocessing.image import Resize


def plot_table(instance, name=None):
    table = dash_table.DataTable(
        id="table" if not name else name,
        columns=[{"name": c, "id": c} for c in instance.columns],
        data=[{c: v for c, v in zip(instance.columns, instance.values[0])}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_table={"overflowX": "scroll"},
    )
    return table


def plot_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return html.Img(id="my-img", className="image", src="data:image/png;base64, " + img_str)


def plot_text(text, limit=800):
    if len(text) > limit:
        text = text[:limit] + "..."
    return dcc.Markdown(id="text-instance", children=f"_{text}_")


def plot_timeseries(ts):
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    traces = []
    color_list = plotly.colors.qualitative.Dark24
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
    return html.Div([dcc.Graph(figure=fig)])


def plot_one_instance(instances, index, name=None):
    if instances is None:
        return None
    elif isinstance(instances, Tabular):
        return plot_table(instances.iloc(index).to_pd(), name)
    elif isinstance(instances, Image):
        img = Resize(300).transform(instances[index])
        return plot_image(img.to_pil())
    elif isinstance(instances, Text):
        return plot_text(instances[index].to_str())
    elif isinstance(instances, Timeseries):
        # Only one time-series in a Timeseries instance
        return plot_timeseries(instances.to_pd())
    else:
        raise NotImplementedError
