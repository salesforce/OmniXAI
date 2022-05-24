#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html
from typing import List
from collections import OrderedDict
from .plot import plot_one_instance
from omnixai.explanations.base import DashFigure


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[html.Img(src=app.get_asset_url("logo_small.png")),
                  html.Plaintext("  Powered by Salesforce AI Research")],
    )


def create_description_card() -> html.Div:
    return html.Div(
        id="description-card",
        children=[
            # html.H5("A Library for Explainable AI"),
            html.H3("Explanation Dashboard"),
            html.Div(id="intro", children="  "),
        ],
    )


def create_control_panel(app) -> html.Div:
    instance = str(app.show_instance)
    num_figures_per_row = str(app.num_figures_per_row)
    instance_indices = app.instance_indices
    return html.Div(
        id="control-card",
        children=[
            html.P("Select instance"),
            html.Div(
                id="select_instance_parent",
                children=[
                    dcc.Dropdown(
                        id="select_instance",
                        options=[{"label": str(s), "value": str(s)} for s in instance_indices],
                        value=instance,
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Plots"),
            html.Div(
                id="select_plots_parent",
                children=[
                    dcc.Dropdown(
                        id="select_plots",
                        options=[{"label": s, "value": s} for s in app.plots],
                        value=app.show_plots,
                        multi=True,
                    )
                ],
            ),
            html.Br(),
            html.P("Number of figures per row"),
            dcc.Dropdown(
                id="select_num_figures",
                options=[{"label": "1", "value": "1"}, {"label": "2", "value": "2"}],
                value=num_figures_per_row,
                style={"width": "350px"},
            ),
        ],
    )


def create_instance_layout(app) -> html.Div:
    if app.instances is not None:
        figure = plot_one_instance(app.instances, app.show_instance)
        return html.Div(
            id="info_card",
            children=[html.B("Query Instance"), html.Hr(), html.Center(id="instance_table", children=figure)],
        )
    else:
        return html.Div()


def create_explanation_layout(app, explanation_type: str) -> List:
    instance_index = app.show_instance
    num_figures_per_row = app.num_figures_per_row
    if explanation_type == "global":
        names = [name.split(":")[0] for name in app.show_plots if name.find(":global") != -1]
        explanations = OrderedDict({name: app.global_explanations[name] for name in names})
    elif explanation_type == "local":
        names = [name.split(":")[0] for name in app.show_plots if name.find(":global") == -1 and name != "predict"]
        explanations = OrderedDict({name: app.local_explanations[name] for name in names})
    elif explanation_type == "predict":
        names = [name for name in app.show_plots if name == "predict"]
        explanations = OrderedDict({name: app.local_explanations[name] for name in names})
    else:
        raise ValueError(f"Unknown `explanation_type`: {explanation_type}")

    children = []
    div_class_name = "nine columns" if num_figures_per_row == 1 else "six columns"
    for i, name in enumerate(explanations.keys()):
        figure = None
        params = {"index": instance_index, "class_names": app.class_names}
        params.update(app.params.get(name, {}))
        try:
            if explanations[name] is not None:
                figure = explanations[name].plotly_plot(**params)
                assert isinstance(
                    figure, DashFigure
                ), f"`plotly_plot` of {type(explanations[name])} should return a `DashFigure` object."
                figure = figure.to_html_div(id=f"{explanation_type}_{name}")
        except Exception as e:
            raise type(e)(f"Explanation {name} -- {str(e)}")
        title = (
            f"{explanation_type.capitalize()} Explanation: {name.upper()}"
            if explanation_type != "predict"
            else f"{explanation_type.capitalize()}"
        )
        children.append(
            html.Div(
                id="info_card",
                children=[html.B(title), html.Hr(), figure],
                className=div_class_name,
                style={"margin-left": "15px"},
            )
        )
    return children


def create_figure_layout(app) -> html.Div:
    explanation_views = [create_instance_layout(app)]
    explanation_views += create_explanation_layout(app, explanation_type="predict")
    explanation_views += create_explanation_layout(app, explanation_type="local")
    explanation_views += create_explanation_layout(app, explanation_type="global")
    return html.Div(id="explanation_views", children=explanation_views)


def create_layout(app) -> html.Div:
    layout = html.Div(
        id="app-container",
        children=[
            # Banner
            create_banner(app),
            # Left column
            html.Div(
                id="left-column",
                className="three columns",
                children=[
                    create_description_card(),
                    create_control_panel(app),
                    html.Div(["initial child"], id="output-clientside", style={"display": "none"}),
                ],
            ),
            # Right column
            html.Div(id="right-column", className="nine columns", children=[
                create_figure_layout(app)
            ]),
        ],
    )
    return layout
