#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import html
from collections import OrderedDict
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


def create_explanation_layout(state, explanation_type: str):
    instance_index = state.get_display_instance("local") \
        if explanation_type == "local" else None
    num_figures_per_row = state.get_num_figures_per_row(explanation_type)
    display_plots = state.get_display_plots(explanation_type)
    explanations = OrderedDict({name: state.get_explanations(explanation_type)[name]
                                for name in display_plots})

    children = []
    div_class_name = "nine columns" if num_figures_per_row == 1 else "six columns"
    for i, name in enumerate(explanations.keys()):
        figure = None
        params = {"index": instance_index, "class_names": state.class_names}
        params.update(state.params.get(name, {}))
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
