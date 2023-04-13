#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html
from .utils import create_explanation_layout
from ..plot import plot_one_instance


def create_control_panel(state) -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select instance"),
            html.Div(
                id="select-instance-parent-local",
                children=[
                    dcc.Dropdown(
                        id="select-instance-local",
                        options=[{"label": str(s), "value": str(s)} for s in state.instance_indices],
                        value=str(state.get_display_instance("local")),
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Plots"),
            html.Div(
                id="select-plots-parent-local",
                children=[
                    dcc.Dropdown(
                        id="select-plots-local",
                        options=[{"label": s, "value": s} for s in state.get_plots("local")],
                        value=state.get_display_plots("local"),
                        multi=True,
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Number of figures per row"),
            dcc.Dropdown(
                id="select-num-figures-local",
                options=[{"label": "1", "value": "1"}, {"label": "2", "value": "2"}],
                value=str(state.get_num_figures_per_row("local")),
                style={"width": "350px"},
            ),
        ],
    )


def create_instance_layout(state) -> html.Div:
    if state.instances is not None:
        figure = plot_one_instance(
            state.instances, state.get_display_instance("local"))
        return html.Div(
            id="info_card",
            children=[
                html.B("Query Instance"),
                html.Hr(),
                html.Center(id="instance_table", children=figure)
            ],
        )
    else:
        return html.Div()


def create_right_column(state) -> html.Div:
    explanation_views = [create_instance_layout(state)]
    explanation_views += create_explanation_layout(state, explanation_type="local")
    return html.Div(
        id="right-column-local",
        children=explanation_views
    )


def create_local_explanation_layout(state) -> html.Div:
    return html.Div(
        id="local_explanation_views",
        children=[
            # Left column
            html.Div(
                id="left-column-local",
                className="three columns",
                children=[
                    create_control_panel(state)
                ],
            ),
            # Right column
            html.Div(
                className="nine columns",
                children=create_right_column(state)
            )
        ]
    )
