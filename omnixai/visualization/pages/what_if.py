#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html
from ..plot import plot_one_instance
from .utils import create_explanation_layout


def create_control_panel(state) -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select instance"),
            html.Div(
                id="select-instance-parent-whatif",
                children=[
                    dcc.Dropdown(
                        id="select-instance-whatif",
                        options=[{"label": str(s), "value": str(s)} for s in state.instance_indices],
                        value=str(state.get_display_instance("local")),
                        style={"width": "350px"},
                    )
                ],
            ),

            html.Hr(),
            html.P("Change the first instance"),
            html.Label("Feature name"),
            html.Div(
                id="first-instance-feature-name-parent",
                children=[
                    dcc.Dropdown(
                        id="first-instance-feature-name",
                        options=[],
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Label("Feature value"),
            html.Div(
                id="first-instance-feature-value-parent",
                children=[
                    dcc.Dropdown(
                        id="first-instance-feature-value",
                        options=[],
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Div(
                children=[
                    html.Button(id="first-instance-set-btn", children="Set", n_clicks=0),
                    html.Button(id="first-instance-reset-btn", children="Reset", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),

            html.Hr(),
            html.P("Change the second instance"),
            html.Label("Feature name"),
            html.Div(
                id="second-instance-feature-name-parent",
                children=[
                    dcc.Dropdown(
                        id="second-instance-feature-name",
                        options=[],
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Label("Feature value"),
            html.Div(
                id="second-instance-feature-value-parent",
                children=[
                    dcc.Dropdown(
                        id="second-instance-feature-value",
                        options=[],
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Div(
                children=[
                    html.Button(id="second-instance-set-btn", children="Set", n_clicks=0),
                    html.Button(id="second-instance-reset-btn", children="Reset", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),

            html.Br(),
            html.Hr(),
            html.Div(
                children=[
                    html.Button(id="whatif-run-btn", children="Run", n_clicks=0),
                    html.Button(id="whatif-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            )
        ]
    )


def create_instance_layout(state, name) -> html.Div:
    assert name in ["a", "b"]
    if state.instances is not None:
        figure = plot_one_instance(
            state.instances,
            state.get_display_instance(f"what-if-{name}"),
            name=f"instance_{name}"
        )
        return html.Div(
            className="nine columns",
            children=html.Div(
                id="info_card",
                children=[
                    html.B(f"Instance {name.upper()}"),
                    html.Hr(),
                    html.Center(id=f"instance_table_{name}", children=figure)
                ]
            )
        )
    else:
        return html.Div()


def create_result_column(state, name) -> html.Div:
    assert name in ["a", "b"]
    explanation_views = [create_instance_layout(state, name)]
    explanation_views += create_explanation_layout(
        state, explanation_type=f"what-if-{name}")
    return html.Div(
        id=f"result-column-{name}",
        children=explanation_views
    )


def create_what_if_layout(state) -> html.Div:
    return html.Div(
        id="what_if_explanation_views",
        children=[
            # Left column
            html.Div(
                id="left-column-what-if",
                className="three columns",
                children=create_control_panel(state)
            ),
            # Right column
            html.Div(
                className="six columns",
                children=create_result_column(state, name="a")
            ),
            html.Div(
                className="six columns",
                children=create_result_column(state, name="b")
            )
        ]
    )
