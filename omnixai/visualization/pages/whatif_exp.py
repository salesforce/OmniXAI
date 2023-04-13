#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html
from ..plot import plot_one_instance
from omnixai.explanations.base import DashFigure


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
                        options=[{"label": str(s), "value": str(s)}
                                 for s in state.instance_indices],
                        value=str(state.get_display_instance()),
                        style={"width": "350px"},
                    )
                ],
            ),

            html.Br(),
            html.P("Change the first instance"),
            html.Label("Feature name"),
            html.Div(
                id="first-instance-feature-name-parent",
                children=[
                    dcc.Dropdown(
                        id="first-instance-feature-name",
                        options=[{"label": str(s), "value": str(s)}
                                 for s in state.get_feature_values().keys()],
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
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="first-instance-set-btn", children="Set", n_clicks=0),
                    html.Button(id="first-instance-reset-btn", children="Reset", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center", "width": "350px"},
            ),

            html.Br(),
            html.P("Change the second instance"),
            html.Label("Feature name"),
            html.Div(
                id="second-instance-feature-name-parent",
                children=[
                    dcc.Dropdown(
                        id="second-instance-feature-name",
                        options=[{"label": str(s), "value": str(s)}
                                 for s in state.get_feature_values().keys()],
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
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="second-instance-set-btn", children="Set", n_clicks=0),
                    html.Button(id="second-instance-reset-btn", children="Reset", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center", "width": "350px"},
            ),

            html.Br(),
            html.P("Generate explanations"),
            html.Div(
                children=[
                    html.Button(id="whatif-run-btn", children="Explain", n_clicks=0),
                ],
                style={"textAlign": "center", "width": "350px"},
            )
        ]
    )


def create_instance_layout(state, name) -> html.Div:
    assert name in ["a", "b"]
    if state.instances is not None:
        figure = plot_one_instance(
            state.get_instance(f"instances-{name}", state.get_display_instance()),
            index=0,
            name=f"instance_{name}"
        )
        return html.Div(
            className="six columns",
            children=html.Div(
                id="info_card",
                children=[
                    html.B(f"Instance {name.upper()}"),
                    html.Hr(),
                    html.Center(id=f"instance_table_{name}", children=figure)
                ]
            ),
            style={"margin-left": "15px"}
        )
    else:
        return html.Div()


def create_explanation_layout(state):
    instance_index = state.get_display_instance()
    explanations_a = state.get_explanations("what-if-a", instance_index)
    explanations_b = state.get_explanations("what-if-b", instance_index)

    def _add_figure(_children, _explanations, _name, _explanation_type):
        figure = None
        params = {"index": 0, "class_names": state.class_names}
        params.update(state.params.get(_name, {}))
        try:
            if _explanations[_name] is not None:
                figure = _explanations[_name].plotly_plot(**params)
                assert isinstance(
                    figure, DashFigure
                ), f"`plotly_plot` of {type(_explanations[_name])} should return a `DashFigure` object."
                figure = figure.to_html_div(id=f"{_explanation_type}_{_name}")
        except Exception as e:
            raise type(e)(f"Explanation {_name} -- {str(e)}")
        title = f"Local Explanation: {_name.upper()}"
        children.append(
            html.Div(
                id="info_card",
                children=[html.B(title), html.Hr(), figure],
                className="six columns",
                style={"margin-left": "15px"},
            )
        )

    children = []
    for i, name in enumerate(explanations_a.keys()):
        _add_figure(children, explanations_a, name, "what-if-a")
        _add_figure(children, explanations_b, name, "what-if-b")
    return children


def create_result_column(state) -> html.Div:
    explanation_views = [
        create_instance_layout(state, "a"),
        create_instance_layout(state, "b")
    ] + create_explanation_layout(state)
    return html.Div(
        id=f"result-column",
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
                className="nine columns",
                children=create_result_column(state)
            ),
        ]
    )
