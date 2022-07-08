#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash
import copy
import json
import omnixai.visualization.state as board
from dash import Input, Output, State, callback
from ..pages.local_exp import create_right_column


@callback(
    Output("local-explanation-state", "data"),
    [
        Input("select-num-figures-local", "value"),
        Input("select-plots-local", "value"),
        Input("select-instance-local", "value")
    ],
    [
        State("local-explanation-state", "data")
    ]
)
def change_parameters(num_figures, plots, instance, data):
    params = json.loads(data) if data is not None else {}
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-local":
            params["num_figures_per_row"] = int(num_figures)
        elif prop_id == "select-plots-local":
            params["display_plots"] = plots
        elif prop_id == "select-instance-local":
            params["display_instance"] = int(instance)
    return json.dumps(params)


@callback(
    Output("right-column-local", "children"),
    Input("local-explanation-state", "data")
)
def update_view(data):
    params = json.loads(data)
    state = copy.deepcopy(board.state)
    for param, value in params.items():
        state.set_param("local", param, value)
    return create_right_column(state)
