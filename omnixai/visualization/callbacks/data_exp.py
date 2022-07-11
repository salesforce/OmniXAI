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
from ..pages.data_exp import create_right_column


@callback(
    Output("data-explanation-state", "data"),
    [
        Input("select-num-figures-data", "value"),
        Input("select-plots-data", "value")
    ],
    [
        State("data-explanation-state", "data")
    ]
)
def change_parameters(num_figures, plots, data):
    params = json.loads(data) if data is not None else {}
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-data":
            params["num_figures_per_row"] = int(num_figures)
        elif prop_id == "select-plots-data":
            params["display_plots"] = plots
    return json.dumps(params)


@callback(
    Output("right-column-data", "children"),
    Input("data-explanation-state", "data")
)
def update_view(data):
    params = json.loads(data)
    state = copy.deepcopy(board.state)
    for param, value in params.items():
        state.set_param("data", param, value)
    return create_right_column(state)
