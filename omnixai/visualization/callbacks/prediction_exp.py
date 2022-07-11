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
from ..pages.prediction_exp import create_right_column


@callback(
    Output("prediction-explanation-state", "data"),
    [
        Input("select-num-figures-prediction", "value"),
        Input("select-plots-prediction", "value")
    ],
    [
        State("prediction-explanation-state", "data")
    ]
)
def change_parameters(num_figures, plots, data):
    params = json.loads(data) if data is not None else {}
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-prediction":
            params["num_figures_per_row"] = int(num_figures)
        elif prop_id == "select-plots-prediction":
            params["display_plots"] = plots
    return json.dumps(params)


@callback(
    Output("right-column-prediction", "children"),
    Input("prediction-explanation-state", "data")
)
def update_view(data):
    params = json.loads(data)
    state = copy.deepcopy(board.state)
    for param, value in params.items():
        state.set_param("prediction", param, value)
    return create_right_column(state)
