#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash
import omnixai.visualization.state as board
from dash import Input, Output, callback
from ..pages.prediction_exp import create_right_column


@callback(
    Output("right-column-prediction", "children"),
    [Input("select-num-figures-prediction", "value"),
     Input("select-plots-prediction", "value")],
)
def change_parameters(num_figures, plots):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-prediction":
            board.state.set_num_figures_per_row("prediction", int(num_figures))
        elif prop_id == "select-plots-prediction":
            board.state.set_display_plots("prediction", plots)
    return create_right_column(board.state)
