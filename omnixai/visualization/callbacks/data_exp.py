#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash
import omnixai.visualization.state as board
from dash import Input, Output, callback
from ..pages.data_exp import create_right_column


@callback(
    Output("right-column-data", "children"),
    [Input("select-num-figures-data", "value"),
     Input("select-plots-data", "value")],
)
def change_parameters(num_figures, plots):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-data":
            board.state.set_num_figures_per_row("data", int(num_figures))
        elif prop_id == "select-plots-data":
            board.state.set_display_plots("data", plots)
    return create_right_column(board.state)
