import dash
import copy
import json
import omnixai.visualization.state as board
from dash import Input, Output, State, callback
from ..pages.whatif_exp import create_result_column


@callback(
    Output("result-column", "children"),
    Input("whatif-explanation-state", "data")
)
def update_view(data):
    state = copy.deepcopy(board.whatif_state)
    if data:
        params = json.loads(data)
        for param, value in params.items():
            state.set_param(param, value)
    return create_result_column(state)
