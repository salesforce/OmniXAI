import dash
import copy
import json
import omnixai.visualization.state as board
from dash import Input, Output, State, callback
from ..pages.whatif_exp import create_result_column


@callback(
    Output("whatif-explanation-state", "data"),
    [
        Input("select-instance-whatif", "value")
    ],
    State("whatif-explanation-state", "data")
)
def change_parameters(instance, data):
    params = json.loads(data) if data is not None else {}
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-instance-whatif":
            params["display_instance"] = int(instance)
    return json.dumps(params)


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
