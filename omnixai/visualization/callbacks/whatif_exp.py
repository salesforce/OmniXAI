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
    Output("first-instance-feature-value", "options"),
    Input("first-instance-feature-value-parent", "n_clicks"),
    State("first-instance-feature-name", "value"),
)
def update_first_feature_value_dropdown(n_clicks, feature_name):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "first-instance-feature-value-parent":
            features = board.whatif_state.get_feature_values()
            options = [{"label": v, "value": v} for v in features.get(feature_name, [])]
    return options


@callback(
    Output("second-instance-feature-value", "options"),
    Input("second-instance-feature-value-parent", "n_clicks"),
    State("second-instance-feature-name", "value"),
)
def update_second_feature_value_dropdown(n_clicks, feature_name):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "second-instance-feature-value-parent":
            features = board.whatif_state.get_feature_values()
            options = [{"label": v, "value": v} for v in features.get(feature_name, [])]
    return options


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
