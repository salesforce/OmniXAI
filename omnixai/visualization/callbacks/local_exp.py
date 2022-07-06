import dash
import omnixai.visualization.state as board
from dash import Input, Output, callback
from ..pages.local_exp import create_right_column


@callback(
    Output("right-column-local", "children"),
    [Input("select-instance-local", "value"),
     Input("select-num-figures-local", "value"),
     Input("select-plots-local", "value")],
)
def change_parameters(instance, num_figures, plots):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-instance-local":
            board.state.show_instance = int(instance)
        elif prop_id == "select-num-figures-local":
            board.state.set_num_figures_per_row("local", int(num_figures))
        elif prop_id == "select-plots-local":
            board.state.show_plots = plots
    return create_right_column(board.state)


@callback(
    Output("select-instance-local", "options"),
    Input("select-instance-parent-local", "n_clicks")
)
def select_instance_parent(n_clicks):
    options = [{"label": '0', "value": '0'}]
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-instance-parent-local":
            options = [{"label": str(s), "value": str(s)}
                       for s in board.state.instance_indices]
    return options


@callback(
    Output("select-plots-local", "options"),
    Input("select-plots-parent-local", "n_clicks")
)
def select_plots_parent(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-plots-parent-local":
            options = [{"label": s, "value": s} for s in board.state.plots]
    return options
