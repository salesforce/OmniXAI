import dash
import omnixai.visualization.state as board
from dash import Input, Output, callback
from ..pages.global_exp import create_right_column


@callback(
    Output("right-column-global", "children"),
    [Input("select-num-figures-global", "value"),
     Input("select-plots-global", "value")],
)
def change_parameters(num_figures, plots):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-num-figures-global":
            board.state.set_num_figures_per_row("global", int(num_figures))
        elif prop_id == "select-plots-global":
            board.state.show_plots = plots
    return create_right_column(board.state)


@callback(
    Output("select-plots-global", "options"),
    Input("select-plots-parent-global", "n_clicks")
)
def select_plots_parent(n_clicks):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-plots-parent-global":
            options = [{"label": s, "value": s} for s in board.state.plots]
    return options
