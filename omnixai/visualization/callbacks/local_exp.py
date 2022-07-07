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
            board.state.set_display_instance("local", int(instance))
        elif prop_id == "select-num-figures-local":
            board.state.set_num_figures_per_row("local", int(num_figures))
        elif prop_id == "select-plots-local":
            board.state.set_display_plots("local", plots)
    return create_right_column(board.state)
