from dash import dcc
from dash import html
from collections import OrderedDict
from omnixai.explanations.base import DashFigure
from ..plot import plot_one_instance


def create_control_panel_local() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select instance"),
            html.Div(
                id="select-instance-parent-local",
                children=[
                    dcc.Dropdown(
                        id="select-instance-local",
                        options=[{"label": str(0), "value": str(0)}],
                        value='0',
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Plots"),
            html.Div(
                id="select-plots-parent-local",
                children=[
                    dcc.Dropdown(
                        id="select-plots-local",
                        options=[],
                        multi=True,
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Number of figures per row"),
            dcc.Dropdown(
                id="select-num-figures-local",
                options=[{"label": "1", "value": "1"}, {"label": "2", "value": "2"}],
                value=2,
                style={"width": "350px"},
            ),
        ],
    )


control_layout_local = html.Div(
    id="left-column-local",
    className="three columns",
    children=[
        create_control_panel_local()
    ],
)


def create_explanation_layout(state, explanation_type: str):
    instance_index = state.show_instance
    num_figures_per_row = state.get_num_figures_per_row("local")
    if explanation_type == "local":
        names = [name.split(":")[0] for name in state.show_plots if name.find(":global") == -1 and name != "predict"]
        explanations = OrderedDict({name: state.local_explanations[name] for name in names})
    elif explanation_type == "predict":
        names = [name for name in state.show_plots if name == "predict"]
        explanations = OrderedDict({name: state.local_explanations[name] for name in names})
    else:
        raise ValueError(f"Unknown `explanation_type`: {explanation_type}")

    children = []
    div_class_name = "nine columns" if num_figures_per_row == 1 else "six columns"
    for i, name in enumerate(explanations.keys()):
        figure = None
        params = {"index": instance_index, "class_names": state.class_names}
        params.update(state.params.get(name, {}))
        try:
            if explanations[name] is not None:
                figure = explanations[name].plotly_plot(**params)
                assert isinstance(
                    figure, DashFigure
                ), f"`plotly_plot` of {type(explanations[name])} should return a `DashFigure` object."
                figure = figure.to_html_div(id=f"{explanation_type}_{name}")
        except Exception as e:
            raise type(e)(f"Explanation {name} -- {str(e)}")
        title = (
            f"{explanation_type.capitalize()} Explanation: {name.upper()}"
            if explanation_type != "predict"
            else f"{explanation_type.capitalize()}"
        )
        children.append(
            html.Div(
                id="info_card",
                children=[html.B(title), html.Hr(), figure],
                className=div_class_name,
                style={"margin-left": "15px"},
            )
        )
    return children


def create_instance_layout(state) -> html.Div:
    if state.instances is not None:
        figure = plot_one_instance(state.instances, state.show_instance)
        return html.Div(
            id="info_card",
            children=[
                html.B("Query Instance"),
                html.Hr(),
                html.Center(id="instance_table", children=figure)
            ],
        )
    else:
        return html.Div()


def create_right_column(state) -> html.Div:
    explanation_views = [create_instance_layout(state)]
    explanation_views += create_explanation_layout(state, explanation_type="predict")
    explanation_views += create_explanation_layout(state, explanation_type="local")
    return html.Div(
        id="right-column-local",
        children=explanation_views
    )


def create_local_explanation_layout(state) -> html.Div:
    return html.Div(
        id="local_explanation_views",
        children=[
            # Left column
            control_layout_local,
            # Right column
            html.Div(
                className="nine columns",
                children=create_right_column(state)
            )
        ]
    )
