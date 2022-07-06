from dash import dcc
from dash import html
from collections import OrderedDict
from omnixai.explanations.base import DashFigure


def create_control_panel_global() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Plots"),
            html.Div(
                id="select-plots-parent-global",
                children=[
                    dcc.Dropdown(
                        id="select-plots-global",
                        options=[],
                        multi=True,
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Number of figures per row"),
            dcc.Dropdown(
                id="select-num-figures-global",
                options=[{"label": "1", "value": "1"}, {"label": "2", "value": "2"}],
                value=2,
                style={"width": "350px"},
            ),
        ],
    )


control_layout_global = html.Div(
    id="left-column-global",
    className="three columns",
    children=[
        create_control_panel_global()
    ],
)


def create_explanation_layout(state, explanation_type: str):
    instance_index = state.show_instance
    num_figures_per_row = state.get_num_figures_per_row("global")
    if explanation_type == "global":
        names = [name.split(":")[0] for name in state.show_plots if name.find(":global") != -1]
        explanations = OrderedDict({name: state.global_explanations[name] for name in names})
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


def create_right_column(state) -> html.Div:
    explanation_views = create_explanation_layout(state, explanation_type="global")
    return html.Div(
        id="right-column-global",
        children=explanation_views
    )


def create_global_explanation_layout(state) -> html.Div:
    return html.Div(
        id="global_explanation_views",
        children=[
            # Left column
            control_layout_global,
            # Right column
            html.Div(
                className="nine columns",
                children=create_right_column(state)
            )
        ]
    )
