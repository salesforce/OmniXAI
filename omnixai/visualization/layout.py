#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html


tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[html.Img(src=app.get_asset_url("logo_small.png")),
                  html.Plaintext("  Powered by Salesforce AI Research")],
    )


def create_layout(state, whatif_state) -> html.Div:
    children, values = [], []
    # Data analysis tab
    if len(state.get_explanations("data")) > 0:
        children.append(
            dcc.Tab(label="Data Analysis", value="data-explanation",
                    style=tab_style, selected_style=tab_selected_style)
        )
        values.append("data-explanation")
    # Prediction analysis tab
    if len(state.get_explanations("prediction")) > 0:
        children.append(
            dcc.Tab(label="Prediction Analysis", value="prediction-explanation",
                    style=tab_style, selected_style=tab_selected_style)
        )
        values.append("prediction-explanation")
    # Local explanation tab
    if len(state.get_explanations("local")) > 0:
        children.append(
            dcc.Tab(label="Local Explanation", value="local-explanation",
                    style=tab_style, selected_style=tab_selected_style)
        )
        values.append("local-explanation")
    # Global explanation tab
    if len(state.get_explanations("global")) > 0:
        children.append(
            dcc.Tab(label="Global Explanation", value="global-explanation",
                    style=tab_style, selected_style=tab_selected_style)
        )
        values.append("global-explanation")
    # What-if explanation tab
    if whatif_state.is_available():
        children.append(
            dcc.Tab(label="What-if Explanation", value="what-if-explanation",
                    style=tab_style, selected_style=tab_selected_style)
        )
        values.append("what-if-explanation")

    layout = html.Div(
        id="app-content",
        children=[
            dcc.Tabs(
                id="tabs",
                value=values[0] if values else "none",
                children=children
            ),
            html.Div(id="plots")
        ],
    )
    return layout
