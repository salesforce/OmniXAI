#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[html.Img(src=app.get_asset_url("logo_small.png")),
                  html.Plaintext("  Powered by Salesforce AI Research")],
    )


def create_layout(state) -> html.Div:
    children, values = [], []
    if len(state.get_explanations("data")) > 0:
        children.append(
            dcc.Tab(label="Data Analysis", value="data-explanation")
        )
        values.append("data-explanation")
    if len(state.get_explanations("local")) > 0:
        children.append(
            dcc.Tab(label="Local Explanation", value="local-explanation")
        )
        values.append("local-explanation")
    if len(state.get_explanations("global")) > 0:
        children.append(
            dcc.Tab(label="Global Explanation", value="global-explanation")
        )
        values.append("global-explanation")

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
