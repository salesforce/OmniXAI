#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The OmniXAI dashboard.
"""
import omnixai.visualization.state as board
board.init()

import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from .layout import create_banner, create_layout
from .pages.local_exp import create_local_explanation_layout
from .pages.global_exp import create_global_explanation_layout

import omnixai.visualization.callbacks.local_exp
import omnixai.visualization.callbacks.global_exp


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="OmniXAI",
)
app.config["suppress_callback_exceptions"] = True
app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])


class Dashboard:
    """
    The OmniXAI dashboard.

    .. code-block:: python

        dashboard = Dashboard(
            instances=instances,
            local_explanations=local_explanations,  # Specify local explanation results
            global_explanations=None,               # If there are no global explanation results
            class_names=class_names,                # A list of class names
            params={"pdp": {"features": ["Age", "Education-Num", "Capital Gain",
                                         "Capital Loss", "Hours per week", "Education",
                                         "Marital Status", "Occupation"]}}
        )
        dashboard.show()
    """

    def __init__(
            self, instances=None, local_explanations=None, global_explanations=None, class_names=None, params=None
    ):
        """
        :param instances: The instances to explain.
        :param local_explanations: The local explanation results.
        :param global_explanations: The global explanation results.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param params: A dict containing the additional parameters for plotting figures,
            e.g., `params={"pdp": {"features": ["Age", "Education-Num", "Capital Gain"]}}`.
        """
        board.state.set(
            instances=instances,
            local_explanations=local_explanations,
            global_explanations=global_explanations,
            class_names=class_names,
            params=params
        )

    def show(self, host=os.getenv("HOST", "127.0.0.1"), port=os.getenv("PORT", "8050")):
        """
        Shows the dashboard.
        """
        if board.state.has_explanations():
            app.run_server(host=host, port=port, debug=False)


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def _display_page(pathname):
    return html.Div(
        id="app-container",
        children=[
            create_banner(app),
            html.Br(),
            create_layout(board.state)
        ],
    )


@app.callback(
    Output("plots", "children"),
    Input("tabs", "value")
)
def _click_tab(tab):
    if tab == 'local-explanation':
        return create_local_explanation_layout(board.state)
    elif tab == 'global-explanation':
        return create_global_explanation_layout(board.state)
