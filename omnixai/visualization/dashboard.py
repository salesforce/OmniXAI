#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The OmniXAI dashboard.
"""
import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from .layout import create_layout


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
        # Explanations
        app.instances = instances
        app.local_explanations = local_explanations if local_explanations is not None else {}
        app.global_explanations = global_explanations if global_explanations is not None else {}
        app.class_names = class_names
        app.params = {} if params is None else params
        app.plots = [name for name in app.local_explanations.keys()] + [
            f"{name}:global" for name in app.global_explanations.keys()
        ]
        # App states
        app.instance_indices = list(range(len(app.instances))) if instances is not None else []
        app.show_instance = 0
        app.num_figures_per_row = 2
        app.show_plots = app.plots

    def show(self, host=os.getenv("HOST", "127.0.0.1"), port=os.getenv("PORT", "8050")):
        """
        Shows the dashboard.
        """
        if len(app.local_explanations) > 0 or len(app.global_explanations) > 0:
            app.run_server(host=host, port=port, debug=False)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def _display_page(pathname):
    return create_layout(app)


@app.callback(
    Output("url", "pathname"),
    [Input("select_num_figures", "value"), Input("select_instance", "value"), Input("select_plots", "value")],
)
def _change_num_figures_per_row(num_figures, instance, plots):
    app.num_figures_per_row = int(num_figures)
    app.show_instance = int(instance)
    app.show_plots = plots
    return f"/dashboard"
