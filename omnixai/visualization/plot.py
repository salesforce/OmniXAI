#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import base64
from io import BytesIO
from dash import dash_table, html, dcc

from ..data.tabular import Tabular
from ..data.image import Image
from ..data.text import Text
from ..preprocessing.image import Resize


def plot_table(instance):
    table = dash_table.DataTable(
        id="table",
        columns=[{"name": c, "id": c} for c in instance.columns],
        data=[{c: v for c, v in zip(instance.columns, instance.values[0])}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_table={"overflowX": "scroll"},
    )
    return table


def plot_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return html.Img(id="my-img", className="image", src="data:image/png;base64, " + img_str)


def plot_text(text, limit=800):
    if len(text) > limit:
        text = text[:limit] + "..."
    return dcc.Markdown(id="text-instance", children=f"_{text}_")


def plot_one_instance(instances, index):
    if instances is None:
        return None
    elif isinstance(instances, Tabular):
        return plot_table(instances.iloc(index).to_pd())
    elif isinstance(instances, Image):
        img = Resize(300).transform(instances[index])
        return plot_image(img.to_pil())
    elif isinstance(instances, Text):
        return plot_text(instances[index].to_str())
    else:
        raise NotImplementedError
