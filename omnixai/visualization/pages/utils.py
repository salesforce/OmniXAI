from dash import html


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[html.Img(src=app.get_asset_url("logo_small.png")),
                  html.Plaintext("  Powered by Salesforce AI Research")],
    )


def create_description_card() -> html.Div:
    return html.Div(
        id="description-card",
        children=[
            # html.H5("A Library for Explainable AI"),
            html.H3("Explanation Dashboard"),
            html.Div(id="intro", children="  "),
        ],
    )
