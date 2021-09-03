import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.express as px

from pycor.config import Config
from pycor.visualisation.get_representation import get_2d_representation_from_lemma

colorscales = px.colors.named_colorscales()


# fig.show()

def plotly_cossim(lemma, wcl, scale):
    pass

app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("Color Scale"),
    dcc.Dropdown(
        id="colorscale",
        options=[{"value": x, "label": x}
                 for x in colorscales],
        value="turbo"),
    dcc.Input(
        id="lemma",
        type="text",
        value="stjerne"),
    dcc.Input(
        id="ordklasse",
        type="text",
        value="sb."),
    dcc.Graph(id="graph"),
])


# if __name__ == "__main__":
#    app.run_server(debug=True)


@app.callback(
    Output("graph", "figure"),
    Input("colorscale", "value"),
    Input("lemma", "value"),
    Input("ordklasse", "value"),
)
def change_colorscale(scale, lemma, ordklasse):
    fig = plotly_cossim(lemma=lemma, wcl=ordklasse, scale=scale)
    return fig


app.run_server(debug=Config.DEBUG)
