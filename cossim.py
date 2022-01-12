import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

from pycor.config import Config
from pycor.visualisation.cosine_repr import get_representations_from_lemma, get_cosine_matrix

colorscales = px.colors.named_colorscales()
infos = ['def', 'citat']
# fig.show()

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
    dcc.Checklist(
        id='info_type',
        options=[{"value": y, "label": y}
                 for y in infos],
        value=["def", "genprox"]),
    dcc.Graph(id="graph"),
])


# if __name__ == "__main__":
#    app.run_server(debug=True)


@app.callback(
    Output("graph", "figure"),
    Input("colorscale", "value"),
    Input("lemma", "value"),
    Input("ordklasse", "value"),
    Input("info_type", "value")
)
def change_colorscale(scale, lemma, ordklasse, info_type):
    data, labels = get_representations_from_lemma(lemma, ordklasse, info_types=info_type)
    matrix = get_cosine_matrix(data, labels)

    fig = px.imshow(matrix, color_continuous_scale=scale, zmin=0, zmax=1)

    return fig


app.run_server(debug=Config.DEBUG)
