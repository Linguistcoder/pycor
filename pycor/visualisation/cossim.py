import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import umap
from dash.dependencies import Input, Output

from pycor.visualisation.config import Config
from pycor.visualisation.get_representation import annotations_to_embeddings


def reduce_dim(X, n_dim=100):
    """Reduces dimensionality of X to n_dim using UMAP"""
    fit = umap.UMAP(n_components=n_dim, metric='cosine', n_neighbors=50, min_dist=0.3)
    fit.fit(X)
    return fit


# create a EmbeddingsForVisualisation object from the dictionary data and embeddings file
annotations = annotations_to_embeddings('data/hum_anno/all_09_06_2022.txt',
                                        'data/hum_anno/annotations_with_embeddings.tsv',
                                        reduce_dim)

colorscales = px.colors.named_colorscales()

# we use Dash to create a webapp to show the figure
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
        id="hom_nr",
        type="text",
        value="1"),
    dcc.Input(
        id="model_name",
        type="text",
        value="bert"),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("colorscale", "value"),
    Input("lemma", "value"),
    Input("hom_nr", "value"),
    Input("model_name", "value")
)
def change_colorscale(scale, lemma, hom_nr, model_name):
    """updates the webapp input into the figure"""
    matrix = annotations.get_cosine_matrix((lemma, int(hom_nr)), model_name, reduce=False)
    fig = px.imshow(matrix, color_continuous_scale=scale, zmin=0, zmax=1)

    return fig


app.run_server(debug=Config.DEBUG)
