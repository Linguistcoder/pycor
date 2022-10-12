import dash
import umap
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.express as px

from pycor.visualisation.config import Config
from pycor.visualisation.get_representation import annotations_to_embeddings


def reduce_dim(X, n_dim=2):
    """Reduces dimensionality of X to n_dim using UMAP"""
    fit = umap.UMAP(n_components=n_dim, metric='cosine', n_neighbors=30, min_dist=0.1)
    fit.fit(X)
    return fit


# create a EmbeddingsForVisualisation object from the dictionary data and embeddings file
annotations = annotations_to_embeddings('data/hum_anno/all_09_06_2022.txt',
                                        'data/hum_anno/annotations_with_embeddings.tsv',
                                        reduce_dim)
# define selection of colour scales that can be chosen in the figure
colorscales = px.colors.named_colorscales()


def plotly_senses(lemmas, model_name, scale, data=annotations):
    """
    Create 2D embedding visualisation for lemmas using model_name.
    :param lemmas: (list or iterable) of tuples (lemmas, homonym number)
    :param model_name: (str) name of embedding model
    :param scale: (str) name of colour scale to use for the dots
    :param data: (EmbeddingsForVisualisation) sense inventory dataset with embeddings
    :return: figure
    """
    # get data
    senses, lemmas, labels, scores, cor_sense = data.get_2d_representations_from_lemmas(lemmas, model_name)

    # define layout of figure
    layout = go.Layout(
        autosize=False,
        width=1000,
        height=800,

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )

    fig = go.Figure(data=[go.Scatter(x=senses[:, 0].reshape(-1),
                                     y=senses[:, 1].reshape(-1),
                                     mode='markers',  # create scatter plot
                                     # the marker size depends on the "sense density score"
                                     # the marker colour depends on the COR sense
                                     marker=dict(size=[s * 2 + 4 if s != 0 else 6 for s in scores],
                                                 color=[int(c) / 10 for c in lemmas],
                                                 colorscale=scale,
                                                 cmin=1),
                                     text=labels)],
                    layout=layout
                    )

    return fig


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
        value="ansigt"),  # "bank, fabrik, skole, opera, parlament, vinhus, cirkus"
    dcc.Input(
        id="homnr",
        type="text",
        value='1'),
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
    Input("homnr", "value"),
    Input("model_name", "value"),
)
def change_colorscale(scale, lemma, homnr, model_name):
    """updates the webapp input into the figure"""
    lemma = lemma.split(', ')
    homnr = [int(hom) for hom in homnr.split(', ')]
    data = zip(lemma, homnr)
    fig = plotly_senses(lemmas=data,
                        model_name=model_name, scale=scale)
    return fig


app.run_server(debug=Config.DEBUG)
