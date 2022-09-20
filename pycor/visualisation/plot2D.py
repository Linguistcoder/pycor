import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.express as px
from configs.webapp_config import Config

from pycor.visualisation.get_representation import annotations_to_embeddings

annotations = annotations_to_embeddings('data/hum_anno/all_09_06_2022.txt',
                                        'data/hum_anno/annotations_with_embeddings.tsv'
                                        )

colorscales = px.colors.named_colorscales()


# fig.show()

def plotly_senses(lemmas: list, model_name: str, scale, data=annotations):
    senses, lemmas, labels, scores, cor_sense = data.get_2d_representations_from_lemmas(lemmas, model_name)

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
                                     mode='markers',
                                     marker=dict(size=[s * 2 + 4 if s != 0 else 6 for s in scores],
                                                 # color=[i * 0 if i == 1 else i for i in lengths],
                                                 color=[int(c) / 10 for c in lemmas],
                                                 colorscale=scale,
                                                 cmin=1),
                                     text=labels)],
                    layout=layout
                    )

    # fig.update_layout(
    #    width=600,
    #   height=600)

    # fig.update_xaxes(
    #     tickvals=[-7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    #               20])
    # fig.update_yaxes(tickvals=[-8, -7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    return fig


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


# if __name__ == "__main__":
#    app.run_server(debug=True)


@app.callback(
    Output("graph", "figure"),
    Input("colorscale", "value"),
    Input("lemma", "value"),
    Input("homnr", "value"),
    Input("model_name", "value"),
)
def change_colorscale(scale, lemma, homnr, model_name):
    lemma = lemma.split(', ')
    homnr = [int(hom) for hom in homnr.split(', ')]
    data = zip(lemma, homnr)
    fig = plotly_senses(lemmas=data,
                        model_name=model_name, scale=scale)
    return fig


app.run_server(debug=Config.DEBUG)
