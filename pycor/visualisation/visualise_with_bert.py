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

def plotly_senses(lemma, wcl, scale, n_sim):
    senses, lemmas, labels, scores, ddo_sense = get_2d_bert_representation_from_lemma(lemma, wcl, n_sim=n_sim)

    fig = go.Figure(data=[go.Scatter(x=senses[:, 0].reshape(-1),
                                     y=senses[:, 1].reshape(-1),
                                     mode='markers',
                                     marker=dict(size=[s * 2 + 4 if s != 0 else 6 for s in scores],
                                                 # color=[i * 0 if i == 1 else i for i in lengths],
                                                 color=ddo_sense,
                                                 colorscale=scale,
                                                 cmin=1),
                                     text=labels)])
    fig.update_layout(
        width=600,
        height=600)

    fig.update_xaxes(
        tickvals=[-7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20])
    fig.update_yaxes(tickvals=[-8, -7, -6, -5, -4, -3, -2, -1, -0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

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
def change_colorscale(scale, lemma, ordklasse, n_sim):
    fig = plotly_senses(lemma=lemma, wcl=ordklasse, scale=scale, n_sim=n_sim)
    return fig


app.run_server(debug=Config.DEBUG)
