import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px

from app import app
import transforms

df = transforms.df

layout = html.Div(
            id='table-paging-with-graph-container-6',
            className="five columns"
        )

@app.callback(Output('table-paging-with-graph-container-6', "children"),
        [Input('direccional', 'value')
        , Input('profundidad-slider', 'value')
        ])

def update_graph(direccional, profundidad):
    dff = df

    low = profundidad[0]
    high = profundidad[1]

    dff = dff.loc[(dff['profundidad_total'] >= low) & (dff['profundidad_total'] <= high)]

    if direccional == ['Y']:
       dff = dff.loc[dff['trayectoria'] == 'DIRECCIONAL']
    else:
        dff

        fig = px.violin(dff, y="Qi_hist", x="tipo", color="tipo", box=True, points="all",
         hover_data=dff.columns
    )
    return html.Div([
        dcc.Graph(
             figure=fig
        )
    ])
