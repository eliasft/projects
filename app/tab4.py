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
            id='table-paging-with-graph-container-2',
            className="five columns"
        )

@app.callback(Output('table-paging-with-graph-container-2', "children"),
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

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dff.columns),
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[dff[k].tolist() for k in dff.columns[:]],
                   fill_color='lavender',
                   align='left'),
        columnwidth = [5,2]
        )
    ])

    fig.update_layout(
    height=800,
    showlegend=False,
    title_text="Tabla resumen del campo Lacamango")

    return html.Div([
        dcc.Graph(
            id='tabla'
            , figure=fig
                    # dict(
                    #     x=df['price'],
                    #     y=df['rating'],
                    #     #text=df[df['continent'] == i]['country'],
                    #     mode='markers',
                    #     opacity=0.7,
                    #     marker={
                    #         'size': 8,
                    #         'line': {'width': 0.5, 'color': 'white'}
                    #     },
                    #     name='Price v Rating'
                    #)

        )
    ])
