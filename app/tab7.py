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
            id='table-paging-with-graph-container-7',
            className="five columns"
        )

@app.callback(Output('table-paging-with-graph-container-7', "children"),
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

        fig = make_subplots(
                rows=2, cols=2,
                shared_xaxes=True,
                vertical_spacing=0.03,
                specs=[[{"type": "xy"}],
                       [{"type": "xy"}],
                       [{"type": "xy"}],
                       [{"type": "xy"}],
        )

        fig.add_trace(
                    px.pie(df,
                    values='pop',
                    names='country',
                    title='Population of European continent'
                    ),
            row=3, col=1
        )

        fig.add_trace(
                    px.pie(df,
                    values='pop',
                    names='country',
                    title='Population of European continent'
                    ),
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=intervenciones.index,
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[intervenciones[k].tolist() for k in df.columns[1:]],
                    align = "left")
            ),
            row=1, col=1
        )
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Bitcoin mining stats for 180 days",
        )
         hover_data=dff.columns
    )
    return html.Div([
        dcc.Graph(
             figure=fig
        )
    ])
