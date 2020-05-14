import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app
import transforms

df = transforms.df_info


fig = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"type": "table"},
           {"type": "table"}],
           [{"type": "table"},
           {"type": "table"}]],

)


fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Descripcion del campo</b>','<b>Datos</b>'],
            font=dict(size=18,
                      family='Helvetica Neue'),
            height=40,
            align="left"
        ),
        cells=dict(
            values=[[k for k in df.index[:5]],
                    [j for j in df.Datos[:5]]],
            font=dict(size=18,
                     family='Helvetica Neue'),
            height=40,
            align = "left")
    ),
    row=1, col=1
)

fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Situacion contractual</b>','<b>Datos</b>'],
            font=dict(size=18),
            height=40,
            align="left"
        ),
        cells=dict(
            values=[[k for k in df.index[5:]],
                    [j for j in df.Datos[5:]]],
            font=dict(size=18,
                     family='Helvetica Neue'),
            height=40,
            align = "left")
    ),
    row=2, col=1
)

fig.add_trace(
    go.Table(
        header=dict(
            values=['<b>Hidrocarburos</b>','<b>Datos</b>'],
            font=dict(size=18,
                      family='Helvetica Neue'),
            height=40,
            align="left"
        ),
        cells=dict(
            values=[[k for k in df.index[5:]],
                    [j for j in df.Datos[5:]]],
            font=dict(size=18,
                      family='Helvetica Neue'),
            height=40,
            align = "left")
    ),
    row=1, col=2

)

fig.update_layout(
    height=800,
    showlegend=False,
    title_text="<b>Informacion general del campo</b> ",
    font=dict(size=20,
              family='Helvetica Neue')
)


layout = html.Div([
        dcc.Graph(
            id='caracteristicas-generales'
            , figure=fig
        )


    ])
