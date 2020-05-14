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
from plotly import subplots

from app import app
import transforms

produccion = transforms.produccion

prod_aceite = go.Scatter(
  x=produccion.year,
  y=produccion.aceite_Mbd,
  name='Aceite Mbd',
  text='Produccion aceite'
)

prod_gas = go.Scatter(
  x=produccion.year,
  y=produccion.gas_asociado_MMpcd,
  name='Gas MMpcd',
  text='Produccion equivalente'
)

prod_condensado = go.Scatter(
  x=produccion.year,
  y=produccion.condensado_Mbd,
  name='Condensado Mbd',
  text='Condensado'
)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(prod_aceite, row=1, col=1, secondary_y=False)
fig.add_trace(prod_gas, row=1, col=1, secondary_y=True)
fig.add_trace(prod_condensado, row=1, col=1, secondary_y=False)


fig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig.update_xaxes(title_text="xaxis title")


fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig['layout'].update(
  height=600,
  width=1500,
  showlegend=True,
  xaxis=dict(
    # tickmode='linear',
    # ticks='outside',
    # tick0=1,
    dtick=5,
    ticklen=8,
    tickwidth=2,
    tickcolor='#000',
    showgrid=True,
    zeroline=True,
    # showline=True,
    # mirror='ticks',
    # gridcolor='#bdbdbd',
    gridwidth=2
),
  )

layout = html.Div([
        dcc.Graph(
            id='intervenciones-produccion',
            figure=fig
        ),
    ])
