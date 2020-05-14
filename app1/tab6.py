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

liquidos = go.Bar(
  x=produccion.year,
  y=produccion.liquidos_Mbd,
  name='Liquidos Mbd',
  text='Liquidos Mbd'
)
corte = go.Scatter(
  x=produccion.year,
  y=produccion.corte_agua,
  name='Corte de agua %',
  text='Corte de agua %'
)



fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(liquidos,row=1, col=1, secondary_y=False)
fig.add_trace(corte, row=1, col=1, secondary_y=True)



fig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig.update_xaxes(title_text="xaxis title")


fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig['layout'].update(
  height= 600,
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

aceite = go.Bar(
  x=produccion.year,
  y=produccion.aceite_Mbd,
  name='Aceite Mbd',
  text='Aceite Mbd'
)

gas = go.Bar(
  x=produccion.year,
  y=produccion.gas_asociado_MMpcd,
  name='Gas MMpcd',
  text='Gas MMpcd'
)

rga = go.Scatter(
  x=produccion.year,
  y=produccion.RGA,
  name='RGA pc/b',
  text='RGA pc/b'
)



fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(aceite,row=1, col=1, secondary_y=False)
fig1.add_trace(gas, row=1, col=1, secondary_y=False)
fig1.add_trace(rga, row=1, col=1, secondary_y=True)



fig1.update_layout(
    barmode='stack',
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig1.update_xaxes(title_text="xaxis title")


fig1.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig1.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig1['layout'].update(
  height= 600,
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
            id='rga',
            figure=fig1,
        ),
        dcc.Graph(
            id='corte-agua',
            figure=fig,
        ),
    ])
