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

df_pozos = transforms.df_pozos

df_activos = transforms.df_activos

status = transforms.status

arq = transforms.arq


pozos1= px.pie(df_pozos,values=df_pozos.values, names=df_pozos.index, title='Productores vs Secos')

pozos2= px.pie(df_activos,values=df_activos.values, names=df_activos.index, title='Activos vs Cerrados')

pozos3= px.pie(status,values=status.values, names=status.index, title='Estado actual')

pozos4=px.bar(arq, x=arq.values, y=arq.index, orientation='h', title='Arquitecturas de pozo')


layout = html.Div([
        dcc.Graph(
            id='productores',
            figure=pozos1
        ),

        dcc.Graph(
            id='activos',
            figure=pozos2
        ),

        dcc.Graph(
            id='estado-actual',
            figure=pozos3
        ),

        dcc.Graph(
            id='arquitecturas',
            figure=pozos4
        ),


    ])
