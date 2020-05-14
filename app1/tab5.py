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

tipos = transforms.tipos

pozos_tipo= px.pie(tipos,
                   values=tipos.values,
                   names=tipos.index,
                   title='<b>Distribucion de pozos tipo</b>')


layout = html.Div([
        dcc.Graph(
            id='intervenciones-produccion',
            figure=pozos_tipo
        ),
    ])
