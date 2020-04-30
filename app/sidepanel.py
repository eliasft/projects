import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas
from dash.dependencies import Input, Output

import plotly.express as px

from app import app
import tab1, tab2, tab3, tab4, tab5, tab6
import transforms

df = transforms.df
min_p=df.profundidad_total.min()
max_p=df.profundidad_total.max()

layout = html.Div([
    html.H1('Analisis de productividad')
    ,dbc.Row([dbc.Col(
        html.Div([
         html.H2('Filters')
        , dcc.Checklist(id='direccional'
        , options = [
            {'label':'Solo direccional ', 'value':'Y'}
        ])
        ,html.Div([html.H5('Filtro Profundidad Total')
            ,dcc.RangeSlider(id='profundidad-slider'
                            ,min = min_p
                            ,max= max_p
                            , marks = {1700: '1700m',
                                       2000: '2000m',
                                       2300: '2300m',
                                       2600: '2600m',
                                       2900: '2900m',
                                       3200: '3200m',
                                       3500: '3400m',
                                       }
                            , value = [1700,3500]
                            )

                            ])

        ], style={'marginBottom': 50, 'marginTop': 25, 'marginLeft':15, 'marginRight':15})
    , width=3)

    ,dbc.Col(html.Div([
            dcc.Tabs(id="tabs", value='tab-1', children=[
                    dcc.Tab(label='Data Table', value='tab-1'),
                    dcc.Tab(label='Profundidad vs Gasto inicial', value='tab-2'),
                    dcc.Tab(label='Exploracion de variables', value='tab-3'),
                    dcc.Tab(label='Tabla', value='tab-4'),
                    dcc.Tab(label='Dias de perforacion', value='tab-5'),
                    dcc.Tab(label='Distribucion', value='tab-6'),
                ])
            , html.Div(id='tabs-content')
        ]), width=9)])

    ])
