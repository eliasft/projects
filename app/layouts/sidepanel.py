import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas
from dash.dependencies import Input, Output

from app import app

from tabs import tab1, tab2
from database import transforms

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
        ,html.Div([html.H5('Price Slider')
            ,dcc.RangeSlider(id='profundidad-slider'
                            ,min = min_p
                            ,max= max_p
                            , marks = {1700: '1700m',
                                       1800: '1800m',
                                       1900: '1900m',
                                       2000: '2000m',
                                       2100: '2100m',
                                       2200: '2200m',
                                       2300: '2300m',
                                       2400: '2400m',
                                       2500: '2500m',
                                       2600: '2600m',
                                       2700: '2700m',
                                       2800: '2800m',
                                       2900: '2900m',
                                       3000: '3000m',
                                       3100: '3100m',
                                       3200: '3200m',
                                       3300: '3300m',
                                       3400: '3400m',
                                       }
                            , value = [1700,3400]
                            )

                            ])

        ], style={'marginBottom': 50, 'marginTop': 25, 'marginLeft':15, 'marginRight':15})
    , width=3)

    ,dbc.Col(html.Div([
            dcc.Tabs(id="tabs", value='tab-1', children=[
                    dcc.Tab(label='Data Table', value='tab-1'),
                    dcc.Tab(label='Scatter Plot', value='tab-2'),
                ])
            , html.Div(id='tabs-content')
        ]), width=9)])

    ])
