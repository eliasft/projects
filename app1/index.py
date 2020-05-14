import dash
import plotly
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import sidepanel, tab1, tab2, tab3, tab4, tab5, tab6
import transforms

import dash
from dash.dependencies import Input, Output
import dash_table
import pandas as pd

import dash
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import app

#app.layout = sidepanel.layout
#app.layout = no_panel.layout

app.layout = html.Div([
    html.H1('Caracterización del campo: Lacamango'),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Informacion general', value='tab-1'),
        dcc.Tab(label='Intervenciones ejecutadas en el campo', value='tab-2'),
        dcc.Tab(label='Historial Produccion vs Intervenciones', value='tab-3'),
        dcc.Tab(label='Produccion historica por tipo de hidrocarburo', value='tab-4'),
        dcc.Tab(label='Tipos', value='tab-5'),
        dcc.Tab(label='RGA y corte de agua', value='tab-6'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1.layout
    elif tab == 'tab-2':
       return tab2.layout
    elif tab == 'tab-3':
       return tab3.layout
    elif tab == 'tab-4':
       return tab4.layout
    elif tab == 'tab-5':
       return tab5.layout
    elif tab == 'tab-6':
       return tab6.layout

@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)

# Tab 2 callback
@app.callback(Output('page-2-content', 'children'),
              [Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug = True)
