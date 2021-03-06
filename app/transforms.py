import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash
from dash.dependencies import Input, Output
import dash_table
import pandas as pd

import plotly.express as px

df = pd.read_csv(r'/Users/fffte/Documents/GitHub/projects/productividad/output/serie_resumen.csv')

dfx = pd.read_csv(r'/Users/fffte/Documents/GitHub/projects/productividad/output/serie_campo.csv')

dfx=dfx.groupby(by='pozo').mean()
intervenciones=dfx.ano_de_perforacion.value_counts()
intervenciones.index=pd.Index.astype(intervenciones.index,dtype='int64')
intervenciones=intervenciones.sort_index(ascending=True)
