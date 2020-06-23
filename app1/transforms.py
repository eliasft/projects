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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os

directorio = os.getcwd()

df = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/serie_tipos.csv')

df_info = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/info_campo.csv')
df_info = df_info.transpose()
df_info = df_info.rename(columns={'':'Caracteristicas',0:'Datos'})

df_resumen = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/resumen.csv',
                        header=0,
                        index_col=0)

df_pozos=df_resumen.loc[['Pozos productores','Pozos secos']]

df_activos=df_resumen.loc[['Pozos activos','Pozos cerrados']]

status=df.estado_actual.value_counts()

arq=df.trayectoria.value_counts()

dfx = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/serie_campo.csv',
                        header=0,
                        index_col=0)
dfx=dfx.groupby(by='pozo').mean()
intervenciones=dfx.ano_de_perforacion.value_counts()
intervenciones.index=pd.Index.astype(intervenciones.index,dtype='int64')
intervenciones=intervenciones.sort_index(ascending=True)

dfxx = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/serie_campo.csv')
dfxx['fecha']=pd.to_datetime(dfxx['fecha'],dayfirst=True)
dfxx=dfxx.set_index('fecha')
produccion=dfxx.resample('Y').mean()
produccion['year']=produccion.index.year


df_tipos = pd.read_csv(r'/Users/felias/Documents/GitHub/projects/output/serie_tipos.csv',
                header=0,
                index_col=0
                )

tipos=df_tipos.tipo.value_counts()
