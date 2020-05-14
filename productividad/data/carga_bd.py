import pandas as pd
import numpy as np
import os

"""
#########################################                            ########################################
########################################   CARGA DE BASES DE DATOS  #########################################
########################################                            #########################################
"""


mx_bd=pd.read_csv(r'/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_full.csv',
                      usecols=['fecha',
                              'pozo',
                              'aceite_Mbd',
                              'gas_asociado_MMpcd',
                              'gas_no_asociado_MMpcd',
                              'condensado_Mbd',
                              'agua_Mbd',
                              'estado_actual',
                              'profundidad_total',
                              'profundidad_vertical',
                              'trayectoria',
                              'ano_de_perforacion',
                              'tipo_de_hidrocarburo',
                              'clasificacion',
                              'disponible',
                              'campo',
                              'cuenca',
                              'entidad',
                              'ubicacion',
                              'contrato'],
                              low_memory=True)

mx_reservas=pd.read_csv(r'/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_reservas.csv',
                      index_col=0)

mx_tiempos=pd.read_csv("/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_tiempos.csv",
                      index_col=0,
                      parse_dates=True,
                      keep_default_na=False)

mx_campos=pd.read_csv(r'/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_campos.csv')
