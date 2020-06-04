
"""
#########################################                            ########################################
########################################    MUESTREO ARQUITECTURAS   #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
from datetime import date

from analisis import pozos_tipo
from analisis import muestreo

df_pozos = pozos_tipo.tipos

pozos_10 = muestreo.pozos_10
pozos_20 = muestreo.pozos_20
pozos_30 = muestreo.pozos_30

df_pozos = df_pozos.dropna(axis=0,how='any')

arquitecturas = pd.unique(df_pozos.trayectoria)

indice_arq=pd.DataFrame()

for x in arquitecturas:

    df = df_pozos[df_pozos.trayectoria == x]
    indice_arq.loc['Qi', x ] = df.Qi_hist.mean()
    indice_arq.loc['mes_max', x ] = df.mes_max.mean()
    indice_arq.loc['EUR',x] = df.EUR.mean()
    indice_arq.loc['pozos',x] = len(df)

    df_10 = pozos_10[pozos_10.trayectoria == x]
    indice_arq.loc['Qi', x +'_10' ] = df_10.Qi_hist.mean()
    indice_arq.loc['mes_max', x +'_10' ] = df_10.mes_max.mean()
    indice_arq.loc['EUR',x +'_10'] = df_10.EUR.mean()
    indice_arq.loc['pozos',x] = len(df_10)

    df_20 = pozos_20[pozos_20.trayectoria == x]
    indice_arq.loc['Qi', x +'_20' ] = df_20.Qi_hist.mean()
    indice_arq.loc['mes_max', x +'_20' ] = df_20.mes_max.mean()
    indice_arq.loc['EUR',x +'_20'] = df_20.EUR.mean()
    indice_arq.loc['pozos',x] = len(df_20)

    df_30 = pozos_30[pozos_30.trayectoria == x]
    indice_arq.loc['Qi', x +'_30' ] = df_30.Qi_hist.mean()
    indice_arq.loc['mes_max', x +'_30' ] = df_30.mes_max.mean()
    indice_arq.loc['EUR',x +'_30'] = df_30.EUR.mean()
    indice_arq.loc['pozos',x] = len(df_30)
