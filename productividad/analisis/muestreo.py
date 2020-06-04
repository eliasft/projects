
"""
#########################################                            ########################################
########################################     MUESTREO EN TIEMPO     #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
from datetime import date

from analisis import pozos_tipo

df_pozos = pozos_tipo.tipos

#################### SERIE MUESTRA (since predetermined date)

pozos_10 = df_pozos[df_pozos.first_oil >= '2010-01-01']
pozos_20 = df_pozos[df_pozos.first_oil >= '2000-01-01']
pozos_30 = df_pozos[df_pozos.first_oil >= '1990-01-01']


indice_prod = {
                'pozos_productores_all': len(df_pozos),
                'pozos_productores_10': len(pozos_10),
                'pozos_productores_20': len(pozos_20),
                'pozos_productores_30': len(pozos_30),
                'qi_all': df_pozos.Qi_hist.mean(),
                'qi_10': pozos_10.Qi_hist.mean(),
                'qi_20': pozos_20.Qi_hist.mean(),
                'qi_30': pozos_30.Qi_hist.mean(),
                'indice_prod_10': pozos_10.Qi_hist.mean() / df_pozos.Qi_hist.mean(),
                'indice_prod_20': pozos_20.Qi_hist.mean() / df_pozos.Qi_hist.mean(),
                'indice_prod_30': pozos_30.Qi_hist.mean() / df_pozos.Qi_hist.mean()
               }

print(indice_prod)
