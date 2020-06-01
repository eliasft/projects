"""
#########################################                            ########################################
########################################    ESTIMACION DE PDP       #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
from datetime import date

from analisis import pozos_tipo

df=pozos_tipo.tipos

df=df[df.estado_actual == 'PRODUCTOR']

unique_pozos=list(pd.unique(df.index))

mes_target=int(df.mes_max.max())

pdp=pd.DataFrame()

for pozo in unique_pozos:

        mes_actual = int(df.loc[pozo,'mes_max'])
        t = mes_target - mes_actual
        qi = df.loc[pozo,'Qi_hist']
        b = df.loc[pozo,'b']
        di = df.loc[pozo,'di_hyp']

        for mes in range(mes_actual,mes_target):

              qo = qi / ((1.0+b*di*mes)**(1.0/b))
              pdp.loc[mes,pozo] = qo
              #pdp[:,pozo]=pdp.append(declinacion)
