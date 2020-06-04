"""
#########################################                            ########################################
########################################   PLOT TIEMPOS PERFORACION  #########################################
########################################                            #########################################
"""


import matplotlib.pyplot as plt
import seaborn as sns

from entrada import user_input
input_campo = user_input.input_campo

from analisis.dca_main import hidrocarburo, gas, condensado, agua

from analisis.muestreo import df_pozos, pozos_10, pozos_20, pozos_30

'''
from analisis import dca_main
serie_muestra = dca_main.serie_muestra
serie_campo = dca_main.serie_campo
fecha_muestra = dca_main.fecha_muestra
'''

############# Gasto inicial Qi

fig1, ax1 = plt.subplots(figsize=(15,8))
sns.distplot(df_pozos.Qi_hist,hist=False, kde=True, label='First oil > ' +str(df_pozos.first_oil.min().year), kde_kws = {'shade': True,'bw':'scott'})
sns.distplot(pozos_10.Qi_hist,hist=False, kde=True, label='First oil > ' +str(pozos_10.first_oil.min().year), kde_kws = {'shade': True,'bw':'scott'})
sns.distplot(pozos_20.Qi_hist,hist=False, kde=True, label='First oil > ' +str(pozos_20.first_oil.min().year), kde_kws = {'shade': True,'bw':'scott'})
sns.distplot(pozos_30.Qi_hist,hist=False, kde=True, label='First oil > ' +str(pozos_30.first_oil.min().year), kde_kws = {'shade': True,'bw':'scott'})
ax1.set_xlabel('Gasto inicial Qi')
ax1.set_ylabel('Densidad')
plt.title('Muestreo en tiempo de gasto inicial '+str(input_campo))
plt.legend(loc='upper right')
plt.show()
