"""
#########################################                            ########################################
########################################   PLOT TIEMPOS PERFORACION  #########################################
########################################                            #########################################
"""


import matplotlib.pyplot as plt
import seaborn as sns

from productividad.input import user_input
input_campo = user_input.input_campo

from productividad.analisis.dca_main import hidrocarburo, gas, condensado, agua

from productividad.analisis import dca_main
serie_muestra = dca_main.serie_muestra
serie_campo = dca_main.serie_campo
fecha_muestra = dca_main.fecha_muestra

fig1, ax1 = plt.subplots(figsize=(15,8))
fff=serie_campo.fecha.min()
sns.distplot(serie_campo[hidrocarburo],hist=False, kde=True, label='Qo since ' +str(fff.year), kde_kws = {'shade': True,'bw':'scott'})
sns.distplot(serie_muestra[hidrocarburo],hist=False, kde=True, label='Qo since First oil > '+str(fecha_muestra.year),kde_kws = {'shade': True,'bw':'scott'})
#plt.hist(serie_campo[hidrocarburo], alpha=0.6, label='Qo since ' +str(fff.year),density=True)
#plt.hist(serie_muestra[hidrocarburo], alpha=0.3, label='Qo since First oil > '+str(input_fecha.year),density=True)
ax1.set_xlabel('Gasto Qo')
ax1.set_ylabel('Densidad')
plt.title('Qo historico vs Qo since First Oil en el campo '+str(input_campo))
plt.legend(loc='upper right')
plt.show

############# Gasto inicial Qi

fig2, ax2 = plt.subplots(figsize=(15,8))
sns.distplot(serie_campo.Qi_hist,hist=False, kde=True, label='Qi since ' +str(fff.year), kde_kws = {'shade': True,'bw':'scott'})
sns.distplot(serie_muestra.Qi_desde,hist=False, kde=True, label='Qi since First oil > ' +str(fecha_muestra.year), kde_kws = {'shade': True,'bw':'scott'})
#plt.hist(serie_campo.Qi_hist, alpha=0.6, label='Qi since ' +str(fff.year),density=True)
#plt.hist(serie_muestra.Qi_desde, alpha=0.3, label='Qi since First oil > ' +str(input_fecha.year),density=True)
ax2.set_xlabel('Gasto inicial Qi')
ax2.set_ylabel('Densidad')
plt.title('Qi historico vs Qi since First Oil en el campo '+str(input_campo))
plt.legend(loc='upper right')
plt.show()
