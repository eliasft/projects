"""
#########################################                            ########################################
########################################   PLOT TIEMPOS PERFORACION  #########################################
########################################                            #########################################
"""


import matplotlib.pyplot as plt
import seaborn as sns

from productividad.input import user_input
input_campo = user_input.input_campo

from productividad.analisis import dca_main
serie_resumen = dca_main.serie_resumen

from productividad.analisis.dca_main import hidrocarburo, gas, condensado, agua

fig1, ax1 = plt.subplots(figsize=(15,8))
sns.distplot(serie_resumen.dias_perforacion,
             hist=False,
             kde=True,
             color='Black',
             label='Dias Perforacion',
             kde_kws = {'shade': True,
                        #'cumulative':True,
                        'bw':'scott'})
ax1.set_xlabel('Dias Perforacion')
ax1.set_ylabel('Probabilidad')
plt.title('Dias de perforacion por pozo en el campo ' +str(input_campo),
          fontsize=18,
          fontweight='semibold')
plt.legend(loc='best')
plt.show


fig2, ax2 = plt.subplots(figsize=(15,8))

sns.scatterplot(x='dias_perforacion', y='profundidad_total',
             hue='estado_actual',
             #size='ultimo_estado_reportado',
             #sizes=(1000,2000),
             alpha=1,
             legend='brief',
             palette='coolwarm',
             style="estado_actual",
             markers=True,
             data=serie_resumen,s=800)

ax2.set_xlabel('Dias de perforacion')
ax2.set_ylabel('Profundidad total')
plt.title('Dispersion de tiempos de perforacion para el campo '+str(input_campo),
          fontsize=18,
          fontweight='semibold')
plt.legend(loc='best',
           fontsize='small')
           #mode='expand',
           #bbox_to_anchor=(1.0,1.0, 0.00, 0.00),ncol=1)
plt.show
