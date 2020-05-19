"""
#########################################                            ########################################
########################################      PLOT ANALISIS        #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from entrada import user_input
input_campo = user_input.input_campo

from analisis import pozos_tipo
tipos = pozos_tipo.tipos

from analisis import dca_main
serie_campo = dca_main.serie_campo
serie_status = dca_main.serie_status

from analisis.dca_main import hidrocarburo, gas, condensado, agua

########### DISPERSION DEL GASTO INICIAL #############

fig4, ax4 = plt.subplots(figsize=(15,10))
sns.scatterplot(x='first_oil', y='Qi_hist',
                 hue='tipo',
                 size='tipo',
                 sizes=(300,800),
                 alpha=0.8,
                 legend='brief',
                 palette='Set1',
                 style="tipo",
                 markers=True,
                 data=tipos)
ax4.set_xlabel('First Oil')
ax4.set_ylabel('Qi')
plt.title('Dispersion del master_df inicial Qi -  ' +str(hidrocarburo)+' para '+str(input_campo),
          fontsize=18,
          fontweight='bold')
plt.legend(loc='upper right',
           fontsize='small',)
           #bbox_to_anchor=(1.0,1.0, 0.00, 0.00),ncol=1)
plt.show()

########## DISTRIBUCION DEL GASTO INICIAL Qi #############

Q_plot = sns.FacetGrid(tipos, col="tipo",hue='tipo',height=5, aspect=0.9)
Q_plot.map(sns.distplot, 'Qi_hist')
plt.subplots_adjust(top=0.8)
Q_plot.fig.suptitle('Distribucion del master_df inicial - Qi - '+str(input_campo),
              fontsize=18,
              fontweight='bold')

d_plot = sns.FacetGrid(tipos,col='tipo',hue='tipo',height=5,aspect=0.9)
d_plot.map(sns.distplot, 'di_hyp')
plt.subplots_adjust(top=0.8)
d_plot.fig.suptitle('Distribucion de la declinacion inicial - di - '+str(input_campo),
               fontsize=18,
              fontweight='bold')

#Distribucion del master_df historico vs pronosticado
fig2, ax2 = plt.subplots(figsize=(15,8))
sns.distplot(serie_campo[hidrocarburo],hist=False, kde=True, label='Qo historico',kde_kws = {'shade': True})#,'bw':'scott'})
sns.distplot(serie_campo.hiperbolica,hist=False, kde=True,label='Hyperbolic Predicted', kde_kws = {'shade': True})#,'bw':'scott'})
sns.distplot(serie_campo.harmonica,hist=False, kde=True, label='Harmonic Predicted',  kde_kws = {'shade': True})#,'bw':'scott'})
sns.distplot(serie_campo.exponencial,hist=False, kde=True, label='Exponential Predicted', kde_kws = {'shade': True})#,'bw':'scott'})
#plt.hist( alpha=0.5, label='Qo historico',density=True)
#plt.hist(serie_campo.hiperbolica, alpha=0.3, label='Hyperbolic Predicted',density=True)#,cumulative=True)
#plt.hist(serie_campo.harmonica, alpha=0.3, label='Harmonic Predicted',density=True)
#plt.hist(serie_campo.exponencial, alpha=0.3, label='Exponential Predicted',density=True)
ax2.set_xlabel('Gasto Qo')
ax2.set_ylabel('Densidad')
plt.title(str(hidrocarburo) +' Qo historico vs Pronosticado para el campo ' +str(input_campo),
          fontsize=18,
          fontweight='bold')
plt.legend(loc='best')

###########  DISTRIBUCION GASTO HISTORICO VS PRONOSTICADO  ###########

if hidrocarburo == 'aceite_Mbd':

    fig2a, ax2a = plt.subplots(figsize=(15,8))
    sns.distplot(serie_campo[gas], hist=False, kde=True,label='Qg historico', kde_kws = {'shade': True,'bw':'scott'})
    sns.distplot(serie_campo.gas_hiperbolica, hist=False, kde=True,label='Hyperbolic Gas', kde_kws = {'shade': True,'bw':'scott'})
    sns.distplot(serie_campo.gas_harmonica, hist=False, kde=True,label='Harmonic Gas', kde_kws = {'shade': True,'bw':'scott'})
    sns.distplot(serie_campo.gas_exponencial, hist=False, kde=True,label='Exponential Gas', kde_kws = {'shade': True,'bw':'scott'})
    #plt.hist(serie_campo[gas], alpha=0.5, label='Qg historico',density=True)
    #plt.hist(serie_campo.gas_hiperbolica, alpha=0.5, label='Hyperbolic Gas',density=True)#,cumulative=True)
    #plt.hist(serie_campo.gas_harmonica, alpha=0.5, label='Harmonic Gas',density=True)
    #plt.hist(serie_campo.gas_exponencial, alpha=0.5, label='Exponential Gas',density=True)
    ax2a.set_xlabel('Gasto Qg')
    ax2a.set_ylabel('Densidad')
    plt.title(' Qg histórico vs Pronosticado para el campo ' +str(input_campo),
              fontsize=18,
              fontweight='bold')
    plt.legend(loc='best')
    plt.show()


###########  GRAFICAS DE STATUS  ###########

############# PLOT POZOS POR TIPO ###########
    distribucion=pd.DataFrame()
    distribucion=tipos.tipo.value_counts()

    fig3 = plt.figure(figsize=(15, 15))
    fig3.suptitle('Status del campo '+str(input_campo),
                    fontsize=18,
                    fontweight='semibold')
    plt.subplots_adjust(top=0.90)


    grid3 = plt.GridSpec(3, 1, hspace=0.2, wspace=0.4)
    ax_distribucion = fig3.add_subplot(grid3[0, 0])
    ax_estado = fig3.add_subplot(grid3[1, 0],sharex=ax_distribucion)
    ax_clasificacion = fig3.add_subplot(grid3[2, 0],sharex=ax_estado)


    ax_distribucion.title.set_text('Clasificacion por tipo')
    ax_distribucion.barh(y=distribucion.keys(),
                            width=distribucion.values,
                            color=['Blue','Green','Red'],
                            alpha=0.5,
                           label='Numero de pozos')


    ############# PLOT ESTADO ACTUAL ###########

    serie_todos=serie_status

    estado=serie_todos.estado_actual.value_counts()
    status=pd.DataFrame(index=estado.index,data=estado)


    ax_estado.title.set_text('Estado actual')
    ax_estado.barh(y=status.index,
                     width=status.estado_actual)


    ############# PLOT TRAYECTORIA ###########

    clasif=serie_todos.trayectoria.value_counts()
    clasificacion=pd.DataFrame(index=clasif.index,data=clasif)

    #for x in clasif:
     #   clasificacion.loc[x,'pozos']=len(serie_todos[serie_todos.trayectoria == str(x)])

    ax_clasificacion.title.set_text('Trayectoria')
    ax_clasificacion.barh(y=clasificacion.index,
                            width=clasificacion.trayectoria)

    ax_clasificacion.set_xlabel('Numero de pozos')
