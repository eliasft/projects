"""
#########################################                            ########################################
########################################   PLOT TIEMPOS PERFORACION  #########################################
########################################                            #########################################
"""
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from entrada import user_input
input_campo = user_input.input_campo
len_perfil = user_input.len_perfil

from analisis.dca_main import hidrocarburo, gas, condensado, agua, master_df, serie_campo

from analisis import dca_analogos
gasto_analogos = dca_analogos.gasto_analogos
unique_analogos = dca_analogos.unique_analogos
serie_analogos = dca_analogos.serie_analogos

df_filtrado=pd.DataFrame()
df_filtrado=gasto_analogos[(gasto_analogos.di_hyp >= master_df.di_hyp.quantile(.30)) & (gasto_analogos.Qi_hist <= master_df.Qi_hist.quantile(0.80))]
df_filtrado=df_filtrado.sort_values('Qi_hist',ascending=False)
unique_filtro=pd.unique(df_filtrado.campo)
print('Número de campos muestra para ' +str(input_campo)+': '+str(len(unique_analogos)))
print('Numero de campos analogos: '+str(len(unique_filtro)))

fig, ax = plt.subplots(figsize=(15,10))

for campo in unique_filtro:

    perfil_analogo=df_filtrado[df_filtrado.campo==campo]

    qi=float(perfil_analogo.Qi_hist)
    b=float(perfil_analogo.b)
    di=float(perfil_analogo.di_hyp)

    #print(qi,b,di)

    perfil=pd.DataFrame()
    mes=range(0,500)

    for t in mes:

        qo=qi/((1.0+b*di*t)**(1.0/b))

        Q={'mes':[t],'Qo':[qo]}
        Q=pd.DataFrame(Q)

        perfil=perfil.append(Q)

    perfil=perfil.set_index('mes')
    ax.plot(perfil.index,perfil.Qo,label=campo,alpha=0.8,linestyle='dotted',linewidth=1)

############# PLOT PERFILES ANALOGOS ###########

qi=float(master_df.Qi_hist.mean())
b=float(master_df.b.mean())
di=float(master_df.di_hyp.mean())


perfil_campo=pd.DataFrame()
mes=range(0,500)

for t in mes:
    qo=qi/((1.0+b*di*t)**(1.0/b))

    Q={'mes':[t],'Qo':[qo]}
    Q=pd.DataFrame(Q)

    perfil_campo=perfil_campo.append(Q)

perfil_campo=perfil_campo.set_index('mes')


ax.plot(perfil_campo.index,perfil_campo.Qo,label=input_campo,alpha=1,linewidth=2)
ax.set_xlabel('Mes')
ax.set_ylabel('Qo')
plt.xlim(0,len_perfil)
plt.ylim(0);
plt.title('Perfiles tipo | Campos ANALOGOS | ' +str(input_campo),
          fontsize=18,
          fontweight='semibold')


plt.legend(loc='upper right',fontsize='medium',ncol=2)
plt.show()



dfx=pd.DataFrame()
dfx=serie_analogos[(serie_analogos.di_hyp >= master_df.di_hyp.quantile(.30)) & (serie_analogos.Qi_hist <= master_df.Qi_hist.quantile(0.80))]
dfx=dfx.reset_index()

print(len(pd.unique(dfx.campo)))

dfxx=pd.DataFrame()
dfxx=serie_campo.groupby(by='mes').mean().reset_index()

############# PLOT GASTO HISTORICO ANALOGOS ###########
fig1, ax1 = plt.subplots(figsize=(15,10))

for campo in unique_filtro:

    plot_analogo=dfx[dfx.campo==campo]
    ax1.plot(plot_analogo.mes,plot_analogo[hidrocarburo],label=campo,alpha=0.8,linestyle='dotted',linewidth=1)

ax1.plot(dfxx.mes,dfxx[hidrocarburo],label=input_campo,alpha=1,linewidth=2)
ax1.set_xlabel('Mes')
ax1.set_ylabel('Qo')
plt.xlim(0,len_perfil)
plt.ylim(0);
plt.title('Historial de produccion | Campos ANALOGOS | ' +str(input_campo),
          fontsize=18,
          fontweight='semibold')

plt.legend(loc='upper right',fontsize='medium',ncol=2)#mode='expand')
plt.show()

############# PLOT Qi ANALOGOS ###########
fig2, ax2 = plt.subplots(figsize=(15,8))
sns.distplot(master_df.Qi_hist, hist=False, kde=True,label=input_campo,
             #hist_kws = {'alpha':0.1},
             kde_kws = {'shade': True, 'bw':'scott'})
sns.distplot(df_filtrado.Qi_hist, hist=False, kde=True,label='Analogos',
             kde_kws = {'shade': True, 'bw':'scott'})
ax2.set_xlabel(hidrocarburo)
ax2.set_ylabel('Probabilidad')
plt.title('Gasto inicial Qi | Analogos | ' +str(input_campo))
plt.legend(loc='best')

############# PLOT Qo ANALOGOS ###########
fig3, ax3 = plt.subplots(figsize=(15,8))
sns.distplot(serie_campo[hidrocarburo], hist=False, kde=True,label=input_campo, kde_kws = {'shade': True, 'bw':'scott'})
sns.distplot(dfx[hidrocarburo], hist=False, kde=True,label='Analogos', kde_kws = {'shade': True, 'bw':'scott'})
ax3.set_xlabel(hidrocarburo)
ax3.set_ylabel('Probabilidad')
plt.title('Gasto mensual | Analogos | ' +str(input_campo))
plt.legend(loc='best')
