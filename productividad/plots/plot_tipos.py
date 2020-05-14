"""
#########################################                            ########################################
########################################      PLOT POZOS TIPO       #########################################
########################################                            #########################################
"""


import matplotlib.pyplot as plt
import seaborn as sns

from productividad.input import user_input
input_campo = user_input.input_campo
len_perfil = user_input.len_perfil

from productividad.analisis import pozos_tipo
perfil = pozos_tipo.perfil

from productividad.analisis.dca_main import hidrocarburo, gas, condensado, agua

fig6, ax6 = plt.subplots(figsize=(15,10))

ax6.plot(perfil.baja_L,linestyle='dotted',color='red',alpha=0.5)
ax6.plot(perfil.baja_M,label='Qi BAJA',linestyle='solid',color='red')
ax6.plot(perfil.baja_H,linestyle='dotted',color='red',alpha=0.5)
ax6.fill_between(perfil.index,perfil.baja_L,perfil.baja_H,alpha=0.05,color='red')

ax6.plot(perfil.media_L,linestyle='dotted',color='blue',alpha=0.5)
ax6.plot(perfil.media_M,label='Qi MEDIA',linestyle='solid',color='blue')
ax6.plot(perfil.media_H,linestyle='dotted',color='blue',alpha=0.5)
ax6.fill_between(perfil.index,perfil.media_L,perfil.media_H,alpha=0.05,color='blue')

ax6.plot(perfil.alta_L,linestyle='dotted',color='green',alpha=0.5)
ax6.plot(perfil.alta_M,label='Qi ALTA',linestyle='solid',color='green')
ax6.plot(perfil.alta_H,linestyle='dotted',color='green',alpha=0.5)
ax6.fill_between(perfil.index,perfil.alta_L,perfil.alta_H,alpha=0.05,color='green')

ax6.legend(loc='best',fontsize='medium')
ax6.set_ylabel('Qo (Mbd / MMpcd)'+str(hidrocarburo))
ax6.set_ylim(0)

plt.xlim(0,len_perfil)
plt.title('Perfil de produccion - Pozos tipo - ' +str(input_campo),
         fontsize=24,
         fontweight='semibold')
plt.show()

fig7, ax7 = plt.subplots(figsize=(15,10))

ax7.plot(perfil.EUR_baja_L,linestyle='dotted',color='red',alpha=0.3)
ax7.plot(perfil.EUR_baja_M,label='EUR BAJA',linestyle='dashed',color='red',alpha=0.7)
ax7.plot(perfil.EUR_baja_H,linestyle='dotted',color='red',alpha=0.3)
ax7.fill_between(perfil.index,perfil.EUR_baja_L,perfil.EUR_baja_H,alpha=0.03,color='red')

ax7.plot(perfil.EUR_media_L,linestyle='dotted',color='blue',alpha=0.3)
ax7.plot(perfil.EUR_media_M,label='EUR MEDIA',linestyle='dashed',color='blue',alpha=0.7)
ax7.plot(perfil.EUR_media_H,linestyle='dotted',color='blue',alpha=0.3)
ax7.fill_between(perfil.index,perfil.EUR_media_L,perfil.EUR_media_H,alpha=0.03,color='blue')

ax7.plot(perfil.EUR_alta_L,linestyle='dotted',color='green',alpha=0.3)
ax7.plot(perfil.EUR_alta_M,label='EUR ALTA',linestyle='dashed',color='green',alpha=0.7)
ax7.plot(perfil.EUR_alta_H,linestyle='dotted',color='green',alpha=0.3)
ax7.fill_between(perfil.index,perfil.EUR_alta_L,perfil.EUR_alta_H,alpha=0.03,color='green')

ax7.legend(loc='best',fontsize='medium')
ax7.set_ylabel('EUR (MMb / MMMpc)')
ax7.set_ylim(0)

plt.xlim(0,len_perfil)
plt.title('EUR - Pozos tipo - ' +str(input_campo),
         fontsize=24,
         fontweight='semibold')
plt.show()

########### SUBPLOTS POZOS TIPO

fig = plt.figure(figsize=(12, 20))
fig.suptitle('Pozos tipo - Curvas de declinacion '+str(input_campo),
       fontsize=24,
       fontweight='semibold')
plt.subplots_adjust(top=0.94)

grid = plt.GridSpec(3, 1, hspace=0.2, wspace=0.2)
plot_alta = fig.add_subplot(grid[0, 0])
plot_media = fig.add_subplot(grid[1, 0])
plot_baja = fig.add_subplot(grid[2, 0])


plot_alta.title.set_text('ALTA Productividad - ' +str(hidrocarburo))
plot_alta.plot(perfil.alta_L,label=None,linestyle='dotted',color='green',alpha=0.5)
plot_alta.plot(perfil.alta_M,label='Qi ALTA',linestyle='solid',color='green')
plot_alta.plot(perfil.alta_H,label=None,linestyle='dotted',color='green',alpha=0.5)
plot_alta.fill_between(perfil.index,perfil.alta_L,perfil.alta_H,alpha=0.05,color='green')

plot_alta.set_ylabel('Qo (Mbd / MMpcd) '+str(hidrocarburo))

ax_alta = plot_alta.twinx()

ax_alta.plot(perfil.EUR_alta_L,label=None,linestyle='dotted',color='green',alpha=0.2)
ax_alta.plot(perfil.EUR_alta_M,label='EUR ALTA',linestyle='dashed',color='green',alpha=0.4)
ax_alta.plot(perfil.EUR_alta_H,label=None,linestyle='dotted',color='green',alpha=0.2)
ax_alta.fill_between(perfil.index,perfil.EUR_alta_L,perfil.EUR_alta_H,alpha=0.05,color='green')

ax_alta.set_ylabel('EUR (MMb / MMMpc)')

plot_alta.legend(loc='best',fontsize='medium')
ax_alta.legend(loc='best',fontsize='medium')
plot_alta.axes.set_xlim(0,len_perfil)
plot_alta.axes.set_ylim(0)

plot_media.title.set_text('MEDIA Productividad - ' +str(hidrocarburo))
plot_media.plot(perfil.media_L,label=None,linestyle='dotted',color='blue',alpha=0.5)
plot_media.plot(perfil.media_M,label='Qi MEDIA',linestyle='solid',color='blue')
plot_media.plot(perfil.media_H,label=None,linestyle='dotted',color='blue',alpha=0.5)
plot_media.fill_between(perfil.index,perfil.media_L,perfil.media_H,alpha=0.05,color='blue')

plot_media.set_ylabel('Qo (Mbd / MMpcd) '+str(hidrocarburo))

ax_media = plot_media.twinx()

ax_media.plot(perfil.EUR_media_L,label=None,linestyle='dotted',color='blue',alpha=0.2)
ax_media.plot(perfil.EUR_media_M,label='EUR MEDIA',linestyle='dashed',color='blue',alpha=0.4)
ax_media.plot(perfil.EUR_media_H,label=None,linestyle='dotted',color='blue',alpha=0.2)
ax_media.fill_between(perfil.index,perfil.EUR_media_L,perfil.EUR_media_H,alpha=0.05,color='blue')

ax_media.set_ylabel('EUR (MMb / MMMpc)')

plot_media.legend(loc='best',fontsize='medium')
ax_media.legend(loc='best',fontsize='medium')
plot_media.axes.set_xlim(0,len_perfil)
plot_media.axes.set_ylim(0)

plot_baja.title.set_text('BAJA Productividad - ' +str(hidrocarburo))
plot_baja.plot(perfil.baja_L,label=None,linestyle='dotted',color='red',alpha=0.5)
plot_baja.plot(perfil.baja_M,label='Qi BAJA',linestyle='solid',color='red')
plot_baja.plot(perfil.baja_H,label=None,linestyle='dotted',color='red',alpha=0.5)
plot_baja.fill_between(perfil.index,perfil.baja_L,perfil.baja_H,alpha=0.05,color='red')

plot_baja.set_ylabel('Qo (Mbd / MMpcd) '+str(hidrocarburo))

ax_baja = plot_baja.twinx()

ax_baja.plot(perfil.EUR_baja_L,label=None,linestyle='dotted',color='red',alpha=0.2)
ax_baja.plot(perfil.EUR_baja_M,label='EUR BAJA',linestyle='dashed',color='red',alpha=0.4)
ax_baja.plot(perfil.EUR_baja_H,label=None,linestyle='dotted',color='red',alpha=0.2)
ax_baja.fill_between(perfil.index,perfil.EUR_baja_L,perfil.EUR_baja_H,alpha=0.05,color='red')

ax_baja.set_ylabel('EUR (MMb / MMMpc)')

plot_baja.legend(loc='best',fontsize='medium')
ax_baja.legend(loc='best',fontsize='medium')
plot_baja.axes.set_xlim(0,len_perfil)
plot_baja.axes.set_ylim(0)

plot_baja.set_xlabel('Mes')
