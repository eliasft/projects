
"""
#########################################                            ########################################
########################################    ANALISIS POZOS TIPO     #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
from datetime import date

from productividad.input import user_input

len_perfil = user_input.len_perfil
alta = user_input.alta
media = user_input.media
baja = user_input.baja


from productividad.analisis import dca_main
gasto = dca_main.gasto
serie_resumen = dca_main.serie_resumen


#########################  POZOS TIPO - PRONOSTICO DE PRODUCCION Qo   #####################

###### Dataframe referencia de fechas

periodo=np.arange(start=0,stop=len_perfil,step=1)
fechas=pd.date_range(start=pd.to_datetime(date.today()),freq='M',periods=len_perfil,normalize=True,closed='left')

df=pd.DataFrame()

df['fecha']=fechas
df['mes']=pd.DatetimeIndex(fechas).month
df['ano']=pd.DatetimeIndex(fechas).year
df['dias']=pd.DatetimeIndex(fechas).day
df['periodo']=periodo

###### Valores Medios

q_baja=gasto.Qi_hist.quantile(baja)
q_media=gasto.Qi_hist.quantile(media)
q_alta=gasto.Qi_hist.quantile(alta)

#d_baja=gasto.di.quantile(baja)
d_media=gasto.di_hyp.quantile(media)
#d_media=gasto.di_harm.quantile(media)
#d_alta=gasto.di.quantile(alta)

d=gasto.di_hyp.mean()
#d=gasto.di_harm.mean()

b=gasto.b.mean()

##################     SUBSET DE POZOS TIPO      #######################

######### POZOS TIPO 1 - Qi BAJA #########

criterio1=(gasto['Qi_hist'] <= q_baja)
tipo1=gasto.loc[criterio1]

q1_baja=tipo1.Qi_hist.quantile(baja)
q1_media=tipo1.Qi_hist.quantile(media)
q1_alta=tipo1.Qi_hist.quantile(alta)

#d1_baja=tipo1.di_hyp.quantile(baja)
#d1_media=tipo1.di_hyp.quantile(media)
#d1_alta_1=tipo1.di_hyp.quantile(alta)

d1=tipo1.di_hyp.mean()
b1=tipo1.b.mean()


######### POZOS TIPO 2 - Qi MEDIA #########

criterio2=(gasto['Qi_hist'] > q_baja) & (gasto['Qi_hist'] < q_alta)
tipo2=gasto.loc[criterio2]

q2_baja=tipo2.Qi_hist.quantile(baja)
q2_media=tipo2.Qi_hist.quantile(media)
q2_alta=tipo2.Qi_hist.quantile(alta)

#d2_baja=tipo2.di_hyp.quantile(baja)
#d2_media=tipo2.di_hyp.quantile(media)
#d2_alta=tipo2.di_hyp.quantile(alta)

d2=tipo2.di_hyp.mean()
b2=tipo2.b.mean()


######### POZOS TIPO 3 - Qi ALTA #########

criterio3=(gasto['Qi_hist'] >= q_alta)
tipo3=gasto.loc[criterio3]

q3_baja=tipo3.Qi_hist.quantile(baja)
q3_media=tipo3.Qi_hist.quantile(media)
q3_alta=tipo3.Qi_hist.quantile(alta)

#d3_baja=tipo3.di_hyp.quantile(baja)
#d3_media_3=tipo3.di_hyp.quantile(media)
#d3_alta=tipo3.di_hyp.quantile(alta)

d3=tipo3.di_hyp.mean()
b3=tipo3.b.mean()

tipo1.loc[:,'tipo']='BAJA'
tipo2.loc[:,'tipo']='MEDIA'
tipo3.loc[:,'tipo']='ALTA'

tipos=pd.DataFrame()
tipos=tipos.append([tipo1,tipo2,tipo3])

dfx=serie_resumen.merge(tipos[['tipo']], how='left',on='pozo')

perfil=pd.DataFrame()

for x in df:

    perfil['mes']=df.periodo

    perfil['baja_L']=(q1_baja/((1.0+b1*d1*df.periodo)**(1.0/b1)))
    perfil['baja_M']=(q1_media/((1.0+b1*d1*df.periodo)**(1.0/b1)))
    perfil['baja_H']=(q1_alta/((1.0+b1*d1*df.periodo)**(1.0/b1)))

    perfil['media_L']=(q2_baja/((1.0+b2*d2*df.periodo)**(1.0/b2)))
    perfil['media_M']=(q2_media/((1.0+b2*d2*df.periodo)**(1.0/b2)))
    perfil['media_H']=(q2_alta/((1.0+b2*d2*df.periodo)**(1.0/b2)))

    perfil['alta_L']=(q3_baja/((1.0+b3*d3*df.periodo)**(1.0/b3)))
    perfil['alta_M']=(q3_media/((1.0+b3*d3*df.periodo)**(1.0/b3)))
    perfil['alta_H']=(q3_alta/((1.0+b3*d3*df.periodo)**(1.0/b3)))

perfil=perfil.set_index('mes')

for x in perfil.columns:

    perfil['EUR_'+str(x)]=(perfil[x].cumsum())*30/1_000

d = {'Qi_hist': [tipo1.Qi_hist.mean(), tipo2.Qi_hist.mean(),tipo3.Qi_hist.mean()],
     'Qi_hyp': [tipo1.Qi_hyp.mean(), tipo2.Qi_hyp.mean(),tipo3.Qi_hyp.mean()],
     'Qi_harm': [tipo1.Qi_harm.mean(), tipo2.Qi_harm.mean(),tipo3.Qi_harm.mean()],
     'Qi_exp': [tipo1.Qi_exp.mean(), tipo2.Qi_exp.mean(),tipo3.Qi_exp.mean()],
     'b': [tipo1.b.mean(), tipo2.b.mean(),tipo3.b.mean()],
     'di_hyp': [tipo1.di_hyp.mean(), tipo2.di_hyp.mean(),tipo3.di_hyp.mean()],
     'di_harm': [tipo1.di_harm.mean(), tipo2.di_harm.mean(),tipo3.di_harm.mean()],
     'di_exp': [tipo1.di_exp.mean(), tipo2.di_exp.mean(),tipo3.di_exp.mean()],
     'error_Qi_hyp':[tipo1.error_Qi_hyp.mean(), tipo2.error_Qi_hyp.mean(),tipo3.error_Qi_hyp.mean()],
     'error_Qi_harm':[tipo1.error_Qi_harm.mean(), tipo2.error_Qi_harm.mean(),tipo3.error_Qi_harm.mean()],
     'error_Qi_exp':[tipo1.error_Qi_exp.mean(), tipo2.error_Qi_exp.mean(),tipo3.error_Qi_exp.mean()],
     'error_di_hyp':[tipo1.error_di_hyp.mean(), tipo2.error_di_hyp.mean(),tipo3.error_di_hyp.mean()],
     'error_di_harm':[tipo1.error_di_harm.mean(), tipo2.error_di_harm.mean(),tipo3.error_di_harm.mean()],
     'error_di_exp':[tipo1.error_di_exp.mean(), tipo2.error_di_exp.mean(),tipo3.error_di_exp.mean()],
     'Qi_gas': [tipo1.Qi_gas.mean(), tipo2.Qi_gas.mean(),tipo3.Qi_gas.mean()],
     'b_gas': [tipo1.b_gas.mean(), tipo2.b_gas.mean(),tipo3.b_gas.mean()],
     'di_gas': [tipo1.di_gas.mean(), tipo2.di_gas.mean(),tipo3.di_gas.mean()],
     'Qi_condensado': [tipo1.Qi_condensado.mean(), tipo2.Qi_condensado.mean(),tipo3.Qi_condensado.mean()],
     'b_condensado': [tipo1.b_condensado.mean(), tipo2.b_condensado.mean(),tipo3.b_condensado.mean()],
     'RSS_exponencial': [tipo1.RSS_exponencial.mean(), tipo2.RSS_exponencial.mean(),tipo3.RSS_exponencial.mean()],
     'RSS_hiperbolica': [tipo1.RSS_hiperbolica.mean(), tipo2.RSS_hiperbolica.mean(),tipo3.RSS_hiperbolica.mean()],
     'RSS_harmonica': [tipo1.RSS_harmonica.mean(), tipo2.RSS_harmonica.mean(),tipo3.RSS_harmonica.mean()],
     'RSS_gas_exponencial': [tipo1.RSS_gas_exponencial.mean(), tipo2.RSS_gas_exponencial.mean(),tipo3.RSS_gas_exponencial.mean()],
     'RSS_gas_hiperbolica': [tipo1.RSS_gas_hiperbolica.mean(), tipo2.RSS_gas_hiperbolica.mean(),tipo3.RSS_gas_hiperbolica.mean()],
     'RSS_gas_harmonica': [tipo1.RSS_gas_harmonica.mean(), tipo2.RSS_gas_harmonica.mean(),tipo3.RSS_gas_harmonica.mean()]}

parametros = pd.DataFrame(data=d,index=['tipo1','tipo2','tipo3'])


###########GENERACION DE ARCHIVO DE RMA's

criterio_rma=['INACTIVO',
             'TAPONADO',
             'CERRADO CON POSIB DE EXP.',
             'CERRADO CON POSIBILIDADES']

filtro_rma=tipos.estado_actual.isin(criterio_rma)
serie_rma=tipos[filtro_rma]
serie_rma=serie_rma.sort_values(by='mes_max',ascending=False)
