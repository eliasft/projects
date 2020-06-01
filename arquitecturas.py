import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from datetime import date
import os

folder=os.getcwd()

len_perfil=30*12

df_tipos=pd.read_csv(folder + '/output/veracruz/master_df.csv')

arq=df_tipos.trayectoria.value_counts()

keys=arq.index

for x in keys:
    print(x)
    

alta=0.80
media=0.50
baja=0.20

x=0

tipos=pd.DataFrame()

for x in keys:

    
    serie_tipos=df_tipos[df_tipos.trayectoria == str(x)]
    
    q_baja=serie_tipos.Qi_hist.quantile(baja)
    q_media=serie_tipos.Qi_hist.quantile(media)
    q_alta=serie_tipos.Qi_hist.quantile(alta)
    
    criterio1=(serie_tipos['Qi_hist'] <= q_baja)
    tipo1=serie_tipos.loc[criterio1]
    
    q1_baja=tipo1.Qi_hist.quantile(baja)
    q1_media=tipo1.Qi_hist.quantile(media)
    q1_alta=tipo1.Qi_hist.quantile(alta)
    
    #d1_baja=tipo1.di_hyp.quantile(baja)
    #d1_media=tipo1.di_hyp.quantile(media)
    #d1_alta_1=tipo1.di_hyp.quantile(alta)
    
    d1=tipo1.di_hyp.mean()
    b1=tipo1.b.mean()
    
    
    ######### POZOS TIPO 2 - Qi MEDIA #########
    
    criterio2=(serie_tipos['Qi_hist'] > q_baja) & (serie_tipos['Qi_hist'] < q_alta)
    tipo2=serie_tipos.loc[criterio2]
    
    q2_baja=tipo2.Qi_hist.quantile(baja)
    q2_media=tipo2.Qi_hist.quantile(media)
    q2_alta=tipo2.Qi_hist.quantile(alta)
    
    #d2_baja=tipo2.di_hyp.quantile(baja)
    #d2_media=tipo2.di_hyp.quantile(media)
    #d2_alta=tipo2.di_hyp.quantile(alta)
    
    d2=tipo2.di_hyp.mean()
    b2=tipo2.b.mean()
    
    
    ######### POZOS TIPO 3 - Qi ALTA #########
    
    criterio3=(serie_tipos['Qi_hist'] >= q_alta)
    tipo3=serie_tipos.loc[criterio3]
    
    q3_baja=tipo3.Qi_hist.quantile(baja)
    q3_media=tipo3.Qi_hist.quantile(media)
    q3_alta=tipo3.Qi_hist.quantile(alta)
    
    #d3_baja=tipo3.di_hyp.quantile(baja)
    #d3_media_3=tipo3.di_hyp.quantile(media)
    #d3_alta=tipo3.di_hyp.quantile(alta)
    
    d3=tipo3.di_hyp.mean()
    b3=tipo3.b.mean()
    
    tipo1.loc[:,'tipo']='BAJA_'+str(x)
    tipo2.loc[:,'tipo']='MEDIA_'+str(x)
    tipo3.loc[:,'tipo']='ALTA_'+str(x)
    
    tipos=tipos.append([tipo1,tipo2,tipo3])
    
tipos.to_csv(folder+'tipos_veracruz.csv')


    
periodo=np.arange(start=0,stop=len_perfil,step=1)
fechas=pd.date_range(start=pd.to_datetime(date.today()),freq='M',periods=len_perfil,normalize=True,closed='left')

df=pd.DataFrame()

df['fecha']=fechas
df['mes']=pd.DatetimeIndex(fechas).month
df['ano']=pd.DatetimeIndex(fechas).year
df['dias']=pd.DatetimeIndex(fechas).day
df['periodo']=periodo



'''
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
'''


