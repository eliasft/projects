import dca

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fitter

dca.productividad()

from dca import gasto

data = gasto[gasto.mes_max > 50]

f_Qi = fitter.Fitter(data.Qi_hist, timeout = 120)
f_Qi.fit()
best_Qi=f_Qi.get_best()
Qi_params = list(best_Qi.values())[0]

f_di = fitter.Fitter(data.di_hyp, timeout = 120)
f_di.fit()
best_di=f_di.get_best()
di_params = list(best_di.values())[0]

f_b = fitter.Fitter(data.b, timeout = 120)
f_b.fit()
best_b=f_b.get_best()
b_params = list(best_b.values())[0]


#print(f_Qi.summary())
print(best_Qi)

#print(f_b.summary())
print(best_b)

#print(f_di.summary())
print(best_di)


from scipy import stats


Qi_dist_name=list(best_Qi.items())
Qi_dist_name=Qi_dist_name[0][0]
#initiate the scipy distribution
Qi_dist = getattr(scipy.stats, Qi_dist_name)

di_dist_name=list(best_Qi.items())
di_dist_name=di_dist_name[0][0]
#initiate the scipy distribution
di_dist = getattr(scipy.stats, di_dist_name)

b_dist_name=list(best_b.items())
b_dist_name=b_dist_name[0][0]
#initiate the scipy distribution
b_dist = getattr(scipy.stats, b_dist_name)

#display(dist.rvs(*Qi_params[:-2], loc=Qi_params[-2], scale=Qi_params[-1]))#, size=n)
#display(dist.rvs(*Qi_params))


def hiperbolica1(len_t):
    
    df=pd.DataFrame()
    d_Qi=Qi_dist.rvs(*Qi_params)
    d_b=b_dist.rvs(*b_params)
    d_di=di_dist.rvs(*di_params)
    
    for t in range(0,len_t):
    
        Qo = (d_Qi)/((1.0+d_b*d_di*t)**(1.0/d_b))
  
        Q={'mes':[t],'Qo':[Qo]}
        Q=pd.DataFrame(Q)
        df=df.append(Q)
    
    df=df.set_index('mes')

    return df

produccion_campo = pd.DataFrame(index=range(0,200))

for i in range(6):
    
                 output = hiperbolica1(200)
                 produccion_campo['Pozo_'+str(i+1)]=output


produccion_campo.plot()
    

def hiperbolica2(len_t):
    
    df=pd.DataFrame()
    d_Qi=Qi_dist.rvs(*Qi_params[:-2], loc=Qi_params[-2], scale=Qi_params[-1])
    d_b=b_dist.rvs(*b_params[:-2], loc=b_params[-2], scale=b_params[-1])
    d_di=di_dist.rvs(*di_params[:-2], loc=di_params[-2], scale=di_params[-1])


    for t in range(0,len_t):
    
        Qo = (d_Qi)/((1.0+d_b*d_di*t)**(1.0/d_b))
  
        Q={'mes':[t],'Qo':[Qo]}
        Q=pd.DataFrame(Q)
        df=df.append(Q)
    
    df=df.set_index('mes')

    return df

produccion_campo = pd.DataFrame(index=range(0,200))

for i in range(6):
    
                 output = hiperbolica2(200)
                 produccion_campo['Pozo_'+str(i+1)]=output


produccion_campo.plot()

import ajuste_distribucion

from ajuste_distribucion import *

qi_params=dst.params[dst.DistributionName]

def hiperbolica3(len_t):
    
    df=pd.DataFrame()
    d_Qi=float(scipy.stats.bradford.rvs(*qi_params))
    d_b=float(scipy.stats.halfgennorm.rvs(*b_params))
    d_di=float(scipy.stats.nakagami.rvs(*di_params))

    for t in range(0,len_t):
    
        Qo = (d_Qi)/((1.0+d_b*d_di*t)**(1.0/d_b))
  
        Q={'mes':[t],'Qo':[Qo]}
        Q=pd.DataFrame(Q)
        df=df.append(Q)
    
    df=df.set_index('mes')

    return df

produccion_campo = pd.DataFrame(index=range(0,200))

for i in range(6):
    
                 output = hiperbolica2(200)
                 produccion_campo['Pozo_'+str(i+1)]=output


produccion_campo.plot()


def hiperbolica(len_t):
    
    df=pd.DataFrame()
    d_Qi=float(scipy.stats.dgamma.rvs(*Qi_params))
    d_b=float(scipy.stats.halfgennorm.rvs(*b_params))
    d_di=float(scipy.stats.nakagami.rvs(*di_params))
    
    for t in range(0,len_t):
    
        Qo = (d_Qi)/((1.0+d_b*d_di*t)**(1.0/d_b))
  
        Q={'mes':[t],'Qo':[Qo]}
        Q=pd.DataFrame(Q)
        df=df.append(Q)
    
    df=df.set_index('mes')

    return df

df=hiperbolica(200)
df.plot()

produccion_campo = pd.DataFrame(index=range(0,200))

for i in range(6):
    
                 output = hiperbolica(200)
                 produccion_campo['Pozo_'+str(i+1)]=output


produccion_campo.plot()

produccion_campo.Pozo_1.plot.hist()

produccion_campo.Pozo_2.plot.hist()

produccion_campo.Pozo_3.plot.hist()

produccion_campo.Pozo_4.plot.hist()

produccion_campo.Pozo_5.plot.hist()

from dca import serie_campo

serie_campo.aceite_Mbd[serie_campo.index=='SIHIL-18'].plot.hist()

rint(stats.dgamma.__doc__)
print(stats.halfgennorm.__doc__)
print(stats.nakagami.__doc__)

display(Qi_params)
display(Qi_params[-2])


display(b_params)
display(b_params[:-2])

display(di_params)
display(di_params[:-2])

df=pd.DataFrame()
test=pd.DataFrame()
x=0

for x in range(0,500):
    
    valor1=float(Qi_dist.rvs(*Qi_params))
    valor2=float(Qi_dist.rvs(*Qi_params[:-2], loc=Qi_params[-2], scale=Qi_params[-1]))
    valor3=float(scipy.stats.dgamma.rvs(*Qi_params))
    
    if valor1<0:
        valor1=0
    if valor2<0:
        valor2=0
    if valor3<0:
        valor3=0
    mes=x
    test=[[mes,valor1,valor2,valor3]]
    
    df=df.append(test)

df=df.set_index(0)

Qi_dist.rvs(*Qi_params)        
Qi_dist.rvs(*Qi_params[:-2], loc=Qi_params[-2], scale=Qi_params[-1])
float(scipy.stats.dgamma.rvs(*Qi_params))
#distribucion=float(scipy.stats.rv_continuous(name=nombre).rvs(*b_params))

display(float(scipy.stats.bradford.rvs(*qi_params)))