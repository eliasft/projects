import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#MX Benchmark Productividad

#MX Produccion historica por pozo

def multiple_load(path):
    
    #path = r"/Users/fffte/Documents/GitHub/Ainda/Proyecto Newton/02_productividad/benchmark/" 
    files = os.listdir(path)
    files_csv = [f for f in files if f[-3:] == 'csv']
    df = pd.DataFrame()
    for f in files_csv:
        data = pd.read_csv(path+f,header='infer',
                           low_memory=False)

        df = df.append(data,sort=True)
    return df

produccion=multiple_load(r"C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_produccion/")

produccion.to_csv(r"C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_produccion.csv")

mx=pd.read_csv(r'C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_produccion.csv',
              index_col=0,
              parse_dates=True,
              low_memory=False)

print(mx.head(),
        mx.shape,
        mx.columns)

print(mx.fecha.min())

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(mx.fecha,mx.aceite_Mbd)
ax.set_xlabel('date')
ax.set_xticklabels(labels=mx.fecha,rotation=75)
ax.set_ylabel('oil production (Mbd)')
plt.show()


#MX Caracteristicas campos

def load_campos(path):
    
    files = os.listdir(path)
    files_csv = [f for f in files if f[-3:] == 'csv']
    df = pd.DataFrame()
    for f in files_csv:
        data = pd.read_csv(path+f,skiprows=1,
                             names=['ID','NOMBRE', 'LOCALIZACION', 'ESTADO', 'LATITUD', 'LONGITUD', 'AREA',
                                    'ZONA', 'ANO DE DESCUBRIMIENTO', 'TIRANTE AGUA', 'ESTATUS',
                                   'ASIGNACION', 'OPERADOR', 'VIGENCIA', 'NUM BLOQUES', 'CAA ORIGINAL',
                                   'CAA ACTUAL', 'CGA ORIGINAL', 'CGA ACTUAL', 'NP (MMB)', 'GP (MMMPC)',
                                   'MECANISMO DE PRODUCCION', 'PRODUCCION RECORD (MBPD)',
                                   'PRODUCCION RECORD (MMPCD)', 'PRODUCCION ACEITE (MBPD)',
                                   'PRODUCCION GAS (MMPCD)', 'PRODUCCION AGUA (MBPD)', 'COLUMNA OP ACEITE',
                                   'PROYECTO IOR/EOR PROPUESTO', 'PROYECTO IOR/EOR SUGERIDO',
                                   'VOUMEN DE INYECCION DE GAS (MMPCD)',
                                   'VOUMEN DE INYECCION DE AGUA (MBPD)', 'TIPO TERMINACION',
                                   'POZOS PERFORADOS', 'POZOS PRODUCTORES', 'POZOS INYECTORES',
                                   'POZOS MONITORES PRODUCTORES', 'POZOS MONITORES',
                                   'POZOS CERRADOS CON POSIBILIDADES', 'POZOS CERRADOS SIN POSIBILIDADES',
                                   'POZOS TAPONADOS', 'REPARACIONES MENORES', 'REPARACIONES MAYORES',
                                   'SISTEMA ARTIFICIAL', 'DESCUBRIMIENTO'],
                             skip_blank_lines=True,
                             skipinitialspace=True,
                             na_filter=True)

        df = df.append(data,sort=True)
    return df

campos=load_campos(r'C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_campos/')
campos.NOMBRE=campos.NOMBRE.str.upper()
cols = campos.select_dtypes(include=[np.object]).columns
campos[cols] = campos[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

      
lista_campos=campos.columns
print(lista_campos)

#for x in lista_campos:
#      lista_campos=lista_campos.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
       
print(lista_campos)

campos=campos.dropna(axis=0,thresh=10)
print(campos.head(),
        campos.tail(),
        campos.shape)

campos.to_csv(r"C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_campos.csv")

#Merge BD de campos y pozos

def load_pozos(path):
    
    files = os.listdir(path)
    files_csv = [f for f in files if f[-3:] == 'csv']
    df = pd.DataFrame()
    for f in files_csv:
        data = pd.read_csv(path+f,header=1,
                             names=['GID',
                                    'POZO',
                                    'CAMPO',
                                    'ENTIDAD',
                                    'UBICACION',
                                    'CLASIFICACION',
                                    'ESTADO ACTUAL',
                                    'TIPO DE HIDROCARBURO',
                                    'ANO DE PERFORACION',
                                    'PROFUNDIDAD TOTAL',
                                    'PROFUNDIDAD VERTICAL',
                                    'TRAYECTORIA',
                                    'DISPONIBLE'],
                             index_col='POZO',
                             skip_blank_lines=True,
                             skipinitialspace=True,
                             na_filter=True)

        df = df.append(data,sort=True)
    return df

mx_pozos=load_pozos(r'C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_pozos/')

mx_pozos.to_csv(r'C:/Users/elias/Google Drive/python/csv/benchmark/mexico/mx_pozos.csv')

produccion=pd.read_csv(r'C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_produccion.csv',
                 low_memory=False,index_col='pozo')

pozos=pd.read_csv(r'C:\Users\elias\Google Drive\python\csv\benchmark\mexico\mx_pozos.csv',
                 low_memory=False,index_col='POZO')

pozos=pozos.reset_index()
pozos.rename(columns={'POZO':'pozo'},inplace=True)
print(pozos.head())

df2=pd.DataFrame.merge(produccion,pozos,how='outer',on='pozo')

df2=df2.set_index('pozo')
print(df2.head())

df2.to_csv(r"C:/Users/elias/Google Drive/python/csv/benchmark/mexico/mx_bd.csv")

print(df2.shape)

