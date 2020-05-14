"""
#########################################                            ########################################
########################################           INPUT CAMPO       #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
import os

import carga_bd

mx_bd = carga_bd.mx_bd
mx_reservas = carga_bd.mx_reservas
mx_tiempos = carga_bd.mx_tiempos
mx_campos = carga_bd.mx_campos

########################################     VARIABLES GLOBALES     #########################################

alta=0.80
media=0.50
baja=0.20

####################        SECCION DE INPUTS          #####################

#INPUT DE CAMPO
input_campo = str(input("Nombre de campo/contrato/asignacion: "))

#INPUT DE ANALOGOS

input_analogos=input("Analisis DCA Analogos (Y/''): ")
input_analogos=str(input_analogos)

if input_analogos == str(''):
    input_analogos='N'

#INPUT DE RANGO DE MUESTRA

input_fecha=input("Tomar muestra desde fecha (yyyy-mm-dd): ")

#INPUT DE ARCHIVOS

input_archivos=input("Generar archivos (Y/''): ")
input_archivos=str(input_archivos)

if input_archivos == str(''):
    input_archivos='N'

#INPUT DE PLOTS

input_plots=input("Generar Plots (Y/''): ")
input_plots=str(input_plots)

if input_plots == str(''):
    input_plots='N'


####################        SUBSET DE LA BASE CAMPO ANALISIS         #####################

pozos=pd.DataFrame()

seleccion_pozo=mx_bd.pozo.str.contains(pat=input_campo,regex=True)
seleccion_campo=mx_bd.campo.str.match(pat=input_campo, na=False)
seleccion_contrato=mx_bd.contrato.str.contains(pat=input_campo,regex=True, na=False)

pozos=mx_bd.loc[seleccion_campo | seleccion_pozo | seleccion_contrato]
lista_pozos=list(pd.unique(pozos.pozo))

####################       INFORMACION COMPLEMENTARIA    #####################

pozos=pozos.merge(mx_tiempos[['pozo','tiempo_perforacion','dias_perforacion']], how='left',on='pozo')

seleccion_reservas=mx_reservas.NOMBRE.str.match(pat=input_campo)
info_reservas=mx_reservas.loc[seleccion_reservas]

seleccion_campo=mx_campos.NOMBRE.str.match(pat=input_campo)
info_campo=mx_campos[['NOMBRE',
                      'ZONA',
                      'ESTADO',
                      'AREA',
                      'ANO DE DESCUBRIMIENTO',
                      'ASIGNACION',
                      'OPERADOR',
                      'VIGENCIA',
                      'ESTATUS']].loc[seleccion_campo]

####################        VALIDACION DATOS DE POZOS        #####################

display('Número de pozos en ' +str(input_campo)+': '+str(len(lista_pozos)))

if len(lista_pozos) == 0:
    raise SystemExit("No existen pozos relacionados con el activo de análisis")


len_perfil=20*12

len_proy=0
duracion=20
len_proy=duracion*12

reservas_aceite=float(info_reservas['CRUDO 2P (MMB)'].sum())
reservas_gas=float(info_reservas['GAS NATURAL 2P (MMBPCE)'].sum())
