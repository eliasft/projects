# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:08:55 2020

@author: Alejandro Alva
"""

#Modelo Económico Mensualisado

#python imputs.py
#python funciones.py
    
#Vectores auxiliares
fecha = mpp.index
dias_de_mes = fecha.days_in_month
dias_de_mes.index = fecha
len_proy = len(mpp["Producción Base Aceite"])
fin_etapa_exploratoria = fecha[duracion_etapa_exploratoria]


#Funcion precios si existe el imput de precios asignar 1
precios = funprecios(1, fecha)

#Indexar conforme a fechas
precios.index = fecha

#Perfil de produccion
gasto_aceite = (mpp["Producción Base Aceite"] + mpp["Producción Incremental Aceite"])
gasto_aceite_MMb = (gasto_aceite*dias_de_mes/1000).fillna(0)
gasto_gas = ((mpp["Producción Base Gas"] + mpp["Producción Incremental Gas"]))
gasto_gas_MMMpcd = (gasto_gas*dias_de_mes/1000).fillna(0)


gasto_condensado = mpp["Producción Base Condensado"]
gasto_condensado_MMb =(gasto_condensado*dias_de_mes/1000).fillna(0)
gasto_equivalente_boed = gastoequiv()
gasto_equivalente_fiscal = gastoequivfiscal()

dataproduccion = pd.DataFrame({"gasto_aceiteMbd" : gasto_aceite,
                             "gasto_gasMMpcd" : gasto_gas,
                             "gasto_condensadoMbd": gasto_condensado})


#Ingresos
ingresos_crudo = (precios["crudo"])*gasto_aceite_MMb
ingresos_gas = (precios["gas"])*gasto_gas_MMMpcd
ingresos_condensado = (precios["condensado"]-costo_transporte_gas)*gasto_condensado_MMb

ingresos = ingresos_crudo + ingresos_gas + ingresos_condensado

#Inversion
pozos_exploratorios = mpp["CAPEX EXPLORACION"]
pozos_delimitadores = pd.Series(0, index = fecha) #mpp["CAPEX Pozos Delimitadores (MMUSD)"]
pozos_desarrollo = mpp["CAPEX DESARROLLO"] + mpp["CAPEX DESARROLLO OTROS"]
pozos_RMA = mpp["CAPEX DESARROLLO RMA"]
pozos_inyectores = pd.Series(0, index = fecha) #mpp["CAPEX Pozos Inyectores (MMUSD)"]
capex_infra = mpp["CAPEX INFRAESTRUCTURA"]
capex_estudios = pd.Series(0, index =  fecha) #mpp["CAPEX Exploración (Estudios/Sísmica) (MMUSD)"]

capex = pozos_exploratorios + pozos_delimitadores + pozos_desarrollo + pozos_inyectores + capex_infra + capex_estudios 

#Operación y Mantemiento
opex_fijo = mpp["OPEX FIJO"] + mpp["OPEX FIJO RME"] + mpp["CAPEX DESARROLLO RMA"]
opex_var = (mpp["OPEX Variable"])*(gasto_aceite_MMb + gasto_gas_MMMpcd/factor_de_conversion)
opex_rme = pd.Series(0, index = fecha)
abandono_pozos = mpp["OPEX ABANDONO DE POZOS"]
abandono_infra = mpp["OPEX ABANDONO DE INFRAESTRUCTURA"]

mano_de_obra = mpp["MANO DE OBRA"]
acondicionamento_transporte = costo_transporte_crudo*gasto_aceite_MMb + costo_transporte_gas*gasto_gas_MMMpcd


gastoabandono = pd.Series(calculoabandono(tipo_gasto_abandono,abandono_pozos,abandono_infra))
opex = opex_fijo + opex_var + gastoabandono + mano_de_obra
administracion = mpp["ADMINISTACION"]

#Derecho e impuestos petroleros
bono = bonof(bono_a_la_firma, fecha)
cuota_cont_exp = cuota_cont_explf(fecha)
impuesto_actividadesEE = impuesto_actividadesEEf(fecha, area_exp = area_exp, area_ext = area_ext)
reg_crudo = reg_crudof(ingresos_crudo, fecha)
reg_gas = reg_gasf(ingresos_gas, fecha)
reg_conden = reg_condenf(ingresos_condensado, fecha)
reg_adic = reg_adicf(fecha)
contracpc = cpcf() 

impuestospetro = derechos_impuestos_petrof(regimen_fiscal)

#Impuestos Corporativos
depreciacion = depreciacion_anual()
ebitda = ingresos - opex
ebit = ebitda - depreciacion
utilidad_bruta = ebit - impuestospetro
pagoisr= pagoisrf(utilidad_bruta)

#Flujos
feantes = ingresos - capex - opex
fel = ingresos - capex - opex - impuestospetro - pagoisr

#Flujos Descontados
descuento = descuentof()
feantes_descontado = feantes*descuento
fel_descontado = fel*descuento

#Indicadores Economicos

parametros = {"Valor presente neto (VPN)": 0,
               "Valor presente de la inversión (VPI)":0,
               "Valor presente de los costos (VPC)":0, 
               "Eficiencia de la inversion (VPN/VPI)":0,
               "Relación beneficio costo (VPN/VPI + VPC)":0,
               "Tasa Interno de Retorno (TIR)": 0, 
               "Participación en el VPN del proyecto":0,
               "Participación en los FE del proyecto":0}

stakeholders = {"Proyecto" : parametros, "Estado" : parametros, "CEE": parametros}

indicadores_economicos = pd.DataFrame(stakeholders)

#VPN
indicadores_economicos.iloc[0,0] = pd.Series.sum(feantes_descontado)
indicadores_economicos.iloc[0,1] = pd.Series.sum(impuestospetro*descuento + pagoisr*descuento)
indicadores_economicos.iloc[0,2] = pd.Series.sum(fel_descontado)
#Valor presente de la inversión
indicadores_economicos.iloc[1,0] = pd.Series.sum(capex*descuento)
indicadores_economicos.iloc[1,1] = 0
indicadores_economicos.iloc[1,2] = pd.Series.sum(capex*descuento)
#Valor presente costos
indicadores_economicos.iloc[2,0] = pd.Series.sum(opex*descuento)
indicadores_economicos.iloc[2,1] = 0
indicadores_economicos.iloc[2,2] = pd.Series.sum(opex*descuento)
#Eficiencia de inversión
indicadores_economicos.iloc[3,0] = indicadores_economicos.iloc[0,0]/indicadores_economicos.iloc[1,0]
indicadores_economicos.iloc[3,1] = 0
indicadores_economicos.iloc[3,2] = indicadores_economicos.iloc[0,2]/indicadores_economicos.iloc[1,2]
#Relación beneficio costo
indicadores_economicos.iloc[4,0] = indicadores_economicos.iloc[0,0]/(indicadores_economicos.iloc[1,0] + indicadores_economicos.iloc[2,0])
indicadores_economicos.iloc[4,1] = 0
indicadores_economicos.iloc[4,2] = indicadores_economicos.iloc[0,2]/(indicadores_economicos.iloc[1,0] + indicadores_economicos.iloc[2,2])
#Tasa interna de retorno
indicadores_economicos.iloc[5,0] = (1 + np.irr(feantes))**12-1
indicadores_economicos.iloc[5,1] = 0
indicadores_economicos.iloc[5,2] = (1 + np.irr(fel))**12-1
#Participación VPN
indicadores_economicos.iloc[6,0] = indicadores_economicos.iloc[0,0]/indicadores_economicos.iloc[0,0]
indicadores_economicos.iloc[6,1] = indicadores_economicos.iloc[0,1]/indicadores_economicos.iloc[0,0] 
indicadores_economicos.iloc[6,2] = indicadores_economicos.iloc[0,2]/indicadores_economicos.iloc[0,0]
#Participación FE
indicadores_economicos.iloc[7,0] = pd.Series.sum(feantes)/pd.Series.sum(feantes)
indicadores_economicos.iloc[7,1] = pd.Series.sum((impuestospetro + pagoisr))/pd.Series.sum(feantes)
indicadores_economicos.iloc[7,2] = pd.Series.sum(fel)/pd.Series.sum(feantes)

#Medidasporbarril


valoresbarril = {"valor" : {"CAPEX (USD/bpce)": pd.Series.sum(capex)/pd.Series.sum(gasto_equivalente_boed), 
                            "Opex (USD/bpce)":pd.Series.sum(opex)/pd.Series.sum(gasto_equivalente_boed), 
                            "Impuestos (USD/bpce)":pd.Series.sum(impuestospetro + pagoisr)/pd.Series.sum(gasto_equivalente_boed), 
                            "Ganancia USD/bpce":pd.Series.sum(fel)/pd.Series.sum(gasto_equivalente_boed)}}

porbarril = pd.DataFrame(valoresbarril)

display(indicadores_economicos)
