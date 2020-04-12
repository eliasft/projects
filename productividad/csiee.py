# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:49:53 2020

@author: Alejandro Alva
"""
#inputs

#Tipo CSSIE especificar si es "fed", "Tarifa" 
tipo = "fed" #input("Tipo de CSIEE: ") #"fed
tipo = str(tipo)

if tipo not in ["fed"]:
    raise SystemExit("Párametro Inválido")
    
if tipo == "fed":
    tasafed = input("Porcentaje FED Decimales: ") #En decimales el porcentaje de regalía adicional para los contratos de licencia
    tasafed = float(tasafed)

duracion_CSIEE = 15 #años
fechadeinicio = fecha[0] - pd.DateOffset(months=1)
fechadetermino = fecha[duracion_CSIEE*12]

duracion_contrato = pd.Series(0, index = fecha)
duracion_contrato[fecha < fechadetermino] = 1



#funciones CSIEEs
def ingresosmin():
    costo = pd.Series(0, index = fecha)
    return(costo)

def costostransportes():
    costo = pd.Series(0, index = fecha)
    return(costo)
    
def produccionref(factor_de_conversion):
    
    baseaceite = gasto_aceite_MMb #insertar base
    basegas = gasto_gas_MMMpcd #insertar gas
    if factor_de_conversion == 0:
        baseequi = baseaceite
    else:
        baseequi = baseaceite + basegas/factor_de_conversion
    
    
    incrementalaceite = pd.Series(0, index = fecha)
    incrementalgas = pd.Series(0, index = fecha)
    
    vector = gasto_aceite_MMb - baseaceite
    vector2 = gasto_gas_MMMpcd -basegas
    incrementalaceite[vector > 0] = vector
    incrementalgas[vector > 0] = vector2
    if factor_de_conversion == 0:
        incrementalequi = incrementalaceite
    else:
        incrementalequi = incrementalaceite + incrementalgas/factor_de_conversion
    
    prodref = pd.DataFrame({"Base Aceite": baseaceite, "Base Gas": basegas, "Base Equivalente": baseequi,
                            "Incremental Aceite": incrementalaceite, "Incremental gas" : incrementalgas,
                            "Incremental Equivalente": incrementalequi })
    return(prodref)
    
def mecanismoajuste(): #Es contractual
    parametroajuste = pd.Series(0, index = fecha)
    ajustefed = pd.Series(0, index = fecha)
    
    acumprod = produccionref(factor_de_conversion)["Incremental Equivalente"].cumsum()
    acumfe = fedisponible.cumsum() 
    parametroajuste[acumprod > 0] = acumfe/acumprod
    
    #Ejemplo Campo los Soldados
    ajustefed[parametroajuste <= 28.27] = 0
    ajustefed[(parametroajuste <= 30.20) & ( parametroajuste >28.27)] = .0326
    ajustefed[(parametroajuste <= 31.35) & ( parametroajuste >30.20)] = .0507
    ajustefed[(parametroajuste <= 32.49) & ( parametroajuste >31.35)] = .0670
    ajustefed[(parametroajuste <= 33.62) & ( parametroajuste >32.49)] = .0822
    ajustefed[(parametroajuste <= 35.85) & ( parametroajuste >33.62)] = .1091
    ajustefed[ parametroajuste > 35.85] = .139
    
    return(ajustefed)
    
    
def contrapresta(tipo, fedisponible, duracion_contrato = duracion_contrato, tasafed = tasafed):
    
    tasafedperiodo = pd.Series(tasafed, index = fecha)
    pagotarifa = pd.Series(0.0, index = fecha)
    ajuste = mecanismoajuste()
    
    if tipo == "fed":        
        vectoraux = fedisponible*(tasafedperiodo - ajuste)
        pagotarifa[vectoraux]
        
        for i in pagotarifa.index:
            pagotarifa[i] = max(0.0, vectoraux[i])
            
    
    contraprestapag = pd.Series(0.0, index = fecha)
    carrypemex = pd.Series(0.0, index = fecha)
    
    carry = 0
    for i in contraprestapag.index:
        carrypemex[i] = carry
        contraprestapag[i] = min(pagotarifa[i] + carrypemex[i], fedisponible[i])
        carry = carry + max(pagotarifa[i] - contraprestapag[i],0)
    contraprestapag*duracion_contrato
    return(contraprestapag)
    
#Consolidador

#Costos Irreductibles
costosirre = (ingresosmin() + usosuperficial(ingresos,impuestospetro,bono) + acondicionamento_transporte + mano_de_obra)*duracion_contrato #Costos Irreductibles    

#Flujo Efectivo Disponible
fedisponible = (ingresos - impuestospetro - costosirre)*duracion_contrato

#Contraprestación
contrapresta = (contrapresta(tipo, fedisponible))

#Inversion y Gasto
opex_csiee = (opex_fijo + opex_var - mpp["ADMINISTACION"] + gastoabandono)*duracion_contrato
capex_csiee = capex*duracion_contrato

#Flujo de Efectivo Libre Contratista
felcontratista = (contrapresta - capex_csiee - opex_csiee)*duracion_contrato #pre ISR
felcontratistadescontado = felcontratista*descuento

#Flujo de Efectivo Libre Pemex
ingresos_csiee = ingresos*duracion_contrato
impuestospetro_csiee = impuestospetro*duracion_contrato


felpemex = ingresos_csiee - contrapresta - costosirre - impuestospetro_csiee


#Indicadores Economicos

parametros = {"Valor presente neto (VPN)": 0,
               "Valor presente de la inversión (VPI)":0,
               "Valor presente de gasto operacion (VPG)":0, 
               "Eficiencia de la inversion (VPN/VPI)":0,
               "Relación beneficio costo (VPN/VPI + VPC)":0,
               "Tasa Interno de Retorno (TIR)": 0,}

stakeholders = {"PEMEX" : parametros, "CSIEE" : parametros}

indicadores_csiee = pd.DataFrame(stakeholders)
indicadores_csiee.index.name = "Indicador"


#VPN
indicadores_csiee.iloc[0,0] = pd.Series.sum(felpemex*descuento)
indicadores_csiee.iloc[0,1] = pd.Series.sum(felcontratista*descuento)
#Valor presente de la inversión
indicadores_csiee.iloc[1,0] = 0
indicadores_csiee.iloc[1,1] = pd.Series.sum(capex*descuento)
#Valor presente del gasto de operacion
indicadores_csiee.iloc[2,0] = pd.Series.sum(contrapresta*descuento)
indicadores_csiee.iloc[2,1] = pd.Series.sum((opex_fijo+opex_var-administracion)*descuento)
#Eficiencia de inversión
indicadores_csiee.iloc[3,0] = 0
indicadores_csiee.iloc[3,1] = indicadores_csiee.iloc[0,1]/indicadores_csiee.iloc[1,1]
#Eficiencia del gasto de operación
indicadores_csiee.iloc[4,0] = indicadores_csiee.iloc[0,0]/(indicadores_csiee.iloc[2,0])
indicadores_csiee.iloc[4,1] = indicadores_csiee.iloc[0,1]/(indicadores_csiee.iloc[2,1])
#Tasa interna de retorno
indicadores_csiee.iloc[5,0] = "Infinita"
indicadores_csiee.iloc[5,1] = (1 + np.irr(felcontratista))**12-1

display(indicadores_csiee)

#Ingreso Contratista = Contraprestaación CSIEE

datasetvisual =pd.DataFrame({"ingreso_contratista" : contrapresta,
                             "capex_csiee" : capex_csiee*-1,
                             "opex_csiee": opex_csiee*-1,
                             "felcontratista": felcontratista,
                             "ingresos_csiee": ingresos_csiee, 
                             "costos_irreductibles" : costosirre*-1,
                             "impuestos_csiee" : impuestospetro_csiee*-1, 
                             "felpemex" : felpemex,
                             "pago_contratista":contrapresta*-1})

datasetvisual.index.name = "fecha"

