
"""
#########################################                            ########################################
########################################   ANALISIS DE DECLINACION   #########################################
########################################                            #########################################
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from datetime import date

import input

pozos = input.pozos
info_reservas = input.info_reservas
input_fecha = input.input_fecha

#############     FUNCIONES PARA DCA   #############

def remove_nan_and_zeroes_from_columns(df, variable):
    """
    This function cleans up a dataframe by removing rows in a specific
    column that are null/NaN or equal to 0. This basically removes zero
    production time periods.
    Arguments:
    df: Pandas dataframe.
    variable: String. Name of the column where we want to filter out
    NaN's or 0 values
    Output:
    filtered_df: Pandas dataframe. Dataframe with NaN's and zeroes filtered out of
    the specified column
    """
    filtered_df = df[(df[variable].notnull()) & (df[variable]>0)]

    return filtered_df

def generate_time_delta_column(df, time_column, date_first_online_column):
    """
    Create column for the time that a well has been online at each reading, with
    the first non-null month in the series listed as the start of production
    Arguments:
    df: Pandas dataframe
    time_column: String. Name of the column that includes the specific record date
    that the data was taken at. Column type is pandas datetime
    date_first_online_column: Name of the column that includes the date that the
    well came online. Column type is pandas datetime
    Outputs:
    Pandas series containing the difference in days between the date the well
    came online and the date that the data was recorded (cumulative days online)
    """
    #df['nb_months'] = ((df.date2 - df.date1)/np.timedelta64(1, 'M'))
    #df['nb_months'] = df['nb_months'].astype(int)
    #timedelta=(df[time_column]-df[date_first_online_column]).dt.days
    timedelta=(df[time_column]-df[date_first_online_column]).dt.days
    return timedelta

def get_min_or_max_value_in_column_by_group(dataframe, group_by_column, calc_column, calc_type):

    """
    This function obtains the min or max value for a column, with a group by applied. For example,
    it could return the earliest (min) RecordDate for each API number in a dataframe
    Arguments:
    dataframe: Pandas dataframe
    group_by_column: string. Name of column that we want to apply a group by to
    calc_column: string. Name of the column that we want to get the aggregated max or min for
    calc_type: string; can be either 'min' or 'max'. Defined if we want to pull the min value
    or the max value for the aggregated column
    Outputs:
    value: Depends on the calc_column type.
    """
    value=dataframe.groupby(group_by_column)[calc_column].transform(calc_type)

    return value

def get_max_initial_production(df, number_first_months, variable_column, date_column):

    """
    This function allows you to look at the first X months of production, and selects
    the highest production month as max initial production
    Arguments:
    df: Pandas dataframe.
    number_first_months: float. Number of months from the point the well comes online
    to compare to get the max initial production rate qi (this looks at multiple months
    in case there is a production ramp-up)
    variable_column: String. Column name for the column where we're attempting to get
    the max volume from (can be either 'Gas' or 'Oil' in this script)
    date_column: String. Column name for the date that the data was taken at
    """
    #First, sort the data frame from earliest to most recent prod date
    df=df.sort_values(by=date_column)
    #Pull out the first x months of production, where number_first_months is x
    df_beginning_production=df.head(number_first_months)
    #Return the max value in the selected variable column from the newly created
    #df_beginning_production df

    return df_beginning_production[variable_column].max()

def hiperbolica(t, qi, b, di):
    """
    Hyperbolic decline curve equation
    Arguments:
    t: Float. Time since the well first came online, can be in various units
    (days, months, etc) so long as they are consistent.
    qi: Float. Initial production rate when well first came online.
    b: Float. Hyperbolic decline constant
    di: Float. Nominal decline rate at time t=0
    Output:
    Returns q, or the expected production rate at time t. Float.
    """
    return qi/((1.0+b*di*t)**(1.0/b))

def exponencial(t, qi, di):
    """
    Exponential decline curve equation
    Arguments:
    t: Float. Time since the well first came online, can be in various units
    (days, months, etc) so long as they are consistent.
    qi: Float. Initial production rate when well first came online.
    di: Float. Nominal decline rate (constant)
    Output:
    Returns q, or the expected production rate at time t. Float.
    """
    return qi*np.exp(-di*t)

def harmonica(t, qi, di):
    """
    Harmonic decline curve equation
    Arguments:
    t: Float. Time since the well first came online, can be in various units
    (days, months, etc) so long as they are consistent.
    qi: Float. Initial production rate when well first came online.
    di: Float. Nominal decline rate (constant)
    Output:
    Returns q, or the expected production rate at time t. Float.
    """
    return qi/(1+(di*t))

def plot_actual_vs_predicted_by_equations(df, x_variable, y_variables, plot_title):
    """
    This function is used to map x- and y-variables against each other
    Arguments:
    df: Pandas dataframe.
    x_variable: String. Name of the column that we want to set as the
    x-variable in the plot
    y_variables: string (single), or list of strings (multiple). Name(s)
    of the column(s) that we want to set as the y-variable in the plot
    """
    #Plot serie_campo
    df.plot(x=x_variable, y=y_variables, title=plot_title,figsize=(10,5),scalex=True, scaley=True)
    plt.show()

serie_campo=pd.DataFrame()
serie_base=pd.DataFrame()
serie_status=pd.DataFrame()
serie_resumen=pd.DataFrame()
gasto=pd.DataFrame()

#Carga data pozos
data_pozos=pozos

#Limpieza de datos y formato de fecha
data_pozos['fecha']=pd.to_datetime(data_pozos['fecha'],dayfirst=True)

#Hidrocarburos de análisis
if data_pozos.aceite_Mbd.sum() > data_pozos.gas_no_asociado_MMpcd.sum():

    hidrocarburo='aceite_Mbd'
    gas='gas_asociado_MMpcd'

else:

    hidrocarburo='gas_no_asociado_MMpcd'
    gas='gas_no_asociado_MMpcd'

condensado='condensado_Mbd'
agua='agua_Mbd'

#Remove all rows with null values in the desired time series column
data_pozos=remove_nan_and_zeroes_from_columns(data_pozos, hidrocarburo)

#Get a list of unique wells to loop through
unique_well_list=list(pd.unique(data_pozos.pozo))

#Get the earliest RecordDate for each Well
data_pozos['first_oil']= get_min_or_max_value_in_column_by_group(data_pozos, group_by_column='pozo',
                                                                calc_column='fecha', calc_type='min')

data_pozos['first_oil']=pd.to_datetime(data_pozos['first_oil'],dayfirst=True)

#Generate column for time online delta
data_pozos['days_online']=generate_time_delta_column(data_pozos, time_column='fecha',
                                                      date_first_online_column='first_oil')

#Generacion de dataframes por rangos de fechas de análisis
data_pozos_range=data_pozos[(data_pozos.fecha>='1900-01-01') & (data_pozos.fecha<=pd.to_datetime(date.today()))]


#Loop para realizar el DCA en cada pozo del campo
for pozo in unique_well_list:

    #Subset del data frame del campo por pozo
    serie_produccion=data_pozos_range[data_pozos_range.pozo == pozo]
    serie_produccion=serie_produccion.set_index('pozo')

    #Cálculo de la máxima producción inicial
    qi=get_max_initial_production(serie_produccion, 6, hidrocarburo, 'fecha')

    qi_g=get_max_initial_production(serie_produccion, 6, gas, 'fecha')
    qi_c=get_max_initial_production(serie_produccion, 6, condensado, 'fecha')

    if qi_g == 0:
        qi_g = 0.00000000000000000000000000000000000000000001

    if qi_c == 0:
        qi_c = 0.00000000000000000000000000000000000000000001

    #Resultados de Qi historica
    serie_produccion.loc[:,'Qi_hist']=qi

    #Columna de mes de producción
    serie_produccion.loc[:,'mes']=np.around((serie_produccion.days_online/30),decimals=0)

    #Calculo de declinacion porcentual
    serie_produccion['pct_cambio_Qo']=serie_produccion[hidrocarburo].pct_change(periods=1)

    #Ajuste Exponencial
    popt_exp, pcov_exp=curve_fit(exponencial, serie_produccion['mes'],
                                serie_produccion[hidrocarburo],bounds=(0, [qi,20]))


    popt_exp_g, pcov_exp_g=curve_fit(exponencial, serie_produccion['mes'],
                                 serie_produccion[gas],bounds=(0, [qi_g,30]))

    #print('Exponential Fit Curve-fitted Variables: qi='+str(popt_exp[0])+', di='+str(popt_exp[1]))

    #Ajuste Hiperbolico
    popt_hyp, pcov_hyp=curve_fit(hiperbolica, serie_produccion['mes'],
                                 serie_produccion[hidrocarburo],bounds=(0, [qi,1,20]))

    popt_hyp_g, pcov_hyp_g=curve_fit(hiperbolica, serie_produccion['mes'],
                                 serie_produccion[gas],bounds=(0, [qi_g,1,30]))

    popt_hyp_c, pcov_hyp_c=curve_fit(hiperbolica, serie_produccion['mes'],
                                 serie_produccion[condensado],bounds=(0.0, [qi_c,1,20]))

    #print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

    #Ajuste Harmonico
    popt_harm, pcov_harm=curve_fit(harmonica, serie_produccion['mes'],
                                 serie_produccion[hidrocarburo],bounds=(0, [qi,20]))

    popt_harm_g, pcov_harm_g=curve_fit(harmonica, serie_produccion['mes'],
                                 serie_produccion[gas],bounds=(0, [qi_g,30]))

    #print('Harmonic Fit Curve-fitted Variables: qi='+str(popt_harm[0])+', di='+str(popt_harm[1]))

    #Resultados de funcion Exponencial
    serie_produccion.loc[:,'exponencial']=exponencial(serie_produccion['mes'],
                          *popt_exp)

    #Resultados de funcion Hiperbolica
    serie_produccion.loc[:,'hiperbolica']=hiperbolica(serie_produccion['mes'],
                              *popt_hyp)

    #Resultados de funcion Harmonica
    serie_produccion.loc[:,'harmonica']=harmonica(serie_produccion['mes'],
                              *popt_harm)

    #Residuales de la funcion Hidrocarburo Principal
    serie_produccion.loc[:,'residual_exponencial']=(serie_produccion[hidrocarburo]-serie_produccion.exponencial)**2
    serie_produccion.loc[:,'residual_hiperbolica']=(serie_produccion[hidrocarburo]-serie_produccion.hiperbolica)**2
    serie_produccion.loc[:,'residual_harmonica']=(serie_produccion[hidrocarburo]-serie_produccion.harmonica)**2


    #Resultados de funcion Gas
    serie_produccion.loc[:,'gas_exponencial']=exponencial(serie_produccion['mes'],
                              *popt_exp_g)

    serie_produccion.loc[:,'gas_hiperbolica']=hiperbolica(serie_produccion['mes'],
                              *popt_hyp_g)

    serie_produccion.loc[:,'gas_harmonica']=harmonica(serie_produccion['mes'],
                              *popt_harm_g)

    #Residuales de la funcion Gas
    serie_produccion.loc[:,'residual_gas_exponencial']=(serie_produccion[gas]-serie_produccion.gas_exponencial)**2
    serie_produccion.loc[:,'residual_gas_hiperbolica']=(serie_produccion[gas]-serie_produccion.gas_hiperbolica)**2
    serie_produccion.loc[:,'residual_gas_harmonica']=(serie_produccion[gas]-serie_produccion.gas_harmonica)**2

    #Resultados de funcion Condensado
    serie_produccion.loc[:,'condensado']=hiperbolica(serie_produccion['mes'],
                             *popt_hyp_c)


    #Calculo del ERROR ESTANDAR para cada parametro
    perr_hyp = np.sqrt(np.diag(pcov_hyp))
    perr_harm = np.sqrt(np.diag(pcov_harm))
    perr_exp = np.sqrt(np.diag(pcov_exp))


    serie_produccion.loc[:,'Np_MMb']=(serie_produccion[hidrocarburo].cumsum())*30/1_000
    serie_produccion.loc[:,'Gp_MMMpc']=(serie_produccion[gas].cumsum())*30/1_000
    serie_produccion.loc[:,'Cp_MMb']=(serie_produccion[condensado].cumsum())*30/1_000
    serie_produccion.loc[:,'Wp_MMb']=(serie_produccion[agua].cumsum())*30/1_000

    serie_produccion.loc[:,'RGA'] = (serie_produccion[gas]*1_000) / serie_produccion[hidrocarburo]
    serie_produccion.loc[:,'corte_agua'] = serie_produccion[agua] / (serie_produccion.aceite_Mbd + serie_produccion.condensado_Mbd)
    serie_produccion.loc[:,'Mbpced'] = serie_produccion.aceite_Mbd + (serie_produccion[gas]*(1/6))
    serie_produccion.loc[:,'acum_BOE'] = serie_produccion.Mbpced.cumsum()*30/1_000
    serie_produccion.loc[:,'liquidos_Mbd'] = serie_produccion.aceite_Mbd + serie_produccion.agua_Mbd + serie_produccion.condensado_Mbd


    seleccion_status=serie_produccion[serie_produccion.fecha == serie_produccion.fecha.max()]
    seleccion_base=serie_produccion[(serie_produccion.fecha == serie_produccion.fecha.max()) & (serie_produccion.fecha >= '2020-01-01')]

    Qi=[[pozo,
         qi,
         popt_hyp[0],
         popt_hyp[1],
         popt_hyp[2],
         perr_hyp[0],
         perr_hyp[1],
         popt_harm[0],
         popt_harm[1],
         perr_harm[0],
         perr_harm[1],
         popt_exp[0],
         popt_exp[1],
         perr_exp[0],
         perr_exp[1],
         float(seleccion_status.at[pozo,hidrocarburo]),
         serie_produccion.fecha.max(),
         serie_produccion.loc[:,'mes'].max(),
         float(seleccion_status.at[pozo,'profundidad_total']),
         str(seleccion_status.at[pozo,'trayectoria']),
         seleccion_status.at[pozo,'first_oil'],
         popt_hyp_g[0],
         popt_hyp_g[1],
         popt_hyp_g[2],
         popt_hyp_c[0],
         popt_hyp_c[1],
         popt_hyp_c[2],
         str(seleccion_status.at[pozo,'estado_actual']),
         serie_produccion.residual_exponencial.sum(),
         serie_produccion.residual_hiperbolica.sum(),
         serie_produccion.residual_harmonica.sum(),
         serie_produccion.residual_gas_exponencial.sum(),
         serie_produccion.residual_gas_hiperbolica.sum(),
         serie_produccion.residual_gas_harmonica.sum(),
         serie_produccion.Np_MMb.max(),
         serie_produccion.Gp_MMMpc.max(),
         serie_produccion.Cp_MMb.max(),
         serie_produccion.Wp_MMb.max(),
         serie_produccion.acum_BOE.max(),
         serie_produccion.RGA.mean(),
         serie_produccion.corte_agua.mean()
         ]]

    resumen_pozos=[[pozo,
                     qi,
                     float(seleccion_status.at[pozo,hidrocarburo]),
                     serie_produccion.fecha.max(),
                     serie_produccion.loc[:,'mes'].max(),
                     float(seleccion_status.at[pozo,'profundidad_total']),
                     str(seleccion_status.at[pozo,'trayectoria']),
                     seleccion_status.at[pozo,'first_oil'],
                     str(seleccion_status.at[pozo,'estado_actual']),
                     float(seleccion_status.at[pozo,'dias_perforacion']),
                     serie_produccion.Np_MMb.max(),
                     serie_produccion.Gp_MMMpc.max(),
                     serie_produccion.Cp_MMb.max(),
                     serie_produccion.Wp_MMb.max(),
                     serie_produccion.acum_BOE.max(),
                     serie_produccion.RGA.mean(),
                     serie_produccion.corte_agua.mean()
                     ]]


    #Plot del Análisis de Declinación de Curvas (DCA)
    #Declare the x- and y- variables that we want to plot against each other
    y_variables=[hidrocarburo,'harmonica','hiperbolica']
    x_variable='mes'

    #Create the plot title
    plot_title=hidrocarburo+' for '+str(pozo)

    #Plot the data to visualize the equation fit
    #plot_actual_vs_predicted_by_equations(serie_produccion, x_variable, y_variables, plot_title)

    #Resultados de DCA
    serie_campo=serie_campo.append(serie_produccion,sort=False)
    gasto=gasto.append(Qi,sort=True)
    serie_status=serie_status.append(seleccion_status)
    serie_base=serie_base.append(seleccion_base)
    serie_resumen=serie_resumen.append(resumen_pozos)



gasto=gasto.rename(columns={0:'pozo',
                            1:'Qi_hist',
                            2:'Qi_hyp',
                            3:'b',
                            4:'di_hyp',
                            5:'error_Qi_hyp',
                            6:'error_di_hyp',
                            7:'Qi_harm',
                            8:'di_harm',
                            9:'error_Qi_harm',
                           10:'error_di_harm',
                           11:'Qi_exp',
                           12:'di_exp',
                           13:'error_Qi_exp',
                           14:'error_di_exp',
                           15:'hidrocarburo_principal',
                           16:'ultima_produccion',
                           17:'mes_max',
                           18:'profundidad_total',
                           19:'trayectoria',
                           20:'first_oil',
                           21:'Qi_gas',
                           22:'b_gas',
                           23:'di_gas',
                           24:'Qi_condensado',
                           25:'b_condensado',
                           26:'di_condensado',
                           27:'estado_actual',
                           28:'RSS_exponencial',
                           29:'RSS_hiperbolica',
                           30:'RSS_harmonica',
                           31:'RSS_gas_exponencial',
                           32:'RSS_gas_hiperbolica',
                           33:'RSS_gas_harmonica',
                           34:'Np',
                           35:'Gp',
                           36:'Cp',
                           37:'Wp',
                           38:'acum_BOE',
                           39:'RGA',
                           40:'corte_agua'
                           })

gasto=gasto.set_index('pozo')

serie_resumen=serie_resumen.rename(columns={0:'pozo',
                                            1:'Qi_hist',
                                            2:'ultima_produccion',
                                            3:'ultima_fecha',
                                            4:'mes_max',
                                            5:'profundidad_total',
                                            6:'trayectoria',
                                            7:'first_oil',
                                            8:'estado_actual',
                                            9:'dias_perforacion',
                                           10:'Np',
                                           11:'Gp',
                                           12:'Cp',
                                           13:'Wp',
                                           14:'acum_BOE',
                                           15:'RGA',
                                           16:'corte_agua'
                                           })

serie_resumen=serie_resumen.set_index('pozo')

estadistica=serie_campo.describe()


########### GENERACION DE OVERVIEW DEL CAMPO

Np=(serie_campo.aceite_Mbd.sum()*30)/1_000
Gp=((serie_campo.gas_asociado_MMpcd.sum()+serie_campo.gas_no_asociado_MMpcd.sum())*30)/1_000
Cp=(serie_campo.condensado_Mbd.sum()*30)/1_000
Wp=(serie_campo.agua_Mbd.sum()*30)/1_000

gas_equiv=((serie_campo[gas].sum()*30)/1_000)*(1/6)


Q_base=serie_base.aceite_Mbd.sum()
G_base=serie_base[gas].sum()
C_base=serie_base.condensado_Mbd.sum()

#base={'Qo base':Q_base, 'Qg base':G_base, 'Qc base':C_base}
lista_pozos=list(pd.unique(pozos['pozo']))
pozos_perforados=len(lista_pozos)
pozos_productores=len(unique_well_list)
pozos_secos=pozos_perforados-pozos_productores

pozos_activos=len(pd.unique(serie_base.index))
pozos_cerrados=pozos_productores-pozos_activos

exito_mecanico=pozos_productores/pozos_perforados

EUR_por_pozo=Np/pozos_productores

resumen_produccion=pd.DataFrame()
resumen_produccion['maxima_produccion_pozo_Mbd']=pozos.groupby(by='pozo')[hidrocarburo].max()
resumen_produccion['EUR_MMb']=pozos.groupby(by='pozo')[hidrocarburo].sum()*30/1_000
resumen_produccion=resumen_produccion.sort_values(by='maxima_produccion_pozo_Mbd',ascending=False)

EUR_max=resumen_produccion.EUR_MMb.max()

produccion_mensual_media=serie_campo[hidrocarburo].quantile(0.50)
produccion_mensual_max=serie_campo[hidrocarburo].max()

produccion_mensual=pd.DataFrame()
produccion_mensual['produccion_mensual_campo_Mbd']=pozos.groupby(by=['fecha'])[hidrocarburo].sum()
produccion_mensual=produccion_mensual.sort_values(by='produccion_mensual_campo_Mbd',ascending=False)

fecha_pico=produccion_mensual.max()
#display(produccion_mensual.head(1))

if float(info_reservas['PRODUCCION ACUMULADA CRUDO (MMB)'].sum()) > Np:
    Np = float(info_reservas['PRODUCCION ACUMULADA CRUDO (MMB)'].sum())

if float(info_reservas['PRODUCCION ACUMULADA GAS (MMMPC)'].sum()) > Gp:
    Gp = float(info_reservas['PRODUCCION ACUMULADA GAS (MMMPC)'].sum())

if  info_reservas['VO CRUDO 1P (MMB)'].empty == True & info_reservas['VO GAS 1P (MMMPC)'].empty == True:
    OOIP = float(0)
    FR_aceite = float(0)
    OGIP = float(0)
    FR_gas = float(0)

else:
    OOIP = float(info_reservas['VO CRUDO 1P (MMB)'].sum())
    OGIP = float(info_reservas['VO GAS 1P (MMMPC)'].sum())
    FR_aceite = float(Np/OOIP)
    FR_gas = float(Gp/OGIP)


resumen=pd.Series()
resumen=pd.Series(data=[pozos_perforados,
                           pozos_productores,
                           pozos_secos,
                           exito_mecanico*100,
                           pozos_activos,
                           pozos_cerrados,
                           EUR_por_pozo,
                           EUR_max,
                           produccion_mensual_media,
                           produccion_mensual_max,
                           str(hidrocarburo),
                           str(gas),
                           Q_base,
                           G_base,
                           C_base,
                           serie_produccion.RGA.quantile(0.50),
                           serie_produccion.corte_agua.quantile(0.50),
                           'N/A',
                           'N/A',
                           'N/A',
                           'N/A',
                           'N/A',
                           'N/A',
                           Np, Gp, Cp, Wp, gas_equiv, OOIP, FR_aceite, OGIP, FR_gas],
                     index=('Pozos perforados',
                            'Pozos productores',
                            'Pozos secos',
                            'Exito mecanico (%)',
                            'Pozos activos',
                            'Pozos cerrados',
                            'EUR por pozo (MMb)',
                            'EUR maxima (MMb)',
                            'Produccion media mensual (Mbd)',
                            'Pico de producción mensual (MMb)',
                            'Hidrocarburo principal',
                            'Hidrocarburo secundario',
                            'Produccion actual de '+str(hidrocarburo),
                            'Progduccion actual de '+str(gas),
                            'Produccion actual de '+str(condensado),
                            'RGA pc/b',
                            'Corte de agua %',
                            'Gravedad API',
                            'C1',
                            'C2',
                            'C3',
                            'C4',
                            'C5+',
                            'Np','Gp','Cp','Wp','Gas Equivalente','OOIP','FR Aceite', 'OGIP','FR Gas',
                            )
                     )
#display(resumen)


#################### SERIE MUESTRA (since predetermined date)

if input_fecha != str(''):

    fecha_muestra=pd.Timestamp(input_fecha)

    serie_muestra=pd.DataFrame()

    pozos_desde=data_pozos[(data_pozos.first_oil>=fecha_muestra) & (data_pozos.first_oil<=pd.to_datetime(date.today()))]

    for pozo in unique_well_list:

        serie_desde=pozos_desde[pozos_desde.pozo==pozo]
        serie_desde=serie_desde.set_index('pozo')

        qi_desde=get_max_initial_production(serie_desde, 6, hidrocarburo, 'fecha')
        qi_g_desde=get_max_initial_production(serie_desde, 6, gas, 'fecha')
        qi_c_desde=get_max_initial_production(serie_desde, 6, condensado, 'fecha')

        if qi_g_desde == 0:
            qi_g_desde = 0.00000000000000000000000000000000000000000001

        if qi_c_desde == 0:
            qi_c_desde = 0.00000000000000000000000000000000000000000001

        serie_desde['Qi_desde']=qi_desde
        serie_desde['mes']=(serie_desde[hidrocarburo] > 0).cumsum()

        serie_muestra=serie_muestra.append(serie_desde)
