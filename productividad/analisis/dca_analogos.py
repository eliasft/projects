"""
#########################################                            ########################################
########################################        DCA ANALOGOS        #########################################
########################################                            #########################################
"""


import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from datetime import date

from entrada import user_input
input_campo = user_input.input_campo
mx_bd = user_input.mx_bd
pozos = user_input.pozos

from analisis.dca_main import hidrocarburo, gas, condensado, agua


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
    df['days_online']=(df[time_column]-df[date_first_online_column]).dt.days
    return (df[time_column]-df[date_first_online_column]).dt.days

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


cuenca=pd.unique(pozos.cuenca)
filtro1=pd.notnull(cuenca)
cuenca=cuenca[filtro1]
cuenca=cuenca[0]
seleccion_cuenca=mx_bd.cuenca.str.contains(pat=cuenca,regex=True)

ubicacion=pd.unique(pozos.ubicacion)
filtro2=pd.notnull(ubicacion)
ubicacion=ubicacion[filtro2]
ubicacion=ubicacion[0]
seleccion_ubicacion=mx_bd.ubicacion.str.contains(pat=ubicacion,regex=True)

fluido=pd.unique(pozos.tipo_de_hidrocarburo)
filtro3=pd.notnull(fluido)
fluido=fluido[filtro3]
fluido=fluido[0]
seleccion_fluido=mx_bd.tipo_de_hidrocarburo.str.match(pat=fluido)

analogos=mx_bd.loc[mx_bd.campo != input_campo]
analogos=analogos.loc[seleccion_cuenca & seleccion_ubicacion & seleccion_fluido]


serie_analogos=pd.DataFrame()
#serie_base=pd.DataFrame()
#serie_status=pd.DataFrame()
Qi_analogos=pd.DataFrame()
master_analogos=pd.DataFrame()

#Entrada de campo de anális
data_analogos=analogos

#Limpieza de datos y formato de fecha
data_analogos['fecha']=pd.to_datetime(data_analogos['fecha'],dayfirst=True)

#Hidrocarburos de análisis
#if data_analogos.aceite_Mbd.sum() > data_analogos.gas_no_asociado_MMpcd.sum():

    #hidrocarburo='aceite_Mbd'
    #gas='gas_asociado_MMpcd'

#else:

    #hidrocarburo='gas_no_asociado_MMpcd'
    #gas='gas_no_asociado_MMpcd'

#condensado='condensado_Mbd'

#Remove all rows with null values in the desired time series column
data_analogos=remove_nan_and_zeroes_from_columns(data_analogos, hidrocarburo)

#Get the earliest RecordDate for each Well
data_analogos['first_oil']= get_min_or_max_value_in_column_by_group(data_analogos, group_by_column='pozo',
                                                                calc_column='fecha', calc_type='min')

data_analogos['first_oil']=pd.to_datetime(data_analogos['first_oil'],dayfirst=True)

#Generate column for time online delta
data_analogos['days_online']=generate_time_delta_column(data_analogos, time_column='fecha',
              date_first_online_column='first_oil')

#Generacion de dataframes por rangos de fechas de análisis
data_analogos_range=data_analogos[(data_analogos.fecha>='1900-01-01') & (data_analogos.fecha<=pd.to_datetime(date.today()))]

data_analogos_range=data_analogos_range.groupby(by=['campo','fecha']).mean()
data_analogos_range=data_analogos_range.dropna()
data_analogos_range=data_analogos_range.reset_index()

#Get a list of unique wells to loop through
unique_analogos=list(pd.unique(data_analogos_range.campo))


#Loop para realizar el DCA en cada pozo del campo
for campo in unique_analogos:

    #Subset del data frame del campo por pozo
    serie_produccion=data_analogos_range[data_analogos_range.campo==campo]
    serie_produccion=serie_produccion.set_index('campo')

    #serie_desde=pozos_desde[pozos_desde.pozo==pozo]
    #serie_desde=serie_desde.set_index('pozo')


    #Calculo de declinacion porcentual
    #serie_produccion['declinacion']=serie_produccion[hidrocarburo].pct_change(periods=1)

    #Cálculo de la máxima producción inicial
    qi=get_max_initial_production(serie_produccion, 6, hidrocarburo, 'fecha')
    #qi_g=get_max_initial_production(serie_produccion, 500, gas, 'fecha')
    #qi_c=get_max_initial_production(serie_produccion, 500, condensado, 'fecha')

    #if qi_g == 0:
     #   qi_g = 0.00000000000000000000000000000000000000000001

    #if qi_c == 0:
     #   qi_c = 0.00000000000000000000000000000000000000000001

    #qi_desde=get_max_initial_production(serie_desde, 500, hidrocarburo, 'fecha')
    #qi_g_desde=get_max_initial_production(serie_desde, 500, gas, 'fecha')
    #qi_c_desde=get_max_initial_production(serie_desde, 500, condensado, 'fecha')

    #if qi_g_desde == 0:
     #   qi_g_desde = 0.00000000000000000000000000000000000000000001

    #if qi_c_desde == 0:
    #    qi_c_desde = 0.00000000000000000000000000000000000000000001

    #Resultados de Qi historica
    serie_produccion.loc[:,'Qi_hist']=qi
    #serie_desde['Qi_desde']=qi_desde

    #Columna de mes de producción
    serie_produccion.loc[:,'mes']=(serie_produccion[hidrocarburo] > 0).cumsum()
    #serie_desde['mes']=(serie_desde[hidrocarburo] > 0).cumsum()

    #serie_produccion.loc[:,'produccion_mensual']=serie_produccion[hidrocarburo]*30/1000
    #serie_produccion.loc[:,'produccion_acumulada']=serie_produccion.produccion_mensual.cumsum()

    #Ajuste Exponencial
    #popt_exp, pcov_exp=curve_fit(exponencial, serie_produccion['mes'],
     #                           serie_produccion[hidrocarburo],bounds=(0, [qi,50]))


    #popt_exp_g, pcov_exp_g=curve_fit(exponencial, serie_produccion['mes'],
     #                            serie_produccion[gas],bounds=(0, [qi_g,50]))

    #print('Exponential Fit Curve-fitted Variables: qi='+str(popt_exp[0])+', di='+str(popt_exp[1]))

    #Ajuste Hiperbolico
    popt_hyp, pcov_hyp=curve_fit(hiperbolica, serie_produccion['mes'],
                                 serie_produccion[hidrocarburo],bounds=(0, [qi,1,20]))

    #popt_hyp_g, pcov_hyp_g=curve_fit(hiperbolica, serie_produccion['mes'],
     #                            serie_produccion[gas],bounds=(0, [qi_g,1,50]))

    #popt_hyp_c, pcov_hyp_c=curve_fit(hiperbolica, serie_produccion['mes'],
     #                            serie_produccion[condensado],bounds=(0.0, [qi_c,1,50]))

    #print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

    #Ajuste Harmonico
    #popt_harm, pcov_harm=curve_fit(harmonica, serie_produccion['mes'],
     #                            serie_produccion[hidrocarburo],bounds=(0, [qi,50]))

    #popt_harm_g, pcov_harm_g=curve_fit(harmonica, serie_produccion['mes'],
      #                           serie_produccion[gas],bounds=(0, [qi_g,50]))

    #print('Harmonic Fit Curve-fitted Variables: qi='+str(popt_harm[0])+', di='+str(popt_harm[1]))

    #Resultados de funcion Exponencial
    #serie_produccion.loc[:,'exponencial']=exponencial(serie_produccion['mes'],
     #                     *popt_exp)

    #Resultados de funcion Hiperbolica
    serie_produccion.loc[:,'hiperbolica']=hiperbolica(serie_produccion['mes'],
                              *popt_hyp)

    serie_produccion.loc[:,'b']=popt_hyp[1]
    serie_produccion.loc[:,'di_hyp']=popt_hyp[2]

    serie_produccion.loc[:,'residual_hiperbolica']=(serie_produccion[hidrocarburo]-serie_produccion.hiperbolica)**2

    serie_produccion.loc[:,'mes_max']=serie_produccion.mes.max()

    serie_produccion.loc[:,'produccion_acumulada']=(serie_produccion[hidrocarburo].cumsum())*30/1_000

    #Resultados de funcion Harmonica
    #serie_produccion.loc[:,'harmonica']=harmonica(serie_produccion['mes'],
     #                         *popt_harm)

    #Resultados de funcion Gas
    #serie_produccion.loc[:,'gas_hiperbolica']=hiperbolica(serie_produccion['mes'],
     #                       *popt_hyp_g)

    #serie_produccion.loc[:,'gas_harmonica']=harmonica(serie_produccion['mes'],
     #                         *popt_harm_g)

    #serie_produccion.loc[:,'gas_exponencial']=exponencial(serie_produccion['mes'],
     #                         *popt_exp_g)

    #Resultados de funcion Condensado
    #serie_produccion.loc[:,'condensado']=hiperbolica(serie_produccion['mes'],
     #                        *popt_hyp_c)

    #Error
    perr_hyp = np.sqrt(np.diag(pcov_hyp))
    #perr_harm = np.sqrt(np.diag(pcov_harm))
    #perr_exp = np.sqrt(np.diag(pcov_exp))

    #seleccion_status=serie_produccion[serie_produccion.fecha == serie_produccion.fecha.max()]
    #seleccion_base=serie_produccion[serie_produccion.fecha >= '2020-01-01']

    Qi_analogos=[[campo,
                 qi,
                 popt_hyp[0],
                 popt_hyp[1],
                 popt_hyp[2],
                 serie_produccion.at[campo,'residual_hiperbolica'],
                 serie_produccion.mes.max()]]
                 #popt_harm[0],
                 #popt_harm[1],
                 #perr_harm[0],
                 #perr_harm[1],
                 #popt_exp[0],
                 #popt_exp[1],
                 #perr_exp[0],
                 #perr_exp[1],
                 #popt_hyp_g[0],
                 #popt_hyp_g[1],
                 #popt_hyp_g[2],
                 #popt_hyp_c[0],
                 #popt_hyp_c[1],
                 #popt_hyp_c[2]]]

    #Plot del Análisis de Declinación de Curvas (DCA)
    #Declare the x- and y- variables that we want to plot against each other
    y_variables=[hidrocarburo,'hiperbolica']
    x_variable='mes'

    #Create the plot title
    plot_title=hidrocarburo+' for '+str(campo)

    #Plot the data to visualize the equation fit
    #plot_actual_vs_predicted_by_equations(serie_produccion, x_variable, y_variables, plot_title)

    #Resultados de DCA
    serie_analogos=serie_analogos.append(serie_produccion,sort=False)
    #serie_muestra=serie_muestra.append(serie_desde)
    master_analogos=master_analogos.append(Qi_analogos,sort=True)
    #serie_status=serie_status.append(seleccion_status)
    #serie_base=serie_base.append(seleccion_base)


master_analogos=master_analogos.rename(columns={0:'campo',
                                                1:'Qi_hist',
                                                2:'Qi_hyp',
                                                3:'b',
                                                4:'di_hyp',
                                                5:'residual_hiperbolica',
                                                6:'mes_max'})
                                                #7:'Qi_harm',
                                                #8:'di_harm',
                                                #9:'error_Qi_harm',
                                               #10:'error_di_harm',
                                               #11:'Qi_exp',
                                               #12:'di_exp',
                                               #13:'error_Qi_exp',
                                               #14:'error_di_exp',
                                               #15:'Qi_gas',
                                               #16:'b_gas',
                                               #17:'di_gas',
                                               #18:'Qi_condensado',
                                               #19:'b_condensado',
                                               #20:'di_condensado'})

estadistica_analogos=serie_analogos.describe()
