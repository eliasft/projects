import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy.optimize import curve_fit




def load_db():

    global mx_bd
    global mx_reservas
    global mx_tiempos

    mx_bd=pd.read_csv(r'/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_full.csv',
                          usecols=['fecha',
                                  'pozo',
                                  'aceite_Mbd',
                                  'gas_asociado_MMpcd',
                                  'gas_no_asociado_MMpcd',
                                  'condensado_Mbd',
                                  'agua_Mbd',
                                  'estado_actual',
                                  'profundidad_total',
                                  'profundidad_vertical',
                                  'trayectoria',
                                  'ano_de_perforacion',
                                  'tipo_de_hidrocarburo',
                                  'clasificacion',
                                  'disponible',
                                  'campo',
                                  'cuenca',
                                  'entidad',
                                  'ubicacion',
                                  'contrato'],
                                  low_memory=True)

    mx_reservas=pd.read_csv(r'/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_reservas.csv',
                          index_col=0)

    mx_tiempos=pd.read_csv("/Users/fffte/ainda_drive/python/csv/benchmark/mexico/mx_tiempos.csv",
                          index_col=0,
                          parse_dates=True)

    return display('Ready to go!')


def inputs():
    
    global pozos, lista, reservas
    
    lista=['BEDEL',
        'EL TREINTA',
        'CINCO PRESIDENTES',
        'RODADOR',
        'GIRALDAS',
        'COMOAPA',
        'SUNUAPA',
        'MUSPAC',
        'CHIAPAS-COPANO',
        'ARTESA',
        'GAUCHO',
        'NISPERO',
        'RIO NUEVO',
        'SITIO GRANDE',
        'LACAMANGO',
        'JUSPI',
        'TEOTLECO',
        'BACAL',
        'NELASH',
        'TIUMUT',
        'ARROYO PRIETO',
        'LOS SOLDADOS',
        'JUJO-TECOMINOACAN',
        'PAREDON',
        'JACINTO',
        'SINI',
        'CACTUS',
        'MADREFIL',
        'CUPACHE',
        'TINTAL',
        'TUPILCO',
        'PACHE',
        'TOKAL',
        'CASTARRICAL',
        'AYOCOTE',
        'GUARICHO',
        'RABASA',
        'IRIDE',
        'PLATANAL',
        'CUNDUACAN',
        'OXIACAQUE',
        'TERRA',
        'CAPARROSO PIJIJE ESCUINTLE',
        'BRILLANTE',
        'EDEN JOLOTE',
        'SHISHITO',
        'SEN',
        'SAMARIA',
        'LUNA-PALAPA']

    pozos=mx_bd[mx_bd.campo.isin(lista)]
    reservas=mx_reservas[mx_reservas.NOMBRE.isin(lista)]
    
    return

def analyze(pozos):

    global unique_well_list
    global serie_pozos, serie_status
    global serie_base
    global hidrocarburo, gas, condensado
    global resumen_pozos, resumen_campos
    global parametros
    global declinacion

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


    serie_pozos=pd.DataFrame()
    serie_base=pd.DataFrame()
    serie_status=pd.DataFrame()
    resumen_pozos=pd.DataFrame()
    parametros=pd.DataFrame()

    #Carga data pozos
    data_pozos=pozos

    #Limpieza de datos y formato de fecha
    data_pozos['fecha']=pd.to_datetime(data_pozos['fecha'],dayfirst=True)

    #Hidrocarburos de análisis
    hidrocarburo='aceite_Mbd'
    gas='gas_asociado_MMpcd'
    condensado='condensado_Mbd'
    agua='agua_Mbd'

    #Remove all rows with null values in the desired time series column
    data_pozos=remove_nan_and_zeroes_from_columns(data_pozos, hidrocarburo)

    #Get the earliest RecordDate for each Well
    data_pozos.loc[:,'first_oil']= get_min_or_max_value_in_column_by_group(data_pozos, group_by_column='pozo',
                                                                    calc_column='fecha', calc_type='min')

    data_pozos.loc[:,'first_oil']=pd.to_datetime(data_pozos['first_oil'],dayfirst=True)

    #Generate column for time online delta
    data_pozos.loc[:,'days_online']=generate_time_delta_column(data_pozos, time_column='fecha',
                                                          date_first_online_column='first_oil')

    #Get a list of unique wells to loop through
    unique_well_list=list(pd.unique(data_pozos.pozo))

    #Loop para realizar el DCA en cada pozo del campo
    for pozo in unique_well_list:

        #Subset del data frame del campo por pozo
        serie_produccion=data_pozos[data_pozos.pozo == pozo]
        serie_produccion=serie_produccion.set_index('pozo')

        #Cálculo de la máxima producción inicial
        qi=get_max_initial_production(serie_produccion, 500, hidrocarburo, 'fecha')
        #qi_g=get_max_initial_production(serie_produccion, 500, gas, 'fecha')
        #qi_c=get_max_initial_production(serie_produccion, 500, condensado, 'fecha')
        
        #if qi_g == 0:
        #    qi_g = 0.00000000000000000000000000000000000000000001

        #if qi_c == 0:
         #   qi_c = 0.00000000000000000000000000000000000000000001

        #Resultados de Qi historica
        serie_produccion.loc[:,'Qi_hist']=qi

        #Columna de mes de producción
        serie_produccion.loc[:,'mes']=np.around((serie_produccion.days_online/30),decimals=0)

        #Calculo de declinacion porcentual
        serie_produccion['pct_cambio_Qo']=serie_produccion[hidrocarburo].pct_change(periods=12)
        
        #Ajuste Exponencial
        popt_exp, pcov_exp=curve_fit(exponencial, serie_produccion['mes'],
                                    serie_produccion[hidrocarburo],bounds=(0, [qi,20]))


        #popt_exp_g, pcov_exp_g=curve_fit(exponencial, serie_produccion['mes'],
         #                            serie_produccion[gas],bounds=(0, [qi_g,50]))

        #print('Exponential Fit Curve-fitted Variables: qi='+str(popt_exp[0])+', di='+str(popt_exp[1]))

        #Ajuste Hiperbolico
        popt_hyp, pcov_hyp=curve_fit(hiperbolica, serie_produccion['mes'],
                                     serie_produccion[hidrocarburo],bounds=(0, [qi,1,20]))

        #popt_hyp_g, pcov_hyp_g=curve_fit(hiperbolica, serie_produccion['mes'],
         #                            serie_produccion[gas],bounds=(0, [qi_g,1,50]))

        #popt_hyp_c, pcov_hyp_c=curve_fit(hiperbolica, serie_produccion['mes'],
         #                            serie_produccion[condensado],bounds=(0.0, [qi_c,1,20]))

        #print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

        #Ajuste Harmonico
        popt_harm, pcov_harm=curve_fit(harmonica, serie_produccion['mes'],
                                     serie_produccion[hidrocarburo],bounds=(0, [qi,20]))

        #popt_harm_g, pcov_harm_g=curve_fit(harmonica, serie_produccion['mes'],
         #                            serie_produccion[gas],bounds=(0, [qi_g,50]))

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
        #serie_produccion.loc[:,'gas_exponencial']=exponencial(serie_produccion['mes'],
         #                         *popt_exp_g)

        #serie_produccion.loc[:,'gas_hiperbolica']=hiperbolica(serie_produccion['mes'],
         #                         *popt_hyp_g)

        #serie_produccion.loc[:,'gas_harmonica']=harmonica(serie_produccion['mes'],
         #                         *popt_harm_g)

        #Residuales de la funcion Gas
        #serie_produccion.loc[:,'residual_gas_exponencial']=(serie_produccion[gas]-serie_produccion.gas_exponencial)**2
        #serie_produccion.loc[:,'residual_gas_hiperbolica']=(serie_produccion[gas]-serie_produccion.gas_hiperbolica)**2
        #serie_produccion.loc[:,'residual_gas_harmonica']=(serie_produccion[gas]-serie_produccion.gas_harmonica)**2

        #Resultados de funcion Condensado
        #serie_produccion.loc[:,'condensado']=hiperbolica(serie_produccion['mes'],
         #                        *popt_hyp_c)


        serie_produccion.loc[:,'Np_MMb']=(serie_produccion[hidrocarburo].cumsum())*30/1_000
        serie_produccion.loc[:,'Gp_MMMpc']=(serie_produccion[gas].cumsum())*30/1_000
        serie_produccion.loc[:,'Cp_MMb']=(serie_produccion[condensado].cumsum())*30/1_000
        serie_produccion.loc[:,'Wp_MMb']=(serie_produccion[agua].cumsum())*30/1_000

        seleccion_status=serie_produccion[serie_produccion.fecha == serie_produccion.fecha.max()]
        seleccion_base=serie_produccion[(serie_produccion.fecha == serie_produccion.fecha.max()) & (serie_produccion.fecha >= '2020-01-01')]

        summary_pozos=[[pozo,
                         seleccion_status.at[pozo,'campo'],
                         qi,
                         seleccion_status.at[pozo,hidrocarburo],
                         serie_produccion.fecha.max(),
                         serie_produccion.loc[:,'mes'].max(),
                         seleccion_status.at[pozo,'profundidad_total'],
                         seleccion_status.at[pozo,'trayectoria'],
                         seleccion_status.at[pozo,'first_oil'],
                         seleccion_status.at[pozo,'estado_actual'],
                         serie_produccion.Np_MMb.max(),
                         serie_produccion.Gp_MMMpc.max(),
                         serie_produccion.Cp_MMb.max(),
                         serie_produccion.Wp_MMb.max()]]
        
        ajuste=[[pozo,
                 seleccion_status.at[pozo,'campo'],
                 qi,
                 popt_hyp[0],
                 popt_hyp[1],
                 popt_hyp[2],
                 popt_harm[0],
                 popt_harm[1],
                 popt_exp[0],
                 popt_exp[1],
                 serie_produccion.residual_exponencial.sum(),
                 serie_produccion.residual_hiperbolica.sum(),
                 serie_produccion.residual_harmonica.sum()]]

        

        serie_pozos=serie_pozos.append(serie_produccion,sort=False)
        serie_status=serie_status.append(seleccion_status)
        serie_base=serie_base.append(seleccion_base)
        resumen_pozos=resumen_pozos.append(summary_pozos,sort=False)
        parametros=parametros.append(ajuste)


    resumen_pozos=resumen_pozos.rename(columns={0:'pozo',
                                                1:'campo',
                                                2:'Qi_hist',
                                                3:'Qo_max',
                                                4:'ultima_fecha_online',
                                                5:'mes_max',
                                                6:'profundidad_total',
                                                7:'trayectoria',
                                                8:'first_oil',
                                                9:'estado_actual',
                                               10:'dias_perforacion',
                                               11:'Np',
                                               12:'Gp',
                                               13:'Cp',
                                               14:'Wp'})

    resumen_pozos=resumen_pozos.set_index('pozo')
    
    
    parametros=parametros.rename(columns={0:'pozo',
                                          1:'campo',
                                          2:'Qi_hist',
                                          3:'Qi_hyp',
                                          4:'b',
                                          5:'di_hyp',
                                          6:'Qi_harm',
                                          7:'di_harm',
                                          8:'Qi_exp',
                                          9:'di_exp',
                                         10:'RSS_exponencial',
                                         11:'RSS_hiperbolica',
                                         12:'RSS_harmonica'})

    #parametros=parametros.set_index('pozo')

    ########### RESUMEN CAMPO
    unique_campos=list(pd.unique(serie_pozos.campo))
    
    resumen_campos=pd.DataFrame()
    
    for campo in unique_campos:
        
            serie_resumen=serie_pozos[serie_pozos.campo == campo]

            resumen_campos.loc[campo,'Qi']=serie_resumen.Qi_hist.quantile(0.50)
            resumen_campos.loc[campo,'Np']=(serie_resumen.aceite_Mbd.sum()*30)/1_000
            resumen_campos.loc[campo,'Gp']=((serie_resumen.gas_asociado_MMpcd.sum()+serie_pozos.gas_no_asociado_MMpcd.sum())*30)/1_000
            resumen_campos.loc[campo,'Cp']=(serie_resumen.condensado_Mbd.sum()*30)/1_000
            resumen_campos.loc[campo,'Wp']=(serie_resumen.agua_Mbd.sum()*30)/1_000
        
        
            resumen_campos.loc[campo,'Q_base']=serie_base.aceite_Mbd[serie_base.campo == campo].sum()
            resumen_campos.loc[campo,'G_base']=serie_base[gas][serie_base.campo == campo].sum()
            resumen_campos.loc[campo,'C_base']=serie_base.condensado_Mbd[serie_base.campo == campo].sum()
                    
            lista_pozos=list(pd.unique(pozos.pozo[pozos.campo == campo]))
            resumen_campos.loc[campo,'pozos_perforados']=len(lista_pozos)
            resumen_campos.loc[campo,'pozos_productores']=len(pd.unique(serie_resumen.index))
            resumen_campos.loc[campo,'pozos_secos']=resumen_campos.loc[campo,'pozos_perforados']- resumen_campos.loc[campo,'pozos_productores']
            
            resumen_campos.loc[campo,'pozos_activos']=len(pd.unique(serie_base.index[serie_base.campo == campo]))
            resumen_campos.loc[campo,'pozos_cerrados']= resumen_campos.loc[campo,'pozos_productores'] - resumen_campos.loc[campo,'pozos_activos']
            
            resumen_campos.loc[campo,'exito_mecanico']=(resumen_campos.loc[campo,'pozos_productores'])/(resumen_campos.loc[campo,'pozos_perforados'])
            
            resumen_campos.loc[campo,'EUR_por_pozo']= resumen_campos.loc[campo,'Np']/resumen_campos.loc[campo,'pozos_productores']
        
            #resumen_produccion=pd.DataFrame()
            #resumen_produccion['maxima_produccion_pozo_Mbd']=pozos.groupby(by='pozo')[hidrocarburo].max()
            #resumen_produccion['EUR_MMb']=pozos.groupby(by='pozo')[hidrocarburo].sum()*30/1_000
            #resumen_produccion=resumen_produccion.sort_values(by='maxima_produccion_pozo_Mbd',ascending=False)
        
            #EUR_max=resumen_produccion.EUR_MMb.max()
        
            #produccion_mensual_media=serie_pozos[hidrocarburo].quantile(0.50)
            #produccion_mensual_max=serie_pozos[hidrocarburo].max()
        
            #produccion_mensual=pd.DataFrame()
            #produccion_mensual['produccion_mensual_campo_Mbd']=pozos.groupby(by=['fecha'])[hidrocarburo].sum()
            #produccion_mensual=produccion_mensual.sort_values(by='produccion_mensual_campo_Mbd',ascending=False)
        
            #fecha_pico=produccion_mensual.max()
            #display(produccion_mensual.head(1)) 
            
            df=parametros.groupby(by='campo').mean()
            declinacion=pd.DataFrame(index=range(0,(12*15)))
              
            for indice in df.index:
              
                  parametro_qi = resumen_campos.Q_base[resumen_campos.index == indice].quantile(0.50)
                  parametro_b = df.b[df.index == campo].quantile(0.50)
                  parametro_di = df.di_hyp[df.index == campo].quantile(0.50)
                  
                  t=0
                  
                  for t in declinacion.index:
                  
                      qo=parametro_qi/((1.0+parametro_b*parametro_di*t)**(1.0/parametro_b))
                      declinacion.loc[t,indice]=qo
        
    
    declinacion.to_csv(r'/Users/fffte/Desktop/declinacion.csv')
    eur=declinacion.sum()
    eur.to_csv(r'/Users/fffte/Desktop/eur.csv')


    resumen_campos.to_csv(r'/Users/fffte/Desktop/resumen_campo.csv')
    parametros.to_csv(r'/Users/fffte/Desktop/parametros.csv')

    return

def run():
    
    load_db()
    
    inputs()
    
    analyze(pozos)
    
    return

