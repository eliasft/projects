"""
#########################################                            ########################################
########################################   MODULO DE PRODUCTIVIDAD  #########################################
########################################                            #########################################
"""

########################################     MODULOS Y SETTINGS     #########################################

import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
from datetime import date
import dateparser
import matplotlib.pyplot as plt

import seaborn as sns

import os
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize

#%matplotlib inline

import timeit
import warnings

plt.style.use('seaborn-dark')

pd.set_option('display.max_rows', 100_000_000)
pd.set_option('display.max_columns', 100_000_000)
pd.set_option('display.width', 1_000)
pd.set_option('precision', 2)
pd.options.display.float_format = '{:,.2f}'.format

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.stretch'] = 'normal'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
#plt.rcParams['image.cmap']='viridis'

#Tahoma, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif,
#Lucida Grande, Verdana, Geneva, Lucida Grande, Arial, Helvetica, Avant Garde

#sns.set_context("paper", font_scale=2.5)
sns.set_palette('husl')
#sns.set(style="fivethirtyeight")


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


import carga_bd
import input

import analisis_dca
import pozos_tipo

import resultados
resultados.run_results()

"""
#########################################                            ########################################
########################################        EXECUTER            #########################################
########################################                            #########################################
"""
