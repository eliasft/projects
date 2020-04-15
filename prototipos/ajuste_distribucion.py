import pandas as pd
import numpy as np
from datetime import datetime
import dateparser 
import matplotlib.pyplot as plt

import seaborn as sns

import os
import scipy
import scipy.stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from datetime import datetime, timedelta,date
#%matplotlib inline

import timeit
import warnings

plt.style.use('seaborn-white')

pd.set_option('display.max_rows', 100_000_000)
pd.set_option('display.max_columns', 100_000_000)
pd.set_option('display.width', 1_000)
pd.set_option('precision', 2)
pd.options.display.float_format = '{:,.2f}'.format

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 20})

sns.set_context("paper", font_scale=2.5) 


##################    AJUSTE DE DISTRIBUCION DE PROBABILIDAD   ######################
    
class Distribution(object):

    def __init__(self,dist_names_list = []):
        self.dist_names = ['beta',
                           'expon',
                           'gamma',
                           'lognorm',
                           'norm',
                           'pearson3',
                           'triang',
                           'uniform',
                           'weibull_min', 
                           'weibull_max',
                           'alpha',             
                           'anglit',            
                           'arcsine',           
                           'argus',          
                           'beta',              
                           'betaprime',         
                           'bradford',          
                           'burr',              
                           'burr12',            
                           'cauchy',            
                           'chi',               
                           'chi2',              
                           'cosine',            
                           'crystalball',       
                           'dgamma',            
                           'dweibull',          
                           'erlang',            
                           'expon',             
                           'exponnorm',         
                           'exponweib',         
                           'exponpow',          
                           'f',                 
                           'fatiguelife',       
                           'fisk',              
                           'foldcauchy',        
                           'foldnorm',          
                           'frechet_r',         
                           'frechet_l',         
                           'genlogistic',       
                           'gennorm',           
                           'genpareto',         
                           'genexpon',          
                           'genextreme',        
                           'gausshyper',        
                           'gamma',             
                           'gengamma',          
                           'genhalflogistic',   
                           'gilbrat',           
                           'gompertz',          
                           'gumbel_r',          
                           'gumbel_l',          
                           'halfcauchy',        
                           'halflogistic',      
                           'halfnorm',          
                           'halfgennorm',       
                           'hypsecant',         
                           'invgamma',          
                           'invgauss',          
                           'invweibull',        
                           'johnsonsb',         
                           'johnsonsu',         
                           'kappa4',            
                           'kappa3',            
                           'ksone',             
                           'kstwobign',         
                           'laplace',           
                           'levy',              
                           'levy_l',
                           'levy_stable',
                           'logistic',          
                           'loggamma',          
                           'loglaplace',        
                           'lognorm',           
                           'lomax',             
                           'maxwell',           
                           'mielke',            
                           'moyal',             
                           'nakagami',          
                           'ncx2',              
                           'ncf',               
                           'nct',               
                           'norm',              
                           'norminvgauss',      
                           'pareto',            
                           'pearson3',          
                           'powerlaw',          
                           'powerlognorm',      
                           'powernorm',         
                           'rdist',             
                           'reciprocal',        
                           'rayleigh',          
                           'rice',              
                           'recipinvgauss',     
                           'semicircular',      
                           'skewnorm',          
                           't',                 
                           'trapz',             
                           'triang',            
                           'truncexpon',        
                           'truncnorm',         
                           'tukeylambda',       
                           'uniform',           
                           'vonmises',          
                           'vonmises_line',     
                           'wald',              
                           'weibull_min',      
                           'weibull_max',       
                           'wrapcauchy']
        
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False


    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)

            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name,p))
        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName,self.PValue, self.params[sel_dist]

    def Random(self, n = 1):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            #initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError('Must first run the Fit method.')

    def Plot(self,y):
        x = self.Random(n=len(y))
        fig, ax = plt.subplots(figsize=(16,8))
        plt.hist(x, alpha=0.5, label='Fitted',bins=50)
        plt.hist(y, alpha=0.5, label='Actual',bins=50)
        plt.legend(loc='upper right')
     
import dca

dca.productividad()

from dca import gasto, serie_campo    
data=gasto     
#Ajuste de distribucion
dst=Distribution()
display(dst.Fit(gasto.Qi_hist))
dst.Plot(gasto.Qi_hist)

qi_params=dst.params[dst.DistributionName]
display(dst.DistributionName)
display(dst.params[dst.DistributionName])

