"""
#########################################                            ########################################
########################################        GENERAR ARCHIVOS     #########################################
########################################                            #########################################
"""

from input import info_campo, input_analogos
from analisis_dca import serie_campo, serie_base, resumen, gasto
from pozos_tipo import perfil, parametros, tipos, dfx, serie_rma
from dca_analogos import serie_analogos, gasto_analogos

######################### GENERAR ARCHIVOS #####################################

info_campo.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/info_campo.csv')

serie_campo.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_campo.csv')
serie_base.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_base.csv')
resumen.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/resumen.csv')
gasto.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_gasto.csv')

tipos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_tipos.csv')
parametros.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/parametros.csv')
perfil.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/perfiles_tipo.csv')
serie_rma.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_rma.csv')
dfx.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_resumen.csv')

if input_analogos == 'Y':

      serie_analogos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/serie_analogos.csv')
      gasto_analogos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/output/gasto_analogos.csv')
