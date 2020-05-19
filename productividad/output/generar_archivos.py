"""
#########################################                            ########################################
########################################        GENERAR ARCHIVOS     #########################################
########################################                            #########################################
"""

from entrada.user_input import info_campo, input_analogos
from analisis.dca_main import serie_campo, serie_base, resumen, master_df
from analisis.pozos_tipo import perfil, parametros, tipos, dfx, serie_rma
from analisis.dca_analogos import serie_analogos, gasto_analogos

######################### GENERAR ARCHIVOS #####################################

info_campo.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/info_campo.csv')

serie_campo.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_campo.csv')
serie_base.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_base.csv')
resumen.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/resumen.csv')
master_df.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_gasto.csv')

tipos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_tipos.csv')
parametros.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/parametros.csv')
perfil.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/perfiles_tipo.csv')
serie_rma.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_rma.csv')
dfx.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_resumen.csv')

if input_analogos == 'Y':

      serie_analogos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/serie_analogos.csv')
      gasto_analogos.to_csv(r'/Users/fffte/Documents/GitHub/projects/output/gasto_analogos.csv')
