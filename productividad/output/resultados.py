from productividad.input import user_input

input_analogos = user_input.input_analogos
input_archivos = user_input.input_archivos
input_plots = user_input.input_plots
input_fecha = user_input.input_fecha

def run_results():

    if input_analogos == 'Y':

        from productividad.analisis import dca_analogos

    if input_archivos == 'Y':

        from productividad.output import generar_archivos

    if input_plots == 'Y':

        from productividad.plots import plot_tipos

        from productividad.plots import plot_analisis

        from productividad.plots import plot_tiempos

        if input_fecha != str(''):

            from productividad.plots import plot_muestra

        if input_analogos == 'Y':

            from productividad.plots import plot_analogos

    return
