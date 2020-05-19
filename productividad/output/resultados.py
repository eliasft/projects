from entrada import user_input

input_analogos = user_input.input_analogos
input_archivos = user_input.input_archivos
input_plots = user_input.input_plots
input_fecha = user_input.input_fecha

def run_results():

    if input_analogos == 'Y':

        from analisis import dca_analogos

    if input_archivos == 'Y':

        from output import generar_archivos

    if input_plots == 'Y':

        from plots import plot_tipos

        from plots import plot_analisis

        from plots import plot_tiempos

        if input_fecha != str(''):

            from plots import plot_muestra

        if input_analogos == 'Y':

            from plots import plot_analogos

    return
