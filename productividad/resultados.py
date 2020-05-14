import input

input_analogos = input.input_analogos
input_archivos = input.input_archivos
input_plots = input.input_plots
input_fecha = input.input_fecha

def run_results():

    if input_analogos == 'Y':

        import dca_analogos

    if input_archivos == 'Y':

        import generar_archivos

    if input_plots == 'Y':

        import plot_tipos

        import plot_analisis

        import plot_tiempos

        if input_fecha != str(''):

            import plot_muestra

        if input_analogos == 'Y':

            import plot_analogos

    return
