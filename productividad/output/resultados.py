from entrada import user_input

input_archivos = user_input.input_archivos
input_plots = user_input.input_plots


def run_results():

    if input_archivos == 'Y':

        from output import generar_archivos

    if input_plots == 'Y':

        from plots import plot_tipos

        from plots import plot_analisis

        from plots import plot_muestra

        from plots import plot_analogos

        from plots import plot_tiempos

        '''
        if input_fecha != str(''):

            from plots import plot_muestra
        '''

        '''
        if input_analogos == 'Y':

            from plots import plot_analogos
        '''

    return
