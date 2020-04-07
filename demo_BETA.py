


def demo():
    

    tic=timeit.default_timer()

    productividad()
    perforacion()
    economico(regimen_fiscal ,regalia_adicional, region_fiscal)

    toc=timeit.default_timer()
    tac= toc - tic

    display('Tiempo de procesamiento: ' +str(tac)+' segundos')
    
    return

