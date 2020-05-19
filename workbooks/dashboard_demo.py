#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:47:37 2020

@author: fffte
"""

import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts

from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6

from holoviews import dim, opts

from bokeh.resources import INLINE


hv.extension('bokeh')

from dca import serie_campo, serie_status, serie_resumen, resumen, perfil, tipos

def plot_overview():

    produccion = serie_campo.groupby(by='fecha').mean()

    estado_mecanico = serie_status
    elementos_status = dict(status=pd.unique(estado_mecanico.estado_actual),numero=estado_mecanico.estado_actual.value_counts())
    elementos_trayectoria = dict(trayectoria=pd.unique(estado_mecanico.trayectoria),numero=estado_mecanico.trayectoria.value_counts())

    elementos_pozos = dict(indice=resumen.index[0:6], valores=resumen[0:6])
    elementos_volumen = dict(indice=resumen.index[13:], valores=resumen[13:])

    tabla_pozos = hv.Table(elementos_pozos,'indice','valores')
    tabla_pozos.opts(height=500,fontscale=20)

    tabla_volumen = hv.Table(elementos_volumen,'indice','valores')
    tabla_volumen.opts(height=500,fontscale=20)

    plot_prod_aceite = hv.Curve(produccion, 'fecha', 'aceite_Mbd',label='Aceite Mbd')
    plot_prod_gas = hv.Curve(produccion,'fecha','gas_asociado_MMpcd',label='Gas Asociado MMpcd')

    plot_produccion = plot_prod_aceite * plot_prod_gas

    plot_produccion.opts(width=600,
                         fontscale=1.5)

    plot_trayectoria = hv.Bars(elementos_trayectoria,'trayectoria','numero')
    plot_trayectoria.opts(stacked=True,
                          color='trayectoria',
                          cmap='Spectral',
                          invert_axes=True,
                          fontscale=1.5,
                          yaxis=None)
                          #fill_color=factor_cmap('trayectoria', palette=Spectral6, factors=elementos_trayectoria['trayectoria']))



    plot_status = hv.Bars(elementos_status,'status','numero')
    plot_status.opts(stacked=True,
                     color='status',
                     fill_color=factor_cmap('status', palette=Spectral6, factors=elementos_status['status']),
                     xrotation=90,
                     invert_axes=True,
                     fontscale=1.5,
                     xticks=None,
                     yaxis=None)

    row1 = tabla_pozos + plot_status + plot_trayectoria

    row2 = tabla_volumen + plot_produccion


    fig1 = hv.render(row1)

    hv.output(row1, backend='bokeh', fig='html', size=200)

    fig2 = hv.render(row2)

    hv.output(row2, backend='bokeh', fig='html', size=200)

    return

def plot_gasto_inicial():

    dispersion_qi = hv.Points(tipos,kdims=['first_oil','Qi_hist'])

    dispersion_qi.opts(width=600,
                       size=20,
                       color='tipo',
                       cmap='Set1',
                       fontscale=1.5,
                       legend_position='top')

    fig = hv.render(dispersion_qi)

    hv.output(dispersion_qi, backend='bokeh', fig='html', size=200)

    return

def plot_pozos_tipo():


    dims_eur_baja = dict(kdims='mes', vdims=['EUR_baja_L','EUR_baja_H'])
    env_eur_baja = hv.Area(perfil, label='EUR Baja', **dims_eur_baja)
    env_eur_baja.opts(alpha=0.3,
                      color='red',
                      fontscale=1.5)

    linea_eur_baja = hv.Curve(perfil.EUR_baja_M)
    linea_eur_baja.opts(color='red',
                        line_dash='dotted')

    dims_eur_media = dict(kdims='mes', vdims=['EUR_media_L','EUR_media_H'])
    env_eur_media= hv.Area(perfil, label='EUR Media', **dims_eur_media)
    env_eur_media.opts(alpha=0.3,
                       color='blue',
                       fontscale=1.5)

    linea_eur_media = hv.Curve(perfil.EUR_media_M)
    linea_eur_media.opts(color='blue',
                         line_dash='dotted')

    dims_eur_alta = dict(kdims='mes', vdims=['EUR_alta_L','EUR_alta_H'])
    env_eur_alta = hv.Area(perfil, label='EUR Alta', **dims_eur_alta)
    env_eur_alta.opts(alpha=0.3,
                      color='green',
                      fontscale=1.5)

    linea_eur_alta = hv.Curve(perfil.EUR_alta_M)
    linea_eur_alta.opts(color='green',
                        line_dash='dotted')

    elementos_tabla=dict(indice=resumen.index[6:13], valores=resumen[6:13])

    tabla_resumen = hv.Table(elementos_tabla,'indice','valores')
    tabla_resumen.opts(height=500,fontscale=20)

    plot_eur = env_eur_baja * linea_eur_baja * env_eur_media * linea_eur_media * env_eur_alta * linea_eur_alta
    plot_eur.opts(legend_position='top_left')

    elementos_tipos = dict(tipo=pd.unique(tipos.tipo), numero=tipos.tipo.value_counts())
    plot_tipos = hv.Bars(elementos_tipos,'tipo','numero')
    plot_tipos.opts(color='tipo',
                    cmap='Set1',
                    fontscale=1.5)
                    #fill_color=factor_cmap('tipo', palette=Spectral6, factors=elementos_tipos['tipo']))

    layout = plot_eur + plot_tipos  + tabla_resumen

    fig1 = hv.render(layout)

    hv.output(layout, backend='bokeh', fig='html', size=200)
    hv.save(layout, 'resumen_produccion.html')


    dims_baja = dict(kdims='mes', vdims=['baja_L','baja_H'])
    env_baja = hv.Area(perfil, label='Envolvente', **dims_baja)
    env_baja.opts(alpha=0.3,color='red',fontscale=1.5,title='Perfil BAJA Qoi')

    linea_baja = hv.Curve(perfil.baja_M,label='P50')
    linea_baja.opts(color='red',line_dash='dotted')

    dims_media = dict(kdims='mes', vdims=['media_L','media_H'])
    env_media= hv.Area(perfil, label='Envolvente', **dims_media)
    env_media.opts(alpha=0.3,color='blue',fontscale=1.5,title='Perfil MEDIA Qoi')

    linea_media = hv.Curve(perfil.media_M,label='P50')
    linea_media.opts(color='blue',line_dash='dotted')

    dims_alta = dict(kdims='mes', vdims=['alta_L','alta_H'])
    env_alta = hv.Area(perfil, label='Envolvente', **dims_alta)
    env_alta.opts(alpha=0.3,color='green',fontscale=1.5,title='Perfil ALTA Qoi')

    linea_alta = hv.Curve(perfil.alta_M,label='P50')
    linea_alta.opts(color='green',line_dash='dotted')

    plots_perfiles = env_baja * linea_baja + env_media * linea_media + env_alta * linea_alta


    fig2 = hv.render(plots_perfiles)

    hv.output(plots_perfiles, backend='bokeh', fig='html', size=200)

    hv.save(plots_perfiles, 'curvas_tipo.html')

    return


def plot_perforacion():

    elementos_perforacion=serie_resumen[['dias_perforacion','Qi_hist','estado_actual']]

    tabla_perforacion = hv.Table(elementos_perforacion,'pozo')
    tabla_perforacion.opts(height=500,width=400,fontscale=20)

    dist = hv.Distribution(serie_resumen.dias_perforacion,
                           label='Dias de perforacion - Función de Probabilidad')




    hist=serie_resumen.dias_perforacion.dropna()
    hist=np.histogram(hist)

    plot_hist = hv.Histogram(hist)


    #kde = univariate_kde(dist,
     #                    bin_range=(0, serie_resumen.dias_perforacion.max()),
      #                   bw_method='scott',
       #                  n_samples=1000)
    #kde

    scatter = hv.Scatter(serie_resumen,
                         kdims=['dias_perforacion','profundidad_total'],
                         label='Dias de perforacion vs Profundidad total')

    #dist = dists.redim.label(dias_perforacion='Dias de perforacion')
    scatter  = scatter.redim.label(dias_perforacion='Dias de perforacion', profundidad_total='Profundidad total')

    tiempos = tabla_perforacion + dist + scatter

    tiempos.opts(
        opts.Distribution(height=500, width=700, xaxis=True,
                          xlabel='Dias de Perforacion',
                          xlim=(0,serie_resumen.dias_perforacion.max()),
                          line_width=1.00,
                          color='grey',
                          alpha=0.5,
                          fontscale=1.5,
                          tools=['hover']),
        opts.Scatter(height=500,
                     width=700,
                     xaxis=True,
                     yaxis=True,
                     size=dim('Qi_hist')*50,
                     line_width=0.25,
                     color='estado_actual',
                     cmap='Set1',
                     fontscale=1.5,
                     legend_position='bottom_right'))
                     #fill_color=factor_cmap('estado_actual', palette=Spectral6, factors=elementos_tipos['tipo']))

    tiempos

    hv.output(tiempos, backend='bokeh', fig='html')

    hv.save(tiempos, 'curvas_tipo.html')

    #hv.save(tiempos, 'tiempos.html')

    return
