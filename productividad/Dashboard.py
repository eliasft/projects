# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:01:50 2020

@author: Alejandro Alva
"""

import pandas as pd
import numpy as np

from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import DaysTicker, Legend, ColumnDataSource, LinearColorMapper, DatetimeTicker, Range1d, HoverTool, DataTable, DateFormatter, TableColumn, Band
from bokeh.layouts import gridplot
from bokeh.transform import dodge
import datetime as dt

#Datos
source = ColumnDataSource(datasetvisual)
sourceprod = ColumnDataSource(dataproduccion)
sourceind = ColumnDataSource(indicadores_csiee)

#Visualisaciones Gráficas

#Gráfica Flujos
hover = HoverTool(tooltips = [
        ("Fecha", "@fecha{%F}")],
        formatters = {"@fecha": "datetime"} )

TOOLS = [hover,'save,pan,box_zoom,reset,wheel_zoom']
plotcontratista = figure(title="Flujo de Pagos Contratista", x_axis_type='datetime', tools = TOOLS, plot_width=400, plot_height=400)
plotcontratista.y_range =Range1d(-3,5, bounds= (-4,6))
plotcontratista.x_range = Range1d(fechadeinicio,fechadetermino, bounds = (fechadeinicio,fechadetermino))
plotcontratista.vbar(x = dodge("fecha", 0, range = plotcontratista.x_range), top = "ingreso_contratista", width= dt.timedelta(days = 5), legend_label="Ingreso por Contraprestación", color="DarkBlue", source = source)
plotcontratista.vbar(x = dodge("fecha", 0, range = plotcontratista.x_range), top = 0, bottom = "opex_csiee", width = dt.timedelta(days= 5), legend_label ="Gasto en Operación", color="DarkOrange", source = source)
plotcontratista.vbar(x = dodge("fecha", 1000*60*60*24*5, range = plotcontratista.x_range), top = 0, bottom = "capex_csiee", width = dt.timedelta(days = 5), legend_label ="Gasto en Inversión", color = 'DarkRed', source = source)
plotcontratista.legend.location = "top_right"
plotcontratista.yaxis.axis_label = 'MMUSD'

#Gráfica Flujos Pemex
plotpemex = figure(title="Flujo de PEMEX", x_axis_type='datetime', tools = TOOLS,  plot_width=400, plot_height=400)
plotpemex.y_range =plotcontratista.y_range
plotpemex.x_range = plotcontratista.x_range
plotpemex.vbar(x = dodge("fecha", 0, range = plotpemex.x_range), top = "ingresos_csiee", width= dt.timedelta(days = 7), legend_label="Ingreso Por Proyecto", color="DarkGreen", source = source)
plotpemex.vbar(x = dodge("fecha", -1000*60*60*24*7, range = plotpemex.x_range), top = 0, bottom = "costos_irreductibles", width = dt.timedelta(days= 7), legend_label ="Costos Irreductibles", color="Brown", source = source)
plotpemex.vbar(x = dodge("fecha", 0, range = plotpemex.x_range), top = 0, bottom = "pago_contratista", width = dt.timedelta(days = 7), legend_label ="Pago Contratista", color = 'DarkBlue', source = source)
plotpemex.vbar(x = dodge("fecha", 1000*60*60*24*7, range = plotpemex.x_range), top = 0, bottom = "impuestos_csiee", width = dt.timedelta(days = 7), legend_label ="Impuestos", color = 'DarkOliveGreen', source = source)
plotpemex.legend.location = "top_right"
plotpemex.yaxis.axis_label = 'MMUSD'

#Gráfica FEL
plotfel = figure(title="Flujo de Efectivo Libre", x_axis_type='datetime', tools = TOOLS,  plot_width=400, plot_height=400)
plotfel.line(x = "fecha", y = "felcontratista", color = "DarkBlue", legend_label = "FEL Contratista", source = source, alpha = 0.8, line_width = 1)
plotfel.line(x = "fecha", y = "felpemex", color = "DarkGreen", legend_label ="FEL PEMEX", source = source, alpha = 0.8, line_width = 1)
plotfel.y_range =plotcontratista.y_range
plotfel.x_range = plotcontratista.x_range
plotfel.legend.location = "top_right"
plotfel.yaxis.axis_label = 'MMUSD'

#Tabla Indicadores
columns = [TableColumn(field = "Indicador", title = "Indicador"),TableColumn(field="PEMEX", title="PEMEX"), TableColumn(field="CSIEE", title="CSIEE")]
                    
tablaindicadores = DataTable(source=sourceind, columns = columns, index_position = None,
                             width=400, height=280)

#Produccion Aceite
plotcrudo = figure(title = "Perfil de producción de Aceite", x_axis_type='datetime', tools = TOOLS,  plot_width=400, plot_height=400)
plotcrudo.line(x = "fecha", y = "gasto_aceiteMbd", source = sourceprod )
plotcrudo.x_range = plotcontratista.x_range

band = Band(base='fecha', upper='gasto_aceiteMbd', source=sourceprod, level='underlay',
            fill_alpha=0.2, fill_color='#55FF88')

plotcrudo.add_layout(band)

#Produccion Gas
plotgas = figure(title = "Perfil de producción de Gas", x_axis_type='datetime', tools = TOOLS,  plot_width=400, plot_height=400)
plotgas.line(x = "fecha", y = "gasto_gasMMpcd", source = sourceprod )
plotgas.x_range = plotcontratista.x_range

band = Band(base='fecha', upper='gasto_gasMMpcd', source=sourceprod, level='underlay',
            fill_alpha=0.2, fill_color='#55FF88')

plotgas.add_layout(band)



layout = gridplot([[tablaindicadores,plotcrudo,plotgas],[plotpemex,plotcontratista,plotfel]], toolbar_options = dict(logo = None)) #Lists Of Rows Layout
show(layout)  #Se abré en navegador

output_file("Dashboard-Los-Soldados.html", title="Dashboard LoS Soldados")
