import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px

from app import app
import transforms

df = transforms.df

col_options = [dict(label=x, value=x) for x in df.columns]
dimensions = ["x", "y", "color", "facet_col", "facet_row", "size"]

layout =html.Div(
            [
                html.H1("Exploracion de variables"),
                html.Div(
                    [
                        html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                        for d in dimensions
                    ],
                    style={"width": "25%", "float": "left"},
                ),
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
            ]
            )

app.layout = layout

@app.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
def make_figure(x, y, color, facet_col, facet_row,size):
    return px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        facet_col=facet_col,
        facet_row=facet_row,
        height=700,
    )
