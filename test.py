from datetime import datetime
from tkinter.tix import AUTO
from dash import Dash, html, dcc,Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import os
import ML
from statsmodels.tsa.seasonal import seasonal_decompose

app = Dash(__name__,external_stylesheets=[dbc.themes.LUX])

app.layout = html.Div(id='page-content',children=[
    html.Br(),
    dbc.Row([
            dbc.Col(html.H1(children="Bolsa de valores"), width=8),
            dbc.Col(width=12),
        ], justify='left'),
    ###########################################
    html.Div([
        html.Div(id=f'test-content'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.H3('Predicción de precios'), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=f'Historico de información de 2010 a 2021'
                ), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                    dcc.Dropdown(
                        id=f'test-dropdown',
                        options=[
                            {'label': 'Actual', 'value': 'Actual'},
                            {'label': 'Predicted', 'value': 'Predicted'}
                        ],
                        value=['Actual', 'Predicted'],
                        multi=True,
                    ), width=4
            ),
            dbc.Col(
                    dcc.Dropdown(
                        id=f'second-dropdown',
                        options=[
                            {'label': 'APPLE', 'value': 'AAPL'},
                            {'label': 'AMAZON', 'value': 'AMZN'},
                            {'label': 'GOOGLE', 'value': 'GOOG'}
                        ],
                        multi=False,
                    ),width=5
            ),
            html.Br(),
            dbc.Col(
                    dcc.Dropdown(
                        id=f'third-dropdown',
                        options=[
                            {'label': 'Polinomial', 'value': 'POLY'},
                            {'label': 'SVR', 'value': 'SVR'},
                            {'label': 'Random Forest', 'value': 'RF'}
                        ],
                        multi=False,
                    ),width=5
            ),
            dbc.Col(width=8),
        ], justify='center'),
        dcc.Graph(id=f'test-graph'),
        dcc.RangeSlider(
            2010, 2022, 1, value=[2010, 2010], id='year-slider',marks={str(year): str(year) for year in range(2010,2022)}),
        html.Br(),
        html.Br(),
    ])


    #################################################
])

def LeerCSV(nombre):
    #Funcion para leer CSV resultado de la funcion obtenersp500
    dataframe = pd.read_csv(nombre,sep=",")
    #Convertimoos como indice las fechas para tratamiento de serie temporal
    dataframe["Date"]=pd.to_datetime(dataframe['Date'])
    dataframe.set_index("Date",inplace = True)
    return dataframe

@app.callback(
    Output('test-graph', 'figure'),
    [Input('test-dropdown', 'value'),Input('second-dropdown', 'value'),Input('year-slider', 'value'),Input('third-dropdown', 'value')])
    
def plot_load_curve(value,tiker,year,modelo):
    fig = go.Figure()
   #Orden de valores: valores "prediccion-actual"
   #año 2010 a 2020 dependiendo de year slider
   #modelo de ML seleccionado
   #No mover rutas de csv
    if 'Actual' in value and tiker!=None and year[0]!=None and year[1]!=None and modelo!=None:
        #data = LeerCSV(os.getcwd()+"/Downloads/MODULO5/SESION4/"+tiker+".csv")
        data = LeerCSV(os.getcwd()+"/"+tiker+".csv")
        YTEMP=data[(data.index>=str(year[0])) & (data.index<=str(year[1]+1))]
        #data=LeerCSV("/Users/luis/Documents/BEDU/MODULO5/SESION4/"+tiker+".csv")
        fig.add_trace(go.Scatter(
            x=data.index[(data.index>=str(year[0])) & (data.index<=str(year[1]+1))],
            #x=data.loc[str(year[0])+"-01-01":str(year[1])+"-12-31"],
            y=YTEMP['Adj Close'],
            name='Precio Real',
            line=dict(color='gray', width=2)))
    
    if 'Predicted' in value and tiker!=None and year[0]!=None and year[1]!=None and modelo!=None:
        x=data.index[(data.index>=str(year[0])) & (data.index<=str(year[1]+1))]
        y=data[(data.index>=str(year[0])) & (data.index<=str(year[1]+1))]
        fig.add_trace(go.Scatter(
            x=data.index[(data.index>=str(year[0])) & (data.index<=str(year[1]+1))],
            y=ML.MLpolynomial(y,modelo,2),
            name = 'Predicción de modelo',
            line=dict(color='red', width=2, dash='dash')))

    return fig.update_layout(
        title="Actual vs. Predicción",
        xaxis_title="Periodo",
        yaxis_title="Valor cierre",
        template='plotly_white'
    )

if __name__ == '__main__':
    app.run_server(debug=True)