import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

#Función para hacer inferencia bayesiana
def bayesian_inference(age, sex, cp, trestbps, chol, fbs, exang):
    
    # Cargamos los datos del archivo CSV
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], na_values='?')
    df = df.drop('restecg', axis=1)
    df = df.drop('thalach', axis=1)
    df = df.drop('oldpeak', axis=1)
    df = df.drop('slope', axis=1)
    df = df.drop('ca', axis=1)
    df = df.drop('thal', axis=1)
    # Eliminamos filas con valores faltantes
    df = df.dropna()

    # Convertimos la columna de diagnóstico en un valor binario
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    # Creamos el modelo bayesiano
    model = BayesianNetwork([('age', 'sex'), ('age', 'chol'), ('age', 'fbs'), ('age', 'trestbps'), ('sex', 'chol'),
                  ('sex', 'fbs'), ('sex', 'trestbps'), ('fbs', 'cp'), ('trestbps', 'cp'), 
                  ('chol','cp'), ('fbs', 'exang'), ('trestbps', 'exang'), ('chol','exang'),('exang','num'),('cp','num')])

    # Estimamos las distribuciones de probabilidad usando MLE y BayesianEstimator
    # Estimamos las distribuciones de probabilidad usando MLE
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=303)

    # Hacemos inferencias en el modelo bayesiano
    infer = VariableElimination(model)

    #inferencia para un paciente con los datos especificados
    q = infer.query(['num'], evidence={'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,'exang': exang})

    return q    

# Función para devolver los valores normales o deseables de los parámetros
def valoresNormales(edad, sexo, cp, trestbps, chol, fbs, exang):
    #Valores deseables de los parametros
    valores ={
        'edad': (20, 80),
        'sexo': ('Hombre', 'Mujer'),
        'cp': ('Asintomático', 'Angina típica'),
        'trestbps': (90, 120),
        'chol': (150, 200),
        'fbs': (0, 120),
        'exang': ('No', 'Sí'),
    }


#Se crear la aplicación en dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server


# Se crea el diseño del tablero
app.layout = html.Div([    html.H1('Test rápido de enfermedad cardíaca', style={'text-align': 'center', 'color': 'White', 'font-weight': 'bold', 'background-color': '#b3d9ff'}),    
                       html.Div([        html.P('Mensaje de advertencia: Antes de tomar el test, debe realizarse los siguientes exámenes: presión sistolica, colesterol (Prueba de sangre), glucemia en ayunas (Prueba de sangre).')    ], style={'text-align': 'center', 'color': 'red', 'font-weight': 'bold'}),

    html.Br(),
    
    html.Div([
        html.Div([
            html.Label('¿Cuál es tu edad?'),
            dcc.Input(id='age', type='number', placeholder='Edad'),
        ], className='six columns', style={'margin-top': '10px'}),

        html.Div([
            html.Label('¿Eres hombre o mujer?'),
            dcc.Dropdown(id='sex', options=[{'label': 'Hombre', 'value': 1}, {'label': 'Mujer', 'value': 0}], placeholder='Sexo'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Tienes dolor torácico? Si es así, ¿qué tipo de dolor es?'),
            html.Ul([
                html.Li('Angina típica: dolor opresivo, generalmente en el centro del pecho, que se puede describir como una sensación de opresión, ardor, constricción o presión.'),
                html.Li('Angina atípica: puede presentar síntomas similares a la angina típica, pero puede tener una ubicación o características diferentes del dolor.'),
                html.Li('Dolor no anginal: puede ser descrito como un dolor punzante, agudo o quemante en el pecho, pero no está relacionado con el corazón.'),
                html.Li('Asintomático: ausencia de síntomas de dolor torácico.')
            ],style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='cp', options=[{'label': 'Angina típica', 'value': 1}, {'label': 'Angina atípica', 'value': 2},
                                        {'label': 'Dolor no anginal', 'value': 3}, {'label': 'Asintomático', 'value': 4}],
                        placeholder='Dolor torácico'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    
        html.Div([
            html.Label('¿Cuál es tu presión arterial en reposo?'),
            html.P('La presión arterial debe medirse cuando te encuentres en reposo, sentado o acostado, durante al menos 5 minutos. Asegúrate de no haber consumido cafeína ni haber realizado actividad física intensa en la última hora antes de la medición. Además, evita hablar o moverte mientras se toma la medida. Si tienes dudas acerca de cómo medir tu presión arterial, consulta con tu médico.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Input(id='trestbps', type='number', placeholder='Presión arterial en reposo'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Teniendo en cuenta tu anterior respuesta sobre dolor toracico, este es inducido por ejercicio?'),
            dcc.Dropdown(id='exang', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}],
                         placeholder='Angina inducida por ejercicio'),
        ], className='six columns'),
    ], className='row'),

    html.Div([
        html.Div([
            html.Label('¿Cuál es tu nivel de colesterol sérico?'),
            html.Div('El nivel de colesterol sérico debe ser medido después de un ayuno de al menos 12 horas. Además, se recomienda no consumir alcohol ni alimentos ricos en grasas durante las 24 horas previas al análisis de sangre. Consulta con tu médico para obtener más información sobre cómo prepararte para el análisis de colesterol.',
                     style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Input(id='chol', type='number', placeholder='Colesterol sérico'),
        ], className='six columns', style={'margin-top': '10px', 'margin-bottom': '10px'}),

        html.Div([
            html.Label('¿Glucemia en ayunas? Si es así, ¿es mayor a 120 mg/dl?'),
            html.P('La glucemia en ayunas debe ser medida después de un ayuno de al menos 8 horas. Además, se recomienda no consumir alcohol ni alimentos ricos en azúcar durante las 24 horas previas al análisis de sangre. Consulta con tu médico para obtener más información sobre cómo prepararte para el análisis de glucemia.',
                   style={'border': '1px solid white', 'padding': '10px', 'border-radius': '10px', 'margin-top': '10px', 'margin-bottom': '10px', 'background-color': '#d9d9d9'}),
            dcc.Dropdown(id='fbs', options=[{'label': 'Mayor a 120 mg/dl', 'value': 1}, {'label': 'Menor a 120 mg/dl', 'value': 0}],
                         placeholder='Azúcar en sangre en ayunas'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row rounded'),


    html.Br(),
    html.Button('Calcular', id='submit', n_clicks=0),
    html.Br(),
    html.Br(),
    html.Div(id='output'),


], className='container', style={'font-family': 'system-ui', 'background-color': '#f2f2f2'})

@app.callback(
    Output('output', 'children'),
    [Input('submit', 'n_clicks')],
    [State('age', 'value'),
     State('sex', 'value'),
     State('cp', 'value'),
     State('trestbps', 'value'),
     State('chol', 'value'),
     State('fbs', 'value'),
     State('exang', 'value')])


def calculate_probability(n_clicks, age, sex, cp, trestbps, chol, fbs, exang):
    if not n_clicks:
        return ''
    else:
        result = bayesian_inference(age, sex, cp, trestbps, chol, fbs, exang)
        probability = round(result.values[0], 2)
        return f'La probabilidad de tener una enfermedad cardíaca es del {probability*100}%'

#Se ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)