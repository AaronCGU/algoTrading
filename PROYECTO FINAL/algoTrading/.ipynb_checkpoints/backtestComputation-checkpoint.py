# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:37:39 2024

@author: frank
"""
# Importamos funcionalidades del stage Backtest
from backtestStage import organizingData, tradingEco

# Importamos modelos de Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# I) Leemos los .csv de features, labels y precios
# genPath = "C:/Users/frank/Desktop/algoTrading/ProcessedFeatures"
genPath = "C:/Users/Dell/Desktop/OTROS/TRADING ALGORITMICO CON PYTHON/S7 BACKTESTING/algoTrading/ProcessedFeatures"
features_filename = "stacked_scaled_features.csv"
labels_filename = "aligned_labels.csv"
prices_filename = "aligned_tprices.csv"

#    Almacenamos la informacion
features, labels, prices = organizingData(
    genPath, features_filename, labels_filename, prices_filename
    )


# II) Inicializamos la clase con los datos de features, labels y precios
trader = tradingEco(features=features, labels=labels.TripleBarrier, prices=prices)

#    Definimos modelos exógenos y sus grids de parámetros para el tuning
dict_exo_models = {
    'RandomForest': (
        RandomForestClassifier(random_state=42), 
        {
            'n_estimators': [50, 100],
            'max_depth': [10, 30]
        }
    ),
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, random_state=42), 
        {
            'C': [0.1, 10],
            'solver': ['liblinear']
        }
    )
}

#   Definimos el modelo endógeno para BetSize (usualmente, un mod. de arboles)
endogenous_model = RandomForestClassifier(random_state=42, n_estimators=100)

# III) Definimos los parámetros del backtest
N = 3 # Grupos
k = 2 # Particiones
capital = 10000 # Capital inicial (en USD)
commissions = 0 # Comisiones (en USD)
activate_tuning = True  # False solo cuando ya se tiene los modelos tuneados
path_location = "C:/Users/Dell/Desktop/OTROS/TRADING ALGORITMICO CON PYTHON/S7 BACKTESTING/algoTrading/backtestResults"  # Reemplazar con la ruta deseada
# path_location = 'C:/Users/frank/Desktop/algoTrading/backtestResults'  # Reemplazar con la ruta deseada
name_model_pickle_file = 'exogenous_model_testn1.pkl'  # Nombre de Archivo pickle si activate_tuning=False
num_cpus = 2 # CPU's a utilizarse con RAY

# Ejecutar el proceso 
trader.get_multi_process(
    dict_exo_models=dict_exo_models,
    endogenous_model=endogenous_model,
    N=N,
    k=k,
    capital=capital,
    commissions=commissions,
    activate_tuning=activate_tuning,
    path_location=path_location,
    #name_model_pickle_file=name_model_pickle_file,
    num_cpus=num_cpus
)
