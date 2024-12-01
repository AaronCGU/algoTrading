# Importamos librerías básicas...
import os
import zarr
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# Importamos TALIB
import talib

# Importamos TQDM (barra de progreso)
from tqdm import tqdm

# Librerías para el reto
import pytz
from datetime import datetime

# Ignoramos algunos warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

full_features_list = [
    'rsi', 'csspread', 'macd', 'bbands', 'stochastic', 'ema', 'sma', 'adx', 'atr', 'mfi', 
    'obv', 'cci', 'willr', 'dxo', 'ultosc', 'tema', 'trix', 'kama', 'mom', 'rocp', 
    'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 
    'alpha9',  'alpha11', 'alpha12', 'alpha13', 'alpha14', 'alpha15', 'alpha_naive'
]

################## Funciones necesarias para los alfas

def rank(series):
    return series.rank(pct=True)

def signedpower(x, p):
    return np.sign(x) * (np.abs(x) ** p)
    
def ts_argmax(series, window):
    return series.rolling(window).apply(np.argmax) + 1

def stddev(series, window):
    return series.rolling(window).std()

def correlation(x, y, window):
    return x.rolling(window).corr(y)

def delta(series, period):
    return series.diff(period)
    
def ts_rank(series, window):
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def covariance(x, y, window):
    return x.rolling(window).cov(y)

def ts_min(series, window):
    return series.rolling(window).min()

def ts_max(series, window):
    return series.rolling(window).max()

def delay(series, period):
    return series.shift(period)

###################### Funcion 1 -> dataprocess.py | triple_barrier_method()

def triple_barrier_method(df, max_holding_period, 
                          pt_factor=1.5, sl_factor=1.5, 
                          tripleBarrierLabels=(1,-1,0,0,0,2), 
                          column_std_name ='feature_stdBar'):
    """
    Función de la triple barrera.

    Inputs:
        - df: dataframe con las columnas `timestamp, open, high, low, close, vwap, total_volume, stdBar`
        - pt_factor/sl_factor: valores de escala profit-taking y stop-loss. 
                               Un valor inicial típico para estos factores suele ser en el rango de 1 a 3.
                               Recuerde que ambos valores escalarán la volatilidad definida por barra,
                               configurando la 'anchura' de la ventana.
        - 'max_holding_period': valor en segundos como maximo periodo de tenencia de la posicion.
    Outputs:
        - DataFrame con una nueva columna llamada 'TripleBarrier' que contiene 1, -1, 0 o 2.
          1 = Ganancia, -1 = Pérdida, 0 = Neutral, 2 = No hay datos suficientes para evaluar.

    La configuración de etiquetado es la estándar [1, 1, 1]
    """
    # Inicialización de la columna para almacenar las etiquetas de cada evento
    df['TripleBarrier'] = 0
    df['TripleBarrierPrice'] = 0

    # Iterar sobre cada fila del dataset
    for i in range(len(df)):
        entry_price = df['close'].iloc[i]
        entry_time = df['timestamp'].iloc[i]

        # Definir barreras
        upper_barrier = entry_price * (1 + pt_factor * df[column_std_name].iloc[i])
        lower_barrier = entry_price * (1 - sl_factor * df[column_std_name].iloc[i])

        # Flag para saber si se tocó alguna barrera
        touched_barrier = False

        # Iterar sobre las siguientes observaciones
        for j in range(i + 1, len(df)):
            current_price = df['close'].iloc[j]
            holding_period = df['timestamp'].iloc[j] - entry_time
            
            # Si el precio actual es superior a la barra horizotal superior
            # Y aun no se ha alcanzado el tiempo de espera maximo
            if current_price >= upper_barrier:
                df.at[i, 'TripleBarrier'] = tripleBarrierLabels[0]  # Toma de ganancias
                df.at[i, 'TripleBarrierPrice'] = current_price
                touched_barrier = True
                break
            # Si el precio actual es inferior a la barra horizontal inferior
            # Y aun no se ha alcanzado el tiempo de espera maximo
            elif current_price <= lower_barrier:
                df.at[i, 'TripleBarrier'] = tripleBarrierLabels[1]  # Detención de pérdidas
                df.at[i, 'TripleBarrierPrice'] = current_price
                touched_barrier = True
                break
            # Si el tiempo de espera maximo ha sido alcanzado
            elif holding_period.total_seconds() >= max_holding_period:
                if current_price > entry_price:
                    df.at[i, 'TripleBarrier'] = tripleBarrierLabels[2]  # Neutral
                elif current_price < entry_price:
                    df.at[i, 'TripleBarrier'] = tripleBarrierLabels[3]  # Neutral
                else:
                    df.at[i, 'TripleBarrier'] = tripleBarrierLabels[4]  # Neutral
                # Guardamos el precio
                df.at[i, 'TripleBarrierPrice'] = current_price
                touched_barrier = True
                break

        # Si no se tocó ninguna barrera y estamos en las últimas barras
        if not touched_barrier:
            if i + 1 == len(df) or (df['timestamp'].iloc[i + 1] - entry_time).total_seconds() < max_holding_period:
                df.at[i, 'TripleBarrier'] = tripleBarrierLabels[5]  # No hay suficientes datos para evaluar
                df.at[i, 'TripleBarrierPrice'] = 0 # Define its tripple barrier price as 0
    return df


###################### Funcion 1 -> dataprocess.py | alpha2()
def alpha2(v,c,o,window=20):
    """
    Alpha 2: 
        Params:
            - v: volume
            - c: close price
            - o: open
            - window: evaluation timeframe (in 'bar' terms)
    """
    alpha2 = [np.nan]*window
    for i in range(len(v))[window:]:
        alpha2.append(
            -1*np.log(v[i-window:i]).diff(2).corr(
                ((c[i-window:i]-o[i-window:i])/o[i-window:i]).rank(),
                method='spearman'
            )
        )
    return alpha2


###################### Funcion 3 -> dataprocess.py | corwinSchultz()
def corwinSchultz(df,mean_win=1):
    """
    Corwin-Schultz Spread.

    mean_win := window in bar-terms to evaluate the effective spread.
    """
    high=df['high']
    low =df['low']
    #get beta
    hl=np.log(high/low)**2
    beta=hl.rolling(window=2).sum()
    beta=beta.rolling(window=mean_win).mean()
    #get gamma
    h2=high.rolling(window=2).max()
    l2=low.rolling(window=2).min()
    gamma=np.log(h2/l2)**2
    #get alpha:
    den = 3 - 2*2**.5
    alpha=(2**.5-1)*(beta**.5)/den - (gamma/den)**.5
    alpha[alpha<0]=0 # set negative alphas to 0 (see p.727 of paper)
    #get Sigma
    k2=(8/np.pi)**.5
    den=3-2*2**.5
    sigma=(2**(-.5)-1)*beta**.5/(k2*den)+(gamma/(k2**2*den))**.5
    sigma[sigma<0]=0
    #get spread
    spread=2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    return spread, sigma