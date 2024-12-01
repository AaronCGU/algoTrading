"""
Archivo: `dataprocess.py`.
Funcionalidades:
    - AlternativeBars
    - DataBundle
    - Features Book
"""

# Importamos utils.py (contiene toda la info de paqueterias y complementarios)
from tsystem.utils import *

################################# Clases de archivo `dataprocess.py`
class AlternativeBars(object):
    """
    La clase `AlternativeBars` se utiliza para generar diferentes tipos de barras basadas en volumen 
    a partir de datos de alta frecuencia almacenados en un archivo Zarr. Los tipos de barras que puede generar son:

    - **Barras de Volumen Simples**: Intervalos donde cada barra contiene un número específico de unidades de volumen.
    - **Barras de Desequilibrio de Volumen**: Barras que se forman cuando el desequilibrio acumulado de volumen 
      alcanza un umbral dinámico, capturando el desequilibrio entre el volumen asociado a movimientos alcistas y bajistas.

    Esto permite analizar el comportamiento del mercado en función de la actividad comercial real, 
    proporcionando perspectivas más dinámicas que las barras basadas en tiempo.

    Esta nueva versión de la clase 'AlternativeBars' incluye en el output de la desviación estándar de cada barra.
    Sin embargo, la desviación estándar es calculada sobre los retornos de los precios agrupados en cada barra.
    """

    def __init__(self, zarr_file, stock_name, init_date, last_date):
        """
        Inicializa la clase `AlternativeBars` con los parámetros necesarios.

        Parámetros:
        - zarr_file: Ruta al archivo Zarr que contiene los datos de alta frecuencia.
        - stock_name: Nombre del archivo Zarr correspondiente al stock específico.
        - init_date: Fecha inicial del rango de análisis (formato 'YYYY-MM-DD').
        - last_date: Fecha final del rango de análisis (formato 'YYYY-MM-DD').
        """
        # Abre el archivo Zarr correspondiente al stock especificado y lo asigna a un atributo de la clase.
        self.zarrObject = zarr.open(zarr_file + stock_name)
        # Almacena el nombre del stock como atributo de la clase.
        self.stock_name = stock_name
        # Almacena la fecha inicial del rango como atributo de la clase.
        self.init_date = init_date
        # Almacena la fecha final del rango como atributo de la clase.
        self.last_date = last_date
        # Obtiene el calendario de la NYSE y lo asigna a un atributo de la clase.
        self.nyse = mcal.get_calendar('NYSE')
        # Convierte las fechas disponibles en el Zarr a un array de NumPy.
        self.zarrDates = np.array(self.zarrObject.date)
        # Selecciona los días de negociación entre las fechas especificadas.
        self.range_dates = self.sel_days(init_date, last_date)
        # Carga los datos filtrados de timestamps, precios, volúmenes y tipos.
        self.infoTimePriceVol = self.open_zarr_general(
            self.zarrDates, self.range_dates, self.zarrObject, self.stock_name
        )

    def sel_days(self, init, last):
        # Obtiene el horario de negociación entre las fechas especificadas utilizando el calendario de la NYSE.
        schedule = self.nyse.schedule(start_date=init, end_date=last)
        # Formatea las fechas y las devuelve como una lista de strings en formato 'YYYY-MM-DD'.
        return [date.strftime('%Y-%m-%d') for date in schedule.index.date]

    def open_zarr_general(self, zarrDates, range_dates, zarrObject, ref_stock_name=None):
        # Verifica que cada fecha en el rango seleccionado exista en las fechas disponibles en el Zarr.
        for date in range_dates:
            if date not in zarrDates:
                # Si una fecha no se encuentra, lanza un error indicando la fecha faltante.
                raise IndexError(f'Date {date} not found in zarrDates for {ref_stock_name}')

        # Encuentra el índice de inicio para la fecha inicial en el array de fechas del Zarr.
        start_idx = np.where(zarrDates == range_dates[0])[0][0]
        # Encuentra el índice de fin para la fecha final en el array de fechas del Zarr.
        end_idx = np.where(zarrDates == range_dates[-1])[0][0] + 1  # +1 para incluir el último índice.

        # Recupera los precios desde el índice de inicio hasta el índice de fin.
        prices = zarrObject.value[start_idx:end_idx]
        # Recupera los volúmenes desde el índice de inicio hasta el índice de fin.
        volume = zarrObject.vol[start_idx:end_idx]
        # Recupera los timestamps desde el índice de inicio hasta el índice de fin.
        ts_ = zarrObject.timestamp[start_idx:end_idx]
        # Recupera los tipos (por ejemplo, 'TRADE', 'BID', 'ASK') desde el índice de inicio hasta el índice de fin.
        types = zarrObject.type[start_idx:end_idx]

        # Crea un array booleano que indica qué timestamps son válidos (diferentes de cero).
        valid_idx = ts_ != 0
        # Filtra los arrays con los índices válidos
        ts_valid = ts_[valid_idx]
        prices_valid = prices[valid_idx]
        volume_valid = volume[valid_idx]
        types_valid = types[valid_idx]
    
        # NUEVO: Encuentra el primer índice donde el precio no es cero
        first_non_zero = np.argmax(prices_valid != 0)
    
        # Selecciona desde el primer precio no cero en adelante
        ts_valid = ts_valid[first_non_zero:]
        prices_valid = prices_valid[first_non_zero:]
        volume_valid = volume_valid[first_non_zero:]
        types_valid = types_valid[first_non_zero:]
        
        # Retorna los valores validos
        return ts_valid, prices_valid, volume_valid, types_valid

    def newVolumeBarConstruction(self, arrayTime, arrayPrice, arrayVol, arrayTypes, alphaCalibrationValue):
        # Calcula la suma acumulada del volumen.
        cumsumVol = np.cumsum(arrayVol)
        # Define el volumen por barra utilizando el valor de calibración proporcionado.
        vol_per_bar = alphaCalibrationValue
        # Asigna un identificador de grupo a cada transacción según el volumen acumulado.
        grpId = cumsumVol // vol_per_bar
        # Obtiene los índices donde se produce un cambio de grupo.
        idx = np.cumsum(np.unique(grpId, return_counts=True)[1])[:-1]
        # Divide el array de timestamps en grupos según los índices calculados.
        groupTime = np.split(arrayTime, idx)
        # Divide el array de precios en grupos según los índices calculados.
        groupPrice = np.split(arrayPrice, idx)
        # Divide el array de volúmenes en grupos según los índices calculados.
        groupVol = np.split(arrayVol, idx)
        # NUEVO: Divide el array de tipos en grupos según los índices calculados.
        groupTypes = np.split(arrayTypes, idx)
        # Devuelve los grupos de timestamps, precios, volúmenes y tipos
        return groupTime, groupPrice, groupVol, groupTypes

    def tick_rule(self, price_series, show_progress=False):
        # Calcula las diferencias de precios entre ticks consecutivos.
        price_diff = np.diff(price_series)
        # Inicializa el array de direcciones con ceros, del mismo tamaño que la serie de precios.
        b_t = np.zeros(len(price_series))
        # Establece el primer valor de b_t basado en el primer cambio de precio.
        b_t[0] = np.sign(price_diff[0]) if price_diff[0] != 0 else 1  # Asume dirección alcista si no hay cambio.
        # Define iterator
        iterator = range(1, len(price_diff))
        # Si el usuario decide mostrar el progreso
        if show_progress:
            # Iterator como objeto TQDM
            iterator = tqdm(iterator, desc="Tick Rule Computation")
        # Recorre las diferencias de precios para aplicar la regla de tick.
        for i in iterator:
            if price_diff[i] > 0:
                b_t[i] = 1  # Indica un incremento de precio.
            elif price_diff[i] < 0:
                b_t[i] = -1  # Indica una disminución de precio.
            else:
                b_t[i] = b_t[i-1]  # Mantiene el valor previo si no hay cambio.
        # Devuelve el array de direcciones.
        return b_t

    def ewma_window_func(self, array, window):
        # Calcula el promedio móvil exponencial (EWMA) del array proporcionado con la ventana especificada.
        ewma_array = pd.Series(array).ewm(span=window, adjust=False).mean().values
        # Devuelve el array con los valores suavizados.
        return ewma_array

    def generate_volume_bars(self, alphaCalibrationValue=500, show_progress=True):
        """
        Tipo de Barra 1: Barra de Volumen Estándar
    
        Parámetros:
        - alphaCalibrationValue: Tamaño del total de volumen a considerar en el acumulado por barra. 
    
        Retorna:
        - volume_bars: DataFrame con las barras de volúmen estándar para el valor alfa dado.        
        """
        # Obtiene los grupos de timestamps, precios, volúmenes y TIPOS utilizando 
        groupTime, groupPrice, groupVol, groupTypes = self.newVolumeBarConstruction(
            self.infoTimePriceVol[0],  # Array de timestamps.
            self.infoTimePriceVol[1],  # Array de precios.
            self.infoTimePriceVol[2],  # Array de volúmenes.
            self.infoTimePriceVol[3],  # >> NUEVO: Array de Tipos (TRADE, BID, ASK)
            alphaCalibrationValue      # Valor de calibración para el volumen por barra.
        )
    
        # Inicializa listas para almacenar los resultados
        vwap_list = []
        total_volume = []
        last_timestamp = []
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        std_per_bar = []
        ratio_bid_ask = []
        
        # Define el iterador
        iterator = zip(groupTime, groupPrice, groupVol, groupTypes)
        # Si el usuario desea mostrar el progreso
        if show_progress:
            # Define objeto TQDM
            iterator = tqdm(iterator, desc="Estandar Bars Construction | Processing Ticks Info", total=len(groupPrice))
        # Realiza la iteracion
        for group_time, group_price, group_vol, group_type in iterator:
            # Calcula VWAP
            vwap = np.sum(group_price * group_vol) / np.sum(group_vol)
            vwap_list.append(vwap)
            
            # Volumen total
            total_vol = np.sum(group_vol)
            total_volume.append(total_vol)
            
            # Último timestamp
            last_timestamp.append(group_time[-1])
            
            # Precio de apertura, máximo, mínimo y cierre
            open_prices.append(group_price[0])
            high_prices.append(np.max(group_price))
            low_prices.append(np.min(group_price))
            close_prices.append(group_price[-1])
            
            # Desviación estándar (std) por barra usando retornos
            if np.sum(group_price != 0) > 1:
                valid_prices = group_price[group_price != 0]
                returns = np.diff(valid_prices) / valid_prices[:-1]
                std_per_bar.append(np.std(returns))
            else:
                std_per_bar.append(np.nan)
            
            # Calcula ratio BID/ASK
            ratio_bid_ask.append(np.sum(group_type == 'BID') / (np.sum(group_type == 'ASK') or 1))
        
        # Convierte los timestamps a datetime y ajusta a la zona horaria de Nueva York
        last_timestamp = pd.to_datetime(last_timestamp, unit='ns', utc=True).tz_convert('America/New_York')
        
        # Crea un DataFrame con la información de las barras de volumen
        volume_bars = pd.DataFrame({
            'timestamp': last_timestamp,   # Marca temporal ajustada a la hora de Nueva York.
            'open': open_prices,           # Precio de apertura.
            'high': high_prices,           # Precio máximo.
            'low': low_prices,             # Precio mínimo.
            'close': close_prices,         # Precio de cierre.
            'vwap': vwap_list,             # Precio Promedio Ponderado por Volumen.
            'total_volume': total_volume,  # Volumen total de la barra.
            'feature_stdBar': std_per_bar,         # NUEVO: desviación estándar por barra
            'feature_bidAskRatio': ratio_bid_ask   # NUEVO: columna bid-ask ratio
        })
        
        # Calculando features de retornos
        volume_bars["feature_returns"] = (volume_bars.close - volume_bars.open)/volume_bars.open
        
        # Devuelve el DataFrame con las barras de volumen generadas
        return volume_bars

    def generate_volume_imbalance_bars(self, init_ema_window=20, show_progress=False):
        """
        Tipo de barra 2: Barra de Volumen de Desequilibrio, o Imbalance.

        Parámetros:
        - init_ema_window: Tamaño de la ventana inicial para el cálculo del EWMA (por defecto es 20).

        Retorna:
        - imbalance_bars: DataFrame con las barras de desequilibrio de volumen generadas.
        """
        # Extrae los timestamps, precios, volúmenes y tipos de los datos cargados.
        timestamps = self.infoTimePriceVol[0]
        prices = self.infoTimePriceVol[1]
        volumes = self.infoTimePriceVol[2]
        gtypes = self.infoTimePriceVol[3] # >>> NUEVO: trabajamos con tipos!

        # Aplica la regla de tick para obtener las direcciones de los movimientos de precio.
        tick_directions = self.tick_rule(prices)
        # Calcula el volumen firmado multiplicando las direcciones por los volúmenes.
        signed_volumes = tick_directions * volumes

        # Inicializa listas para almacenar los resultados de las barras.
        bar_timestamp = []
        bar_open = []
        bar_high = []
        bar_low = []
        bar_close = []
        bar_vwap = []
        bar_volume = []
        bar_bidaskratio = [] # >> NUEVO: almacena el ratio de bid/ask
        bar_std = [] # NUEVO : list vacía para almacenar la std por barra

        # Inicializa variables para el cálculo dinámico del desequilibrio.
        cumulative_theta = 0  # Desequilibrio acumulado.
        theta_list = []       # Lista para almacenar los valores absolutos del desequilibrio acumulado.
        ema_theta = None      # Valor actual del umbral dinámico calculado con EWMA.

        # Variables temporales para acumular datos de cada barra.
        temp_prices = []
        temp_volumes = []
        temp_times = []
        temp_gtpyes = []
        
        # Define el iterador
        iterador = range(len(prices)) 
        
        # Si el usuario decide mostrar el progreso
        if show_progress:
            # Define el iterador como objeto TQDM
            iterador = tqdm(
                range(len(prices)),
                desc="Imbalance Bars Construction | Processing Ticks Info"
                )

        # Itera sobre cada tick en los datos utilizando el iterador
        for i in iterador:
            # Actualiza el desequilibrio acumulado sumando el volumen firmado actual.
            cumulative_theta += signed_volumes[i]
            # Agrega el precio, volumen, timestamp y tipos actual a las listas temporales.
            temp_prices.append(prices[i])
            temp_volumes.append(volumes[i])
            temp_times.append(timestamps[i])
            temp_gtpyes.append(gtypes[i])

            # Agrega el valor absoluto del desequilibrio acumulado a la lista de theta.
            theta_list.append(abs(cumulative_theta))

            # Si se han acumulado suficientes valores, calcula el EWMA del desequilibrio.
            if len(theta_list) >= init_ema_window:
                ema_theta = self.ewma_window_func(theta_list, init_ema_window)[-1]

            # Verifica si el desequilibrio acumulado supera el umbral dinámico.
            if ema_theta is not None and abs(cumulative_theta) >= ema_theta:
                # Registra la marca temporal de la barra usando el último timestamp acumulado.
                bar_timestamp.append(temp_times[-1])
                # Registra el precio de apertura de la barra.
                bar_open.append(temp_prices[0])
                # Calcula y registra el precio máximo de la barra.
                bar_high.append(np.max(temp_prices))
                # Calcula y registra el precio mínimo de la barra.
                bar_low.append(np.min(temp_prices))
                # Registra el precio de cierre de la barra.
                bar_close.append(temp_prices[-1])
                # Calcula el VWAP de la barra.
                vwap = np.sum(np.array(temp_prices) * np.array(temp_volumes)) / np.sum(temp_volumes)
                bar_vwap.append(vwap)
                # Calcula y registra el volumen total de la barra.
                bar_volume.append(np.sum(temp_volumes))
                
                # >>> NUEVO: calcula la desviación estándar para la barra utilizando RETORNOS, y guarda la info
                temp_prices = np.array(temp_prices)
                nonZeroPrices = temp_prices[temp_prices != 0]
                validReturns = np.diff(nonZeroPrices) / nonZeroPrices[:-1]
                stdReturns = np.std(validReturns)
                bar_std.append(stdReturns) 

                # >>> NUEVO: calcula y almacena el ratio de tipos
                ask_and_ratios_array = np.array(temp_gtpyes)
                ratio_bid_ask = \
                np.sum(ask_and_ratios_array == 'BID') / (np.sum(ask_and_ratios_array == 'ASK') or 1)
                bar_bidaskratio.append(ratio_bid_ask)

                # Reinicia las variables temporales y el desequilibrio acumulado para iniciar una nueva barra.
                cumulative_theta = 0
                temp_prices = []
                temp_volumes = []
                temp_times = []
                theta_list = []
                ema_theta = None
                temp_gtpyes= []

        # Convierte las marcas temporales a datetime y las ajusta a la zona horaria de Nueva York.
        bar_timestamp = pd.to_datetime(bar_timestamp, unit='ns', utc=True).tz_convert('America/New_York')

        # Crea un DataFrame con la información de las barras de desequilibrio de volumen.
        imbalance_bars = pd.DataFrame({
            'timestamp': bar_timestamp,   # Marca temporal de la barra.
            'open': bar_open,             # Precio de apertura.
            'high': bar_high,             # Precio máximo.
            'low': bar_low,               # Precio mínimo.
            'close': bar_close,           # Precio de cierre.
            'vwap': bar_vwap,             # Precio Promedio Ponderado por Volumen.
            'total_volume': bar_volume,    # Volumen total de la barra.
            'feature_stdBar': bar_std,             # NUEVO: desviación estándar por barra
            'feature_bidAskRatio': bar_bidaskratio # NUEVO: columna bid-ask ratio
        })

        # Calculando feature de retornos (Close - Open / Open)
        imbalance_bars["feature_returns"] = (imbalance_bars.close - imbalance_bars.open)/imbalance_bars.open

        # Devuelve el DataFrame con las barras de desequilibrio de volumen generadas.
        return imbalance_bars


class DataBundle(AlternativeBars):
    """
    Clase que reune los procesos de:
        - Clase `AlternativeBars`
        - Función `triple_barrier_method`
    Para detalles específicos sobre parametrización, por favor revisar las clases mencionadas.
    """
    def __init__(self, zarr_file, stock_name, init_date, last_date):
        # Call the constructor of the AlternativeBars parent class
        super().__init__(zarr_file, stock_name, init_date, last_date)
        # Define Error Messages
        self.errorMsg1 = "Param 'init_ema_window' is not an positive integer."
        self.errorMsg2 = "Param 'alphaCalibrationValue' is not an positive integer."
        self.errorMsg3 = "Parm 'typeBar' is not an aceptable input."

    def get_bars_and_labels(self, typeBar, 
                            labels=True, 
                            init_ema_window=20, 
                            alphaCalibrationValue=500, 
                            max_holding_period=100, # Es en segundos!
                            pt_factor = 1.5, sl_factor = 1.5, 
                            tripleBarrierLabels = (1,-1,0,0,0,2), 
                            column_std_name ='feature_stdBar', 
                            show_progress=False):
        """
        Método principal: obtener barras y generar labels.
        """
        # Assert Evaluation 1 
        assert type(init_ema_window) == int, self.errorMsg1
        # Assert Evaluation 2
        assert type(alphaCalibrationValue) == int, self.errorMsg2
        # TypeBar is not in list
        assert typeBar.lower() in ["estandar_volume", "imbalance_volume"], self.errorMsg3

        # If selected bar is estandar
        if typeBar.lower() == "imbalance_volume":
            # Generating Bars
            bars = self.generate_volume_imbalance_bars(init_ema_window)
        elif typeBar.lower() == "estandar_volume":
            # Generating Bars
            bars = self.generate_volume_bars(alphaCalibrationValue)
        else:
            print("ErrorRaising: Not Available Bar-Generation Functionality")
            
        # If user wants also to generate the labels
        if labels:
            # Adding Labels
            bars = triple_barrier_method(
                df = bars, max_holding_period = max_holding_period,
                pt_factor=pt_factor, sl_factor=sl_factor,
                tripleBarrierLabels=tripleBarrierLabels,
                column_std_name=column_std_name
            ) 
        # Returning Bars Dataframe (adding the 2 triple barrier columns)
        return bars



class FeaturesBook(object):
    """
    A class to compute various features based on different methods 
    such as RSI, Alpha 2, Corwin-Schultz Spread, and more.
    """

    def __init__(self, df_bars):
        self.df = df_bars

    #################################################### ALPHAS ####################################################
    ################################################################################################################
    def calculate_alpha2(self, window=20):
        self.df['feature_alpha2'] = alpha2(v=self.df['total_volume'], c=self.df['close'], o=self.df['open'], window=window)

    def calculate_alpha1(self):
        self.df['feature_alpha1'] = rank(ts_argmax(signedpower(((self.df['feature_returns'] < 0).astype(int) * stddev(self.df['feature_returns'], 20) + self.df['close']), 2), 5)) - 0.5

    def calculate_alpha3(self):
        self.df['feature_alpha3'] = -1 * correlation(rank(self.df['open']), rank(self.df['total_volume']), 10)

    def calculate_alpha4(self):
        self.df['feature_alpha4'] = -1 * ts_rank(rank(self.df['low']), 9)

    def calculate_alpha5(self):
        self.df['feature_alpha5'] = rank((self.df['open'] - (sum(self.df['vwap'], 10) / 10))) * (-1 * abs(rank((self.df['close'] - self.df['vwap']))))

    def calculate_alpha6(self):
        self.df['feature_alpha6'] = -1 * correlation(self.df['open'], self.df['total_volume'], 10)

    def calculate_alpha7(self, thresh=100):
        self.df['feature_alpha7'] = ((thresh < self.df['total_volume']).astype(int) * (-1 * ts_rank(abs(delta(self.df['close'], 7)), 60)) * np.sign(delta(self.df['close'], 7))) - 1

    def calculate_alpha8(self):
        self.df['feature_alpha8'] =  np.exp(self.df['open']  / self.df['low'])

    def calculate_alpha9(self):
        self.df['feature_alpha9'] = np.where(ts_min(delta(self.df['close'], 1), 5) > 0, delta(self.df['close'], 1), np.where(ts_max(delta(self.df['close'], 1), 5) < 0, delta(self.df['close'], 1), -1 * delta(self.df['close'], 1)))

    # def calculate_alpha10(self):
    #     self.df['feature_alpha10'] = np.sum(np.where(ts_min(delta(self.df['close'], 1), 4) > 0, delta(self.df['close'], 1), np.where(ts_max(delta(self.df['close'], 1), 4) < 0, delta(self.df['close'], 1), -1 * delta(self.df['close'], 1))))

    def calculate_alpha11(self):
        self.df['feature_alpha11'] = (rank(ts_max((self.df['vwap'] - self.df['close']), 3)) + rank(ts_min((self.df['vwap'] - self.df['close']), 3))) * rank(delta(self.df['total_volume'], 3))

    def calculate_alpha12(self):
        self.df['feature_alpha12'] = np.sign(delta(self.df['total_volume'], 1)) * (-1 * delta(self.df['close'], 1))

    def calculate_alpha13(self):
        self.df['feature_alpha13'] = -1 * rank(covariance(rank(self.df['close']), rank(self.df['total_volume']), 5))

    def calculate_alpha14(self):
        self.df['feature_alpha14'] = -1 * rank(delta(self.df['feature_returns'], 3)) * correlation(self.df['open'], self.df['total_volume'], 10)

    def calculate_alpha15(self):
        self.df['feature_alpha15'] = -1 * (self.df['high']/ self.df['close'])

    def calcualte_naive_alpha(self):
        self.df["feature_alphaNaive"] = np.sqrt(np.exp(self.df['feature_returns'])) * np.pi
    
    ###################################################### AVANZADOS ###############################################
    ################################################################################################################
    def calculate_corwin_schultz(self, mean_win=1):
        self.df['feature_spread'], self.df['feature_sigma'] = corwinSchultz(self.df, mean_win=mean_win)

    ############################################### TECNICOS #######################################################
    ################################################################################################################
    def calculate_rsi(self):
        self.df['feature_RSI'] = talib.RSI(self.df['close'], timeperiod=14)
        
    def calculate_macd(self):
        self.df['feature_MACD'], self.df['feature_MACD_signal'], self.df['feature_MACD_hist'] = talib.MACD(self.df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    def calculate_bbands(self):
        self.df['feature_upper_band'], self.df['feature_middle_band'], self.df['feature_lower_band'] = talib.BBANDS(self.df['close'], timeperiod=20)

    def calculate_stochastic(self):
        self.df['feature_slowk'], self.df['feature_slowd'] = talib.STOCH(self.df['high'], self.df['low'], self.df['close'], fastk_period=14, slowk_period=3, slowd_period=3)

    def calculate_ema(self):
        self.df['feature_EMA'] = talib.EMA(self.df['close'], timeperiod=30)

    def calculate_sma(self):
        self.df['feature_SMA'] = talib.SMA(self.df['close'], timeperiod=30)

    def calculate_adx(self):
        self.df['feature_ADX'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

    def calculate_atr(self):
        self.df['feature_ATR'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

    def calculate_mfi(self):
        self.df['feature_MFI'] = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['total_volume'], timeperiod=14)

    def calculate_obv(self):
        self.df['feature_OBV'] = talib.OBV(self.df['close'], self.df['total_volume'])

    def calculate_cci(self):
        self.df['feature_CCI'] = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

    def calculate_willr(self):
        self.df['feature_WILLR'] = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

    def calculate_dxo(self):
        self.df['feature_DXO'] = talib.DX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

    def calculate_ultosc(self):
        self.df['feature_ULTOSC'] = talib.ULTOSC(self.df['high'], self.df['low'], self.df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

    def calculate_tema(self):
        self.df['feature_TEMA'] = talib.TEMA(self.df['close'], timeperiod=30)

    def calculate_trix(self):
        self.df['feature_TRIX'] = talib.TRIX(self.df['close'], timeperiod=30)

    def calculate_kama(self):
        self.df['feature_KAMA'] = talib.KAMA(self.df['close'], timeperiod=30)

    def calculate_mom(self):
        self.df['feature_MOM'] = talib.MOM(self.df['close'], timeperiod=10)

    def calculate_rocp(self):
        self.df['feature_ROCP'] = talib.ROCP(self.df['close'], timeperiod=10)


    def get_features(self, features_list):
        if len(features_list) == 1 and features_list[0].lower() == 'all':
            features_list = full_features_list
        
        for featName in tqdm(features_list, desc="Computing Features"):
            if featName == 'rsi':
                self.calculate_rsi()
            elif featName == 'alpha1':
                self.calculate_alpha1()
            elif featName == 'alpha2':
                self.calculate_alpha2()
            elif featName == 'alpha3':
                self.calculate_alpha3()
            elif featName == 'alpha4':
                self.calculate_alpha4()
            elif featName == 'alpha5':
                self.calculate_alpha5()
            elif featName == 'alpha6':
                self.calculate_alpha6()
            elif featName == 'alpha7':
                self.calculate_alpha7()
            elif featName == 'alpha8':
                self.calculate_alpha8()
            elif featName == 'alpha9':
                self.calculate_alpha9()
            # elif featName == 'alpha10':
            #     self.calculate_alpha10()
            elif featName == 'alpha11':
                self.calculate_alpha11()
            elif featName == 'alpha12':
                self.calculate_alpha12()
            elif featName == 'alpha13':
                self.calculate_alpha13()
            elif featName == 'alpha14':
                self.calculate_alpha14()
            elif featName == 'alpha15':
                self.calculate_alpha15()
            elif featName == 'alpha_naive':
                self.calcualte_naive_alpha()
            elif featName == 'csspread':
                self.calculate_corwin_schultz()
            elif featName == 'macd':
                self.calculate_macd()
            elif featName == 'bbands':
                self.calculate_bbands()
            elif featName == 'stochastic':
                self.calculate_stochastic()
            elif featName == 'ema':
                self.calculate_ema()
            elif featName == 'sma':
                self.calculate_sma()
            elif featName == 'adx':
                self.calculate_adx()
            elif featName == 'atr':
                self.calculate_atr()
            elif featName == 'mfi':
                self.calculate_mfi()
            elif featName == 'obv':
                self.calculate_obv()
            elif featName == 'cci':
                self.calculate_cci()
            elif featName == 'willr':
                self.calculate_willr()
            elif featName == 'dxo':
                self.calculate_dxo()
            elif featName == 'ultosc':
                self.calculate_ultosc()
            elif featName == 'tema':
                self.calculate_tema()
            elif featName == 'trix':
                self.calculate_trix()
            elif featName == 'kama':
                self.calculate_kama()
            elif featName == 'mom':
                self.calculate_mom()
            elif featName == 'rocp':
                self.calculate_rocp()
    
        return self.df