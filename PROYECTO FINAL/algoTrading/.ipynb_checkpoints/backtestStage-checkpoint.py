"""
Main file for Model Tuning & Combinatorial Backtest 

Information flow: 

    Organic (no dependencies)
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.base import clone
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Importar ray para procesamiento en paralelo
import ray
# Importar tqdm para barras de progreso
from tqdm import tqdm


############################ Funcion de Org. De Archivos
def organizingData(genPath, features_filename, labels_filename, prices_filename):
    """
    Lee y procesa los datos de características, etiquetas y precios.

    Args:
        genPath (str): Ruta general donde se encuentran los archivos.
        features_filename (str): Nombre del archivo CSV de características.
        labels_filename (str): Nombre del archivo CSV de etiquetas.
        prices_filename (str): Nombre del archivo CSV de precios.

    Returns:
        features (pd.DataFrame): DataFrame de características con timestamp como índice.
        labels (pd.DataFrame): DataFrame de etiquetas alineadas con las características.
        prices (pd.DataFrame): DataFrame de precios alineados con las características.
    """

    # Construir rutas completas de los archivos
    features_path = os.path.join(genPath, features_filename)
    labels_path = os.path.join(genPath, labels_filename)
    prices_path = os.path.join(genPath, prices_filename)

    # Leer las características
    features = pd.read_csv(features_path)
    # Transformar timestamp en UNIX Timestamp
    # '%Y-%m-%d %H:%M:%S.%f%z'
    features.timestamp = pd.to_datetime(features.timestamp, format='%Y-%m-%d %H:%M:%S%z').astype(np.int64) // 10**6
    # Establecer el timestamp como índice
    features = features.set_index("timestamp")
    
    # Leer las etiquetas
    labels = pd.read_csv(labels_path)
    # Establecer el mismo índice para las etiquetas
    labels.index = features.index
    
    # Leer los precios
    prices = pd.read_csv(prices_path)
    # Establecer el mismo índice para los precios
    prices.index = features.index
    
    return features, labels, prices


####################### Clase Principal: Ecosystema de Trading (Backtest)
class tradingEco(object):
    """
    Clase tradingEco principal.
    
    Permite realizar el tuning de modelos exógenos (BetSide) y endógenos (BetSize),
    ejecutar un backtest combinatorial, y guardar los resultados y modelos.
    """
    
    def __init__(self, features: pd.DataFrame, labels: pd.Series, 
                 prices: pd.DataFrame, scale_features_input: bool = False):
        """
        Inicializa la clase tradingEco.
        """
        # Almacenar las características originales
        self.features = features
        # Almacenar las etiquetas
        self.labels = labels
        # Almacenar los precios
        self.prices = prices
        # Indicar si se deben escalar las características
        self.scale_features_input = scale_features_input
        # Crear un escalador si es necesario
        self.scaler = StandardScaler() if self.scale_features_input else None
        # Escalar las características si es necesario
        if self.scale_features_input:
            # Escalar y convertir a DataFrame manteniendo el índice y las columnas
            self.features_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.features),
                columns=self.features.columns,
                index=self.features.index
            )
        else:
            # Copiar las características sin escalar
            self.features_scaled = self.features.copy()
        # Inicializar modelos exógeno y endógeno
        self.exogenous_model = None
        self.endogenous_model = None
        # Inicializar DataFrames para métricas y resultados
        self.metrics = pd.DataFrame()
        self.backtest_results = pd.DataFrame()
    
    def tune_model(self, model, param_grid, cv=5, num_cpus=2):
        """
        Realiza el tuning de un modelo usando GridSearchCV.
        """
        # Crear instancia de GridSearchCV con paralelización
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=cv, scoring='accuracy', n_jobs=num_cpus)
        # Ajustar el modelo a los datos escalados
        grid_search.fit(self.features_scaled, self.labels)
        # Imprimir los mejores parámetros y score
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor score: {grid_search.best_score_}")
        # Retornar el mejor estimador
        return grid_search.best_estimator_
    
    def calculate_financial_metrics(self, returns, capital):
        """
        Calcula métricas financieras básicas a partir de los retornos.
        """
        # Inicializar diccionario de métricas
        metrics = {}
        # Reemplazar infinitos y eliminar NaNs
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        # Comprobar si hay retornos disponibles
        if len(returns) == 0:
            # Asignar NaN si no hay datos
            metrics['Cumulative Return'] = np.nan
            metrics['Annualized Return'] = np.nan
            metrics['Annualized Volatility'] = np.nan
            metrics['Sharpe Ratio'] = np.nan
            metrics['Max Drawdown'] = np.nan
            metrics['Win Rate'] = np.nan
            return metrics
        # Calcular retorno total
        total_return = returns.sum()
        # Calcular retorno acumulado
        metrics['Cumulative Return'] = total_return / capital
        # Calcular retornos diarios en porcentaje
        daily_returns = returns / capital
        # Calcular retorno anualizado
        metrics['Annualized Return'] = np.mean(daily_returns) * 252 * 100
        # Calcular volatilidad anualizada
        metrics['Annualized Volatility'] = np.std(daily_returns) * np.sqrt(252) * 100
        # Calcular ratio de Sharpe
        # Asumiendo 'rf' = 0
        if metrics['Annualized Volatility'] != 0:
            metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility']
        else:
            metrics['Sharpe Ratio'] = 0
        # Calcular retornos acumulados diarios
        cumulative_returns = daily_returns.cumsum()
        # Calcular el pico acumulado
        peak = cumulative_returns.cummax()
        # Calcular drawdown
        drawdown = cumulative_returns - peak
        # Calcular máximo drawdown
        metrics['Max Drawdown'] = drawdown.min()
        # Calcular tasa de ganancias (media de retornos positivos! - NAIVE!)
        metrics['Win Rate'] = (returns > 0).mean()
        # Retornar métricas calculadas
        return metrics
    
    def get_multi_process(self, dict_exo_models: dict, endogenous_model, 
                          N: int, k: int, capital: float = 1000, commissions: float = 1.0,
                          activate_tuning: bool = False, path_location: str = '', 
                          name_model_pickle_file: str = '', num_cpus: int = 2):
        """
        Realiza el tuning de modelos, ejecuta el backtest y guarda los resultados.
        """
        # Imprimir inicio del proceso
        print("> Iniciando proceso de backtest")
        
        # Verificar si se activa el tuning
        if activate_tuning:
            # Inicializar Ray para el tuning
            ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=num_cpus)
            # Tuning de modelos exógenos
            print("> Tuning de modelos exógenos")
            tuned_exo_models = {}
            # Iterar sobre los modelos exógenos (dados como input en el diccionario `dict_exo_models` por el usuario!)
            for name, (model, params) in dict_exo_models.items():
                print(f"::>> Tuning modelo exógeno: {name}")
                # Obtener el mejor modelo utilizando Ray
                best_model = self.tune_model(model, params, num_cpus=num_cpus)
                # Almacenar la mejor parametrizacion de cada modelo!
                tuned_exo_models[name] = best_model
            # Seleccionar el mejor modelo basado en `accuracy` (consejo de pata: CAMBIEN ESTO!)
            print("> Seleccionando el mejor modelo exógeno basado en accuracy completa")
            best_accuracy = 0 
            for name, model in tuned_exo_models.items():
                score = model.score(self.features_scaled, self.labels) # ACCURACY!
                print(f"::>> Modelo: {name}, Accuracy: {score}")
                if score > best_accuracy:
                    best_accuracy = score
                    # Clonamos el modelo; es decir, creamos un modelo identico (con los mismos parametros)
                    # pero 'VACIO' en terminos de aprendizaje (es decir, no esta entrenado!)
                    self.exogenous_model = clone(model)
                    self.exo_model_name = name
            print(f"> Mejor modelo exógeno: {self.exo_model_name} con accuracy {best_accuracy}")
            # Guardar el modelo exógeno
            exo_pickle_path = f"{path_location}/exogenous_model_{self.exo_model_name}.pkl"
            with open(exo_pickle_path, 'wb') as f:
                pickle.dump(self.exogenous_model, f)
            print(f"::>> Modelo exógeno guardado en {exo_pickle_path}")
            # Cerrar Ray después del tuning
            ray.shutdown()
        else:
            # Cargar el modelo exógeno desde archivo
            exo_pickle_path = f"{path_location}/{name_model_pickle_file}"
            with open(exo_pickle_path, 'rb') as f:
                self.exogenous_model = pickle.load(f)
            print(f"> Modelo exógeno cargado desde {exo_pickle_path}")
        
        # Entrenar modelo endógeno
        print("> Entrenando el modelo endógeno para BetSize")
        # Filtramos eventos que no nos interesan
        # Es decir, eventos donde no existe 'ejecucion' (labels = 0)
        mask = self.labels != 0
        # Seleccionamos a traves del filtro (solo eventos 1 y -1)
        X_endo = self.features_scaled[mask]
        # Redefinimos los eventos 1 y -1 con 1 y 0 respectivamente
        y_endo = self.labels[mask].map({-1: 0, 1: 1}).astype(int)
        self.endogenous_model = clone(endogenous_model)
        self.endogenous_model.fit(X_endo, y_endo)
        # Guardar el modelo endógeno
        endo_pickle_path = f"{path_location}/endogenous_model.pkl"
        with open(endo_pickle_path, 'wb') as f:
            pickle.dump(self.endogenous_model, f)
        print(f"::>> Modelo endógeno guardado en {endo_pickle_path}")
        
        ###################### INICIO DE BACKTEST ######################
        # Ejecutar backtest con Combinatorial Cross-Validation
        print("> Ejecutando backtest con Combinatorial Cross-Validation")
        backtest_list = []
        n_samples = len(self.features_scaled)
        group_size = n_samples // N
        groups = np.array([i for i in range(N) for _ in range(group_size)])
        # Manejar muestras restantes
        if len(groups) < n_samples:
            groups = np.append(groups, [N - 1] * (n_samples - len(groups)))
        else:
            groups = groups[:n_samples]
        # Definimos el indice de grupos para cada evento (fila) en particular
        # Donde los eventos corresponden a la combinacion de vector de features,
        # valor de etiqueta (1 o -1) y precios de entrada/salida.
        group_indices = [np.where(groups == i)[0] for i in range(N)]
        all_combinations = list(combinations(range(N), k))
        total_splits = len(all_combinations)
        
        # Procesar cada split sin usar Ray
        for idx, test_group_indices in enumerate(tqdm(all_combinations, desc="------> Ejecutando splits de backtest")):
            print(f"::>> Split {idx + 1}/{total_splits}")
            # Obtener índices de prueba y entrenamiento
            test_indices = np.concatenate([group_indices[i] for i in test_group_indices])
            train_group_indices = [i for i in range(N) if i not in test_group_indices]
            train_indices = np.concatenate([group_indices[i] for i in train_group_indices])
            # Dividir datos en entrenamiento y prueba
            X_train, X_test = self.features_scaled.iloc[train_indices], self.features_scaled.iloc[test_indices]
            y_train, y_test = self.labels.iloc[train_indices], self.labels.iloc[test_indices]
            prices_test = self.prices.iloc[test_indices]
            # Entrenar modelos
            exo = clone(self.exogenous_model)
            exo.fit(X_train, y_train)
            endo = clone(self.endogenous_model)
            mask_train = y_train != 0
            endo.fit(X_train[mask_train], y_train[mask_train].map({-1: 0, 1: 1}).astype(int))
            # Realizar predicciones
            y_pred_exo = exo.predict(X_test)
            y_pred_betsize = endo.predict_proba(X_test)[:,1]
            # Combinar predicciones para obtener señales
            y_final_side = y_pred_exo
            y_final_size = y_pred_betsize
            # Senal de trading...!
            signals = y_final_side * y_final_size
            signals = pd.Series(signals, index=X_test.index)
            signals = signals.where(y_final_side != 0, 0)
            # Calcular retornos
            returns = signals * ((prices_test['exit'] - prices_test['entry']) / prices_test['entry'])
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            # Aplicar comisiones::: LA PARTE MAS CRITICA DEL BACKTEST!
            num_transactions = signals != 0
            returns[num_transactions] -= commissions / capital
            returns = returns * capital
            # Calcular métricas financieras
            financial_metrics = self.calculate_financial_metrics(returns, capital)
            # Calcular métricas de modelo
            report = classification_report(y_test, y_final_side, output_dict=True, zero_division=0)
            model_metrics = {
                'precision_1': report.get('1', {}).get('precision', 0),
                'recall_1': report.get('1', {}).get('recall', 0),
                'f1_1': report.get('1', {}).get('f1-score', 0),
                'precision_neg1': report.get('-1', {}).get('precision', 0),
                'recall_neg1': report.get('-1', {}).get('recall', 0),
                'f1_neg1': report.get('-1', {}).get('f1-score', 0),
            }
            # Crear diccionario de resultados
            result = {
                'split': idx + 1,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'Cumulative Return': financial_metrics['Cumulative Return'],
                'Annualized Return': financial_metrics['Annualized Return'],
                'Annualized Volatility': financial_metrics['Annualized Volatility'],
                'Sharpe Ratio': financial_metrics['Sharpe Ratio'],
                'Max Drawdown': financial_metrics['Max Drawdown'],
                'Win Rate': financial_metrics['Win Rate'],
                'precision_1': model_metrics['precision_1'],
                'recall_1': model_metrics['recall_1'],
                'f1_1': model_metrics['f1_1'],
                'precision_neg1': model_metrics['precision_neg1'],
                'recall_neg1': model_metrics['recall_neg1'],
                'f1_neg1': model_metrics['f1_neg1']
            }
            # Agregar resultados a la lista
            backtest_list.append(result)
        
        # Guardar resultados del backtest
        self.backtest_results = pd.DataFrame(backtest_list)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backtest_csv_path = f"{path_location}/backtest_results_{timestamp}.csv"
        self.backtest_results.to_csv(backtest_csv_path, index=False)
        print(f"> Resultados del backtest guardados en {backtest_csv_path}")

