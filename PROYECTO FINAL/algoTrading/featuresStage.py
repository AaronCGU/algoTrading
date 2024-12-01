import os
import ray
import pickle
import pandas as pd
from tsystem.utils import *
from datetime import datetime
from tsystem.featImportance import FeatImp
from sklearn.preprocessing import MinMaxScaler
from tsystem.dataprocess import triple_barrier_method, FeaturesBook

@ray.remote
def compute_and_save_features(csv_file, pathFiles, pathSaveFeatures, features_list):
    try:
        # Cargar el dataframe
        df = pd.read_csv(os.path.join(pathFiles, csv_file))

        # Inicializar FeaturesBook
        fb = FeaturesBook(df)
        # Calcular features
        fb.get_features(features_list)
        df_features = fb.df

        # Guardar features globales
        global_filename = os.path.splitext(csv_file)[0] + '_Features_Global.csv'
        df_features.to_csv(os.path.join(pathSaveFeatures, global_filename), index=False)

        return global_filename, df_features
    except Exception as e:
        return f"Error en {csv_file}: {str(e)}", None

#### Clase Principal : Features Computation & Selection
class classExecuteFeature(object):
    def __init__(self, pathSaveFeatures, features_list=None):
        """
        Inicializa la clase ExecuteFeature.

        Parámetros:
        - pathSaveFeatures (str): Ruta donde se guardarán los archivos de features.
        - features_list (list): Lista de nombres de features a calcular. Si es None, usa todas.
        """
        self.pathSaveFeatures = pathSaveFeatures
        self.features_list = features_list if features_list is not None else full_features_list

    def escaling_procedure(self, features_df_input):
        """
        Realiza el escalado de las características.

        Parámetros:
        - features_df (DataFrame): DataFrame de características a escalar.

        Retorna:
        - scaled_features (DataFrame): DataFrame de características escaladas.
        """
        print(">>> Scalling Process Initialized")
        # Creamos una copia por seguridad
        features_df = features_df_input.copy()
        # Eliminamos inf y NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        # Inicializa el escalador
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(features_df)

        # Convertir a DataFrame
        scaled_features = pd.DataFrame(
            scaled_array,
            columns=features_df.columns, index=features_df.index
            )

        # Extract current time to personalized scalar pickle name
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m-%d-%H-%M")

        # Guardar el objeto scaler como pickle...
        scaler_filename = os.path.join(
            # self.pathSaveFeatures, f'scaler_pickle_{formatted_time}.pkl'
            self.pathSaveFeatures, f'scaler_pickle.pkl'
            )
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(">>> Scaling Process Finalized")
        return scaled_features

    def ExecuteFeature(self, pathFiles,
                       save_part1=True,
                       save_part2=True,
                       rayCores = 3,
                       n_estimators = 200,
                       rs_value=42,
                       importance_method = 'mdi'):
        """
        Ejecuta el procesamiento de features y la selección de features.

        Parámetros:
        - pathFiles (str): Ruta de la carpeta que contiene los archivos .csv.
        - save_part1 (bool): Si es True, guarda los features globales.
        - save_part2 (bool): Si es True, guarda los features seleccionados.

        Retorna:
        - Lista de mensajes de éxito o error para cada archivo procesado.
        """
        messages = []
        dataframes = {}

        # Proceso de computacion de features globales!
        if save_part1:
            # Lista todos los archivos .csv en pathFiles que no son de features
            csv_files = \
                [
                    f for f in os.listdir(pathFiles)
                    if f.endswith('.csv')
                    and not f.endswith('_Features_Global.csv')
                    and not f.endswith('_Features_Selected.csv')
                    ]

            # Inicia Ray si no está iniciado
            if not ray.is_initialized():
                ray.init(
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    num_cpus=rayCores
                    )

            # Ejecuta el procesamiento paralelo
            futures = [
                compute_and_save_features.remote(
                    csv_file, pathFiles, self.pathSaveFeatures, self.features_list
                )
                for csv_file in csv_files
            ]

            results = ray.get(futures)

            # Guardar mensajes y dataframes
            for result in results:
                message, df_features = result
                messages.append(message)
                if df_features is not None and isinstance(df_features, pd.DataFrame):
                    equity_name = os.path.splitext(result[0])[0].replace('_Features_Global', '')
                    dataframes[equity_name] = df_features

        # Proceso de seleccion de features
        if save_part2:
            if not save_part1:
                # Lista todos los archivos _Features_Global.csv en pathSaveFeatures
                global_files = [f for f in os.listdir(self.pathSaveFeatures) if f.endswith('_Features_Global.csv')]
                for f in global_files:
                    # Definimos el nombre del equity
                    equity_name = os.path.splitext(f)[0].replace('_Features_Global', '')
                    # Leemos el csv que contiene los features del equity
                    df = pd.read_csv(os.path.join(self.pathSaveFeatures, f))
                    # Ahora, para los fines de seleccion de features
                    # Debemos eliminar las observaiones
                    # con el label 'TripleBarrier' = 2 (not enough data)
                    df = df.query("TripleBarrier !=2").reset_index(drop=True)
                    # Save the information in the main dictionary
                    dataframes[equity_name] = df
            else:
                # dataframes ya contiene los dataframes procesados en save_part1
                pass

            if not dataframes:
                messages.append("No hay dataframes para procesar en la parte 2.")
                return messages

            # Concatenar todos los dataframes
            stacked_df = pd.concat(dataframes.values(), ignore_index=True)
            stacked_df = stacked_df.sort_values(by='timestamp')

            # Seleccionar solo columnas con prefijo 'feature_'
            feature_columns = [col for col in stacked_df.columns if col.startswith('feature_')]

            # Extraer labels
            labels = stacked_df['TripleBarrier']

            # Extraer labels-price
            labels_price = stacked_df['TripleBarrierPrice']

            # Extraer características
            features = stacked_df[feature_columns]

            # Ejecutar procedimiento de escalado
            scaled_features = self.escaling_procedure(features)

            # Extraer los timestamps correspondientes usando el índice de scaled_features
            timestamps_scaled = stacked_df.loc[scaled_features.index, 'timestamp']

            # Extraer los precios correspondientes usando el indice de scaled_features
            close_scaled = stacked_df.loc[scaled_features.index, 'close']

            # Crear un nuevo DataFrame con las características escaladas y el timestamp como índice
            scaled_features_with_time = scaled_features.copy()
            scaled_features_with_time.index = timestamps_scaled

            # Alinear labels con features escalados
            labels_aligned = labels.loc[scaled_features.index]

            # Alinear labels-price con features escalados
            labels_price_aligned = labels_price.loc[scaled_features.index]

            print(f":::>>> Feature Importance | Procedure: {importance_method} | Status: Active")
            print(f"----------> Input Features Matrix Shape: {scaled_features.shape}")
            # Ejecutar Feature Importance
            feat_imp = FeatImp(
                method=importance_method,
                automatize=True,
                labels_return=True,
                nestimators=n_estimators,
                rs=rs_value
                )
            selected_features = feat_imp.fit_transform(
                target=labels_aligned,
                features=scaled_features, # Features stacked y scaled
                only_list_features=True
            )
            print(f":::>>> Feature Importance | Procedure: {importance_method} | Status: Finalized")
            selected_features_list = selected_features # feat_imp.selected_features

            # Dataframe con columnas 'entry' + 'exit' (close price + labels-price)
            tradingInfoDf = pd.concat([close_scaled, labels_price_aligned], axis=1)
            tradingInfoDf.columns = ['entry', 'exit']

            # 1) PROCEDIMIENTO GENERICO DE ALMACENAMIENTO
            # a) Guardar el dataset stackeado y escalado solo con los feat seleccionados
            # > Guardando el dataset de features
            selected_scaled_features_with_time = \
                scaled_features_with_time[selected_features_list]
            # > Procediendo con el guardado de feat preservando el indice temporal...
            selected_scaled_features_with_time.to_csv(
                os.path.join(
                    self.pathSaveFeatures, "stacked_scaled_features.csv"),
                index=True
                )
            # > Guardando el vector de labels
            labels_aligned.to_csv(
                os.path.join(
                    self.pathSaveFeatures, "aligned_labels.csv"),
                index=False
                )
            # > Guardando el vector de precios (close + labels)
            tradingInfoDf.to_csv(
                os.path.join(
                    self.pathSaveFeatures, "aligned_tprices.csv"),
                index=False
                )

            # Guardar features seleccionados por equity
            for equity, df_features in dataframes.items():

                # Seleccionar features importantes
                df_selected = df_features[selected_features_list].copy()
                df_selected['TripleBarrier'] = df_features['TripleBarrier']
                df_selected['timestamp'] = df_features["timestamp"]
                # Guardar el archivo seleccionado
                selected_filename = equity + '_Features_Selected.csv'
                df_selected.to_csv(
                    os.path.join(
                        self.pathSaveFeatures, selected_filename),
                    index=False
                    )

            messages.append("Features seleccionados y guardados correctamente.")

        return messages
