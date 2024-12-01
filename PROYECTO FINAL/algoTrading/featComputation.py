"""
Main file for Features Computation & Features Importance

Information flow:

    tsystem.dataprocess > featuresStage > featComputation
"""

# main_feature_script.py
from tsystem.utils import *
from featuresStage import classExecuteFeature

# Define los parámetros
pathFiles = "./bars/"
pathSaveFeatures = "./ProcessedFeatures/"
features_list = full_features_list  # O especifica una lista personalizada

# Inicializa la clase ExecuteFeature
execute_feature = classExecuteFeature(pathSaveFeatures, features_list=features_list)

# Ejecuta la función para computar y seleccionar features
result_messages = execute_feature.ExecuteFeature(
    pathFiles=pathFiles,
    save_part1=True,  # Computar y Guardar features globales
    save_part2=True,   # Seleccionar y Guardar features importantes
    n_estimators = 100,
    rs_value=42,
    importance_method = 'mdi'
)

# Imprime los resultados
for message in result_messages:
    print(message)
