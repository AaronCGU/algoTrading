"""
Main file for Bars Computation

Information flow:

    tsystem.dataprocess > barsStage > barsComputation
"""

import ray
from barsStage import ExecuteDataBundle

# Inicializa Ray | num_cpus es un parametro IMPORTANTE!
ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=2)

# Define los parámetros
pathFiles = "./tickdata/"
pathSaveFeatures = "./bars/"
init_date = '2024-08-01'
last_date = '2024-08-31'

# Define parámetros de barras para estandar_volume
typeBar = "estandar_volume"
labels = True
# init_ema_window = 20  # No utilizado para estandar_volume
alphaCalibrationValue = 1000 # Umbral de volumen para barras estandar
max_holding_period = 900  # Es en segundos
pt_factor = 1.5
sl_factor = 1.5
tripleBarrierLabels = (1, -1, 0, 0, 0, 2)

# Ejecuta la función para procesar y guardar las barras estandar_volume
result_messages = ExecuteDataBundle(pathFiles = pathFiles,
                      pathSaveFeatures = pathSaveFeatures,
                      init_date = init_date,
                      last_date = last_date,
                      typeBar=typeBar,
                      labels=labels,
                      #init_ema_window=20,
                      alphaCalibrationValue=alphaCalibrationValue,
                      max_holding_period=max_holding_period,
                      pt_factor=pt_factor,
                      sl_factor=sl_factor,
                      tripleBarrierLabels=tripleBarrierLabels,
                      show_progress=False)

# Imprime los resultados
for message in result_messages:
    print(message)

# Cierra Ray
ray.shutdown()
