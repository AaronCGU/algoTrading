import os
import ray
from tsystem.dataprocess import DataBundle

@ray.remote
def process_and_save(zarr_file, 
                     pathFiles, 
                     pathSaveFeatures, 
                     init_date, 
                     last_date,
                     typeBar, 
                     labels, 
                     init_ema_window, 
                     alphaCalibrationValue,
                     max_holding_period, 
                     pt_factor, 
                     sl_factor, 
                     tripleBarrierLabels, 
                     column_std_name, 
                     show_progress):
    """
    Procesa un archivo .zarr y guarda las barras generadas como un archivo .csv.
    """
    # Informational Prints
    print("> Bar Computation Process:::")
    print(f">>>>> Equity: {zarr_file}")
    
    # Main Bars Computation Process
    try:
        dbObj = DataBundle(pathFiles, zarr_file, init_date, last_date)
        bars = dbObj.get_bars_and_labels(
            typeBar=typeBar,
            labels=labels,
            init_ema_window=init_ema_window,
            alphaCalibrationValue=alphaCalibrationValue,
            max_holding_period=max_holding_period,
            pt_factor=pt_factor,
            sl_factor=sl_factor,
            tripleBarrierLabels=tripleBarrierLabels,
            column_std_name=column_std_name,
            show_progress=show_progress 
        )
        csv_filename = os.path.splitext(zarr_file)[0] + '.csv'
        csv_path = os.path.join(pathSaveFeatures, csv_filename)
        bars.to_csv(csv_path, index=False)
        return f">> Process Finilized - Procesado y guardado: {csv_filename}"
    except Exception as e:
        return f">>> WARNING: Error en {zarr_file}: {str(e)}"

def ExecuteDataBundle(pathFiles, 
                      pathSaveFeatures, 
                      init_date, 
                      last_date,
                      typeBar="imbalance_volume", 
                      labels=True, 
                      init_ema_window=20,
                      alphaCalibrationValue=500, 
                      max_holding_period=100,
                      pt_factor=1.5, 
                      sl_factor=1.5,
                      tripleBarrierLabels=(1,-1,0,0,0,2),
                      column_std_name='feature_stdBar', 
                      show_progress=False):
    """
    Ejecuta el procesamiento de todos los archivos .zarr en una carpeta y guarda las barras generadas.
    """
    # Lectura de todos los zar files
    zarr_files = [f for f in os.listdir(pathFiles) if f.endswith('.zarr')]
    os.makedirs(pathSaveFeatures, exist_ok=True)
    # Procesamiento remoto de barras con Ray
    futures = [
        process_and_save.remote(
            zarr_file, pathFiles, pathSaveFeatures, init_date, last_date,
            typeBar, labels, init_ema_window, alphaCalibrationValue,
            max_holding_period, pt_factor, sl_factor, tripleBarrierLabels, 
            column_std_name, show_progress
        )
        for zarr_file in zarr_files
    ]
    # Ejecucion y almacenamiento de resultados (ray-get)
    results = ray.get(futures)
    return results