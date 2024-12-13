import os
from src.data.data_pipeline import data_pipeline
from src.utils.config import get_config
from src.mlflow_tracking.pp_tracking import log_preprocessing, log_splitter, log_temporal_splitter

config_dict = get_config('model')
temporal_data = config_dict['temporal_model']

def check_or_create_processed_data(raw_path, processed_path):
    """
    Verifica si existen datos procesados en la carpeta `processed`.
    Si no existen, corre el preprocesamiento para generarlos.
    """

    if not os.path.exists(processed_path):
        print(f"No se encontraron datos procesados en {processed_path}\n")
        print("Extrayendo datos para iniciar preprocesamiento...")
         # Get and merge data
        data = data_pipeline()
        print("Ejecutando preprocesamiento...")
        log_preprocessing(data, output_path=processed_path)
        print("Preprocesamiento completado. Datos procesados guardados.")
    else:
        print(f"Datos procesados encontrados en {processed_path}.")

def check_or_create_splitted_data(data, output_path):
    """
    Verifica si existen datos particionados en la carpeta `splits`.
    Si no existen, corre la partición para generarlos.
    """

    if os.path.exists(output_path) and os.path.isdir(output_path):
        if os.listdir(output_path):  # Verifica si la lista de contenido no está vacía
            print(f"Datos particionados encontrados en {output_path}.")
        
        else:
            print(f"No se encontraron datos particionados en {output_path}. Ejecutando partición...")
            if temporal_data:
                log_temporal_splitter(data, target='venta_total_neto', output_path=output_path)
            else:
                log_splitter(data, output_path=output_path)

            print("Partición completada. Datos particionados guardados.")
    else:
        print(f"La ruta '{output_path}' no es una carpeta válida.")