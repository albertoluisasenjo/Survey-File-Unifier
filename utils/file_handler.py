"""
Módulo para manejo de archivos SPSS
"""

import pandas as pd
import pyreadstat
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List
from io import BytesIO


def load_spss_file(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """
    Carga un archivo SPSS desde un objeto UploadedFile de Streamlit
    
    Args:
        uploaded_file: Archivo subido por Streamlit
        
    Returns:
        Tuple con (DataFrame, metadata)
    """
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Leer con pyreadstat
        df, meta = pyreadstat.read_sav(tmp_path)
        
        # Limpiar archivo temporal
        Path(tmp_path).unlink()
        
        # Convertir metadata a dict para facilitar el uso
        metadata = {
            'column_names': meta.column_names,
            'column_labels': meta.column_names_to_labels,
            'variable_value_labels': meta.variable_value_labels,
            'number_rows': meta.number_rows,
            'number_columns': meta.number_columns,
        }
        
        return df, metadata
    
    except Exception as e:
        raise Exception(f"Error al cargar {uploaded_file.name}: {str(e)}")


def export_to_spss(df: pd.DataFrame) -> BytesIO:
    """
    Exporta un DataFrame a formato SPSS (.sav) en memoria
    
    Args:
        df: DataFrame a exportar
        
    Returns:
        BytesIO con el archivo .sav
    """
    output = BytesIO()
    
    # Escribir a archivo temporal primero
    with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Escribir con pyreadstat
        pyreadstat.write_sav(df, tmp_path)
        
        # Leer a BytesIO
        with open(tmp_path, 'rb') as f:
            output.write(f.read())
        
        output.seek(0)
        
    finally:
        # Limpiar archivo temporal
        Path(tmp_path).unlink(missing_ok=True)
    
    return output


def get_variable_info(df: pd.DataFrame, metadata: dict, var_name: str) -> Dict:
    """
    Obtiene información completa de una variable
    
    Args:
        df: DataFrame con los datos
        metadata: Metadata del archivo SPSS
        var_name: Nombre de la variable
        
    Returns:
        Diccionario con información de la variable
    """
    info = {
        'name': var_name,
        'label': metadata['column_labels'].get(var_name, ''),
        'type': str(df[var_name].dtype),
        'non_null_count': df[var_name].notna().sum(),
        'unique_values': df[var_name].nunique(),
    }
    
    # Añadir value labels si existen
    if var_name in metadata['variable_value_labels']:
        info['value_labels'] = metadata['variable_value_labels'][var_name]
    else:
        info['value_labels'] = {}
    
    return info
