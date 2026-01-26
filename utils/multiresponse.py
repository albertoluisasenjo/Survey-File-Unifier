"""
Módulo para detección y transformación de variables multirespuesta
"""

import pandas as pd
import re
from typing import List, Dict, Tuple
import os


def detect_multiresponse_variables(df: pd.DataFrame, metadata: dict) -> List[List[str]]:
    """
    Detecta grupos de variables multirespuesta
    
    Criterios:
    - Variables consecutivas con mismo prefijo
    - Valores binarios (1.0, 0.0 o 2.0)
    - Mínimo 3 variables por grupo
    
    Args:
        df: DataFrame con los datos
        metadata: Metadata del archivo
        
    Returns:
        Lista de grupos, cada grupo es una lista de nombres de variables
    """
    
    def is_multiresponse_candidate(var_name: str) -> bool:
        """Verifica si una variable es candidata a multirespuesta por valores"""
        if var_name not in metadata['variable_value_labels']:
            return False
        
        var_codes = metadata['variable_value_labels'][var_name]
        
        # Filtrar códigos 98.0 y 99.0 (NS/NC)
        filtered_keys = set(k for k in var_codes.keys() if k not in [98.0, 99.0])
        
        if not filtered_keys:
            return False
        
        # Criterios de multirespuesta
        return (
            filtered_keys == {1.0} or
            filtered_keys == {1.0, 0.0} or
            filtered_keys == {1.0, 2.0}
        )
    
    def extract_base_name(var_name: str) -> str:
        """Extrae el nombre base de una variable (sin sufijo numérico)"""
        match = re.match(r'^(.+?)_?(\d+[A-Z]*)$', var_name)
        if match:
            return match.group(1)
        return var_name
    
    # Obtener todas las variables
    all_var_names = list(metadata['column_names'])
    multiresponse_groups = []
    
    i = 0
    while i < len(all_var_names):
        var_name = all_var_names[i]
        
        # Verificar si cumple criterios
        if not is_multiresponse_candidate(var_name):
            i += 1
            continue
        
        # Iniciar grupo potencial
        current_group = [var_name]
        base_name = extract_base_name(var_name)
        consecutive_non_matches = 0
        
        j = i + 1
        while j < len(all_var_names):
            next_var = all_var_names[j]
            next_base = extract_base_name(next_var)
            
            # Verificar si cumple criterios y tiene mismo base_name
            if (next_base == base_name and 
                is_multiresponse_candidate(next_var)):
                current_group.append(next_var)
                consecutive_non_matches = 0
                j += 1
                continue
            
            # No cumple criterios
            consecutive_non_matches += 1
            
            if consecutive_non_matches >= 3:
                break
            
            j += 1
        
        # Si encontramos grupo válido (≥3 variables), guardarlo
        if len(current_group) >= 3:
            multiresponse_groups.append(current_group)
            i = j  # Saltar las variables ya procesadas
        else:
            i += 1
    
    return multiresponse_groups


def extract_differential_part(labels: List[str]) -> Dict[str, str]:
    """
    Extrae la parte diferencial de cada label
    
    Ejemplo:
    Input: ["Partido: VOX", "Partido: PSOE", "Partido: PP"]
    Output: {"Partido: VOX": "VOX", "Partido: PSOE": "PSOE", ...}
    """
    if not labels:
        return {}
    
    normalized = [' '.join(label.split()) for label in labels]
    differential_parts = {}
    
    for label in normalized:
        # Buscar patrones comunes
        
        # 1. Entre corchetes: [VOX]
        match = re.search(r'\[([^\]]+)\]', label)
        if match:
            differential_parts[label] = match.group(1).strip()
            continue
        
        # 2. Entre paréntesis (último): (VOX)
        match = re.search(r'\(([^)]+)\)(?!.*\()', label)
        if match:
            content = match.group(1).strip()
            # Evitar paréntesis aclaratorios
            if content.upper() not in ['ESPONTÁNEA', 'MULTIRRESPUESTA', 'NO LEER']:
                differential_parts[label] = content
                continue
        
        # 3. Después de dos puntos o guión: "Opción: VOX"
        match = re.search(r'[:\-]\s*([^:\-]+)$', label)
        if match:
            differential_parts[label] = match.group(1).strip()
            continue
        
        # 4. Última palabra (fallback)
        words = label.split()
        if words:
            differential_parts[label] = words[-1].strip('.,;:!?')
    
    # Si no funcionó, usar prefijo común
    if len(differential_parts) < len(normalized):
        common_prefix = os.path.commonprefix(normalized)
        
        for label in normalized:
            if label not in differential_parts:
                diff = label[len(common_prefix):].strip()
                diff = diff.lstrip('.,;:!?-–—[] ')
                differential_parts[label] = diff if diff else label
    
    return differential_parts


def transform_multiresponse_to_list(
    df: pd.DataFrame, 
    group: List[str], 
    metadata: dict
) -> Tuple[pd.Series, str, Dict]:
    """
    Transforma un grupo de variables multirespuesta en una columna con listas
    
    Args:
        df: DataFrame con los datos
        group: Lista de nombres de variables del grupo
        metadata: Metadata del archivo
        
    Returns:
        (nueva_columna, nombre_columna, metadata_transformacion)
    """
    
    # Obtener labels de cada variable
    labels = []
    label_to_var = {}
    
    for var in group:
        label = metadata['column_labels'].get(var, var)
        labels.append(label)
        label_to_var[label] = var
    
    # Extraer partes diferenciales
    differential_parts = extract_differential_part(labels)
    
    # Crear mapeo: variable → valor diferencial
    var_to_differential = {}
    for label, diff in differential_parts.items():
        var = label_to_var[label]
        # Limpiar el valor
        cleaned_diff = diff.strip('[](){}')
        cleaned_diff = re.sub(r'^\d+[.)]\s*', '', cleaned_diff)
        var_to_differential[var] = cleaned_diff
    
    # Crear nueva columna tipo lista
    def create_list_for_row(row):
        selected = []
        for var in group:
            if pd.notna(row[var]):
                value = row[var]
                # Si tiene valor 1.0, está seleccionada
                if value == 1.0 or value == True or value == 1:
                    diff_value = var_to_differential.get(var, var)
                    selected.append(diff_value)
        
        return selected if selected else None
    
    # Aplicar transformación
    new_column = df[group].apply(create_list_for_row, axis=1)
    
    # Nombre de la nueva columna
    if labels:
        first_label = labels[0]
        # Remover la parte diferencial
        for diff in differential_parts.values():
            first_label = first_label.replace(f'[{diff}]', '').replace(f'({diff})', '')
        
        base_question = first_label.strip().strip('.,;:!?-–— ')
        new_column_name = f"{base_question}_multirespuesta"
    else:
        new_column_name = f"{group[0].split('_')[0]}_multirespuesta"
    
    # Metadata de la transformación
    transformation_metadata = {
        'column_name': new_column_name,
        'original_columns': group,
        'value_mappings': var_to_differential
    }
    
    return new_column, new_column_name, transformation_metadata


def apply_multiresponse_transformations(
    df: pd.DataFrame,
    metadata: dict,
    groups: List[List[str]]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Aplica transformaciones de multirespuesta a todos los grupos
    
    Args:
        df: DataFrame original
        metadata: Metadata del archivo
        groups: Lista de grupos de variables multirespuesta
        
    Returns:
        (df_transformado, lista_de_metadatas)
    """
    df_result = df.copy()
    transformations = []
    
    for group in groups:
        # Transformar grupo
        new_column, new_name, trans_meta = transform_multiresponse_to_list(
            df_result, group, metadata
        )
        
        # Añadir nueva columna
        df_result[new_name] = new_column
        
        # Eliminar columnas originales
        df_result = df_result.drop(columns=group)
        
        # Guardar metadata
        transformations.append(trans_meta)
    
    return df_result, transformations
