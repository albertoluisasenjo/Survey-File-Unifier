"""
Módulo para mapeo manual de variables entre datasets
"""

import pandas as pd
from typing import Dict, List, Set
from collections import defaultdict


def create_variable_inventory(files_data: List[tuple]) -> Dict:
    """
    Crea un inventario de todas las variables en todos los archivos
    
    Args:
        files_data: Lista de tuplas (nombre_archivo, dataframe, metadata)
        
    Returns:
        Diccionario con inventario de variables
    """
    inventory = {}
    
    for file_name, df, metadata in files_data:
        for var_name in df.columns:
            if var_name not in inventory:
                inventory[var_name] = {
                    'label': metadata['column_labels'].get(var_name, ''),
                    'datasets': [],
                    'value_labels': {}
                }
            
            inventory[var_name]['datasets'].append(file_name)
            
            # Añadir value labels si existen
            if var_name in metadata['variable_value_labels']:
                inventory[var_name]['value_labels'][file_name] = metadata['variable_value_labels'][var_name]
    
    return inventory


def suggest_variable_mappings(inventory: Dict) -> Dict[str, List[str]]:
    """
    Sugiere agrupaciones de variables basándose en similitud de nombres y labels
    
    Args:
        inventory: Inventario de variables
        
    Returns:
        Diccionario con sugerencias: {nombre_unificado: [var1, var2, ...]}
    """
    suggestions = defaultdict(list)
    processed = set()
    
    # Ordenar variables por nombre
    sorted_vars = sorted(inventory.keys())
    
    for var in sorted_vars:
        if var in processed:
            continue
        
        # Buscar variables similares
        similar_vars = [var]
        var_lower = var.lower()
        var_label = inventory[var]['label'].lower()
        
        for other_var in sorted_vars:
            if other_var == var or other_var in processed:
                continue
            
            other_lower = other_var.lower()
            other_label = inventory[other_var]['label'].lower()
            
            # Criterios de similitud
            name_similar = (
                var_lower in other_lower or 
                other_lower in var_lower or
                var_lower.replace('_', '') == other_lower.replace('_', '')
            )
            
            label_similar = (
                var_label and other_label and
                (var_label in other_label or other_label in var_label)
            )
            
            if name_similar or label_similar:
                similar_vars.append(other_var)
                processed.add(other_var)
        
        if len(similar_vars) > 1:
            # Usar el nombre más común o el label como nombre unificado
            if inventory[var]['label']:
                unified_name = inventory[var]['label']
            else:
                unified_name = var
            
            suggestions[unified_name] = similar_vars
        
        processed.add(var)
    
    return dict(suggestions)


def create_mapping_dataframe(inventory: Dict, suggestions: Dict = None) -> pd.DataFrame:
    """
    Crea un DataFrame para visualizar y editar el mapeo de variables
    
    Args:
        inventory: Inventario de variables
        suggestions: Sugerencias de mapeo (opcional)
        
    Returns:
        DataFrame con columnas: variable_original, label, datasets, nombre_unificado
    """
    rows = []
    
    # Si hay sugerencias, usarlas
    if suggestions:
        for unified_name, vars_list in suggestions.items():
            for var in vars_list:
                rows.append({
                    'variable_original': var,
                    'label': inventory[var]['label'],
                    'datasets': ', '.join(inventory[var]['datasets']),
                    'nombre_unificado': unified_name
                })
    else:
        # Sin sugerencias, lista simple
        for var, info in inventory.items():
            rows.append({
                'variable_original': var,
                'label': info['label'],
                'datasets': ', '.join(info['datasets']),
                'nombre_unificado': var  # Por defecto, mismo nombre
            })
    
    df = pd.DataFrame(rows)
    return df.sort_values('variable_original').reset_index(drop=True)


def apply_variable_mapping(
    files_data: List[tuple],
    mapping_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aplica el mapeo de variables y unifica los datasets
    
    Args:
        files_data: Lista de tuplas (nombre_archivo, dataframe, metadata)
        mapping_df: DataFrame con el mapeo de variables
        
    Returns:
        DataFrame unificado con variables renombradas
    """
    # Crear diccionario de mapeo
    rename_dict = {}
    for _, row in mapping_df.iterrows():
        original = row['variable_original']
        unified = row['nombre_unificado']
        if pd.notna(unified) and unified.strip():
            rename_dict[original] = unified
    
    # Procesar cada archivo
    unified_dfs = []
    
    for file_name, df, metadata in files_data:
        df_copy = df.copy()
        
        # Renombrar columnas según mapeo
        df_copy = df_copy.rename(columns=rename_dict)
        
        # Añadir columna de origen
        df_copy['dataset_origen'] = file_name
        
        unified_dfs.append(df_copy)
    
    # Concatenar todos los dataframes
    result = pd.concat(unified_dfs, axis=0, ignore_index=True, sort=False)
    
    return result


def get_mapping_stats(mapping_df: pd.DataFrame) -> Dict:
    """
    Calcula estadísticas sobre el mapeo de variables
    
    Args:
        mapping_df: DataFrame con el mapeo
        
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        'total_variables': len(mapping_df),
        'variables_unificadas': mapping_df['nombre_unificado'].nunique(),
        'variables_agrupadas': len(mapping_df[mapping_df.duplicated('nombre_unificado', keep=False)]),
        'variables_sin_cambios': len(mapping_df[mapping_df['variable_original'] == mapping_df['nombre_unificado']])
    }
    
    return stats
