import streamlit as st
import pandas as pd
import pyreadstat
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import tempfile
import io
import os
import numpy as np
from openai import OpenAI

# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

st.set_page_config(
    page_title="Unificador de Encuestas Longitudinales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INICIALIZAR SESSION STATE
# ============================================================================

if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.uploaded_files = []
    st.session_state.importer = None
    st.session_state.metadata_collection = {}
    st.session_state.variable_inventory = {}
    st.session_state.variable_mapping = None
    st.session_state.variable_mapping_edited = None
    st.session_state.variable_mapping_validated = None
    st.session_state.unified_dataset = None
    st.session_state.unifier = None
    st.session_state.discrepancies = {}
    st.session_state.value_mapping = None
    st.session_state.value_mapping_edited = None
    st.session_state.final_dataset = None
    st.session_state.api_key = st.secrets.get("openai_key", "")
    st.session_state.nulls_report_pre = None
    st.session_state.nulls_report_post = None

# ============================================================================
# CONSTANTES
# ============================================================================

MONTHS_ES = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'abril1': '04', 'abril2': '04',
    'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
    'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
}

METADATA_COLS = ['dataset_origen', 'fecha', 'tipo_encuesta']

# ============================================================================
# CARGA DE PROMPTS DESDE ARCHIVOS EXTERNOS
# ============================================================================

def load_prompt_from_file(filepath: str, default: str = "") -> str:
    """Carga un prompt desde archivo externo o usa el default"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception:
        pass
    return default

# Cargar prompts (si existen los archivos)
VARIABLE_MAPPING_SYSTEM_PROMPT = load_prompt_from_file(
    "prompts/variable_mapping_system.txt",
    "Eres un experto en análisis de encuestas. Agrupa variables que midan lo mismo."
)

VARIABLE_MAPPING_USER_TEMPLATE = load_prompt_from_file(
    "prompts/variable_mapping_user.txt",
    "Analiza este inventario y genera VARIABLE_MAPPING: {inventory}"
)

VALUE_MAPPING_SYSTEM_PROMPT = load_prompt_from_file(
    "prompts/value_mapping_system.txt",
    "Eres un experto en normalización de datos. Unifica valores discrepantes."
)

VALUE_MAPPING_USER_TEMPLATE = load_prompt_from_file(
    "prompts/value_mapping_user.txt",
    "Analiza estas discrepancias y genera VALUE_MAPPING: {discrepancies}"
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def extract_date_from_filename(filename: str) -> str:
    """Extrae fecha del nombre del archivo"""
    match = re.search(r"(\b\w+)\D*(\d{4})", filename.lower())
    if match:
        month_str, year = match.groups()
        month_str = month_str.replace("barometro_andaluz_", "").replace("_", "")
        month = MONTHS_ES.get(month_str, '01')
        return f"{year}-{month}"
    return "1900-01"

def parse_dataset_name(dataset_name: str) -> dict:
    """Parsea el nombre del dataset para extraer tipo, mes y año"""
    parts = dataset_name.split('_')
    tipo_raw = parts[0]
    tipo = tipo_raw[:-1] if tipo_raw.endswith('s') else tipo_raw
    
    mes = None
    año = None
    
    for part in parts[1:]:
        part_lower = part.lower()
        for mes_nombre, mes_num in MONTHS_ES.items():
            if mes_nombre in part_lower:
                mes = int(mes_num)
                break
        if part.isdigit() and len(part) == 4:
            año = int(part)
    
    if mes is None:
        mes = 1
    if año is None:
        año = 2020
    
    return {'tipo': tipo, 'mes': mes, 'año': año}

def create_timestamp(año: int, mes: int, dia: int = 15) -> pd.Timestamp:
    """Crea un timestamp para el día 15 del mes/año indicado"""
    return pd.Timestamp(year=año, month=mes, day=dia)

def normalize_label_aggressive(text: str) -> str:
    """Normalización agresiva de labels para comparación"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'^\d+\.\s*', '', text)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = unicodedata.normalize('NFC', text)
    
    tilde_map = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ñ': 'n', 'ç': 'c'
    }
    
    for char_with_tilde, char_without in tilde_map.items():
        text = text.replace(char_with_tilde, char_without)
    
    symbols_to_replace = ['/', '-', '–', '—', ':', ';', ',', '.', '?', '¿', '!', '¡', 
                          '(', ')', '[', ']', '{', '}', '_', '"', "'", '«', '»']
    for symbol in symbols_to_replace:
        text = text.replace(symbol, ' ')
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcula similitud entre dos textos normalizados"""
    return SequenceMatcher(None, text1, text2).ratio()

def should_group_by_similarity(label1: str, label2: str, threshold: float = 0.985) -> Tuple[bool, Dict]:
    """Determina si dos labels deben agruparse por similitud"""
    norm1 = normalize_label_aggressive(label1)
    norm2 = normalize_label_aggressive(label2)
    
    if norm1 == norm2:
        return True, {
            'similarity': 1.0,
            'reason': 'identical_after_normalization',
            'normalized': norm1
        }
    
    similarity = calculate_similarity(norm1, norm2)
    should_group = similarity >= threshold
    
    return should_group, {
        'similarity': similarity,
        'threshold': threshold,
        'threshold_met': should_group,
        'norm1': norm1[:80],
        'norm2': norm2[:80]
    }

# ============================================================================
# CLASE: DETECTOR DE MULTIRESPUESTA (COMPLETO)
# ============================================================================

def detect_multiresponse_variables(df: pd.DataFrame, meta) -> List[List[str]]:
    """
    Detecta grupos de variables multirespuesta usando MÉTODO HÍBRIDO:
    1. Criterio inicial: base_name + códigos correctos
    2. Extensión flexible: si base_name cambia, verificar si label comparte prefijo/sufijo común
    """
    
    def is_multiresponse_candidate_by_values(var_codes: Dict) -> bool:
        """Verifica si una variable cumple criterios de multirespuesta por valores"""
        if not var_codes:
            return False
        
        filtered_keys = set(k for k in var_codes.keys() if k not in [98.0, 99.0])
        
        if not filtered_keys:
            return False
        
        if filtered_keys == {1.0}:
            return True
        elif filtered_keys == {1.0, 0.0}:
            return True
        elif filtered_keys == {1.0, 2.0}:
            return True
        
        return False
    
    def extract_base_name(var_name: str) -> str:
        """Extrae el nombre base de una variable (sin sufijo numérico)"""
        match = re.match(r'^(.+?)_?(\d+[A-Z]*)$', var_name)
        if match:
            return match.group(1)
        return var_name
    
    def find_common_prefix_suffix(labels: List[str]) -> tuple:
        """Encuentra el prefijo y sufijo común de un conjunto de labels"""
        if not labels or len(labels) < 2:
            return ("", "")
        
        normalized = [' '.join(label.split()) for label in labels]
        
        # Prefijo común
        prefix = os.path.commonprefix(normalized)
        
        # Sufijo común
        suffix = normalized[0]
        for label in normalized[1:]:
            min_len = min(len(suffix), len(label))
            i = 1
            while i <= min_len and suffix[-i] == label[-i]:
                i += 1
            suffix = suffix[-(i-1):] if i > 1 else ""
        
        # Evitar solapamiento
        if prefix and suffix:
            min_len = len(min(normalized, key=len))
            if len(prefix) + len(suffix) >= min_len:
                if len(prefix) > len(suffix):
                    suffix = ""
                else:
                    prefix = ""
        
        prefix = prefix.strip()
        suffix = suffix.strip()
        
        return (prefix, suffix)
    
    def label_matches_pattern(label: str, prefix: str, suffix: str, min_prefix_len: int = 10) -> bool:
        """Verifica si un label cumple con el patrón de prefijo/sufijo"""
        if not prefix and not suffix:
            return False
        
        if prefix and len(prefix) < min_prefix_len:
            return False
        
        label_normalized = ' '.join(label.split())
        
        if prefix and suffix:
            return label_normalized.startswith(prefix) and label_normalized.endswith(suffix)
        elif prefix:
            return label_normalized.startswith(prefix)
        elif suffix:
            return label_normalized.endswith(suffix)
        
        return False
    
    # Obtener todas las variables
    all_var_names = list(meta.column_names)
    multiresponse_groups = []
    
    i = 0
    while i < len(all_var_names):
        var_name = all_var_names[i]
        
        if var_name not in meta.variable_value_labels:
            i += 1
            continue
        
        var_codes = meta.variable_value_labels[var_name]
        if not is_multiresponse_candidate_by_values(var_codes):
            i += 1
            continue
        
        # Iniciar grupo potencial
        current_group = [var_name]
        base_name = extract_base_name(var_name)
        consecutive_non_matches = 0
        
        # Obtener labels del grupo para análisis de patrón
        group_labels = [meta.column_names_to_labels.get(var_name, var_name)]
        label_prefix = ""
        label_suffix = ""
        
        j = i + 1
        while j < len(all_var_names):
            next_var = all_var_names[j]
            next_base = extract_base_name(next_var)
            
            # Verificar códigos
            if next_var in meta.variable_value_labels:
                next_codes = meta.variable_value_labels[next_var]
                meets_value_criteria = is_multiresponse_candidate_by_values(next_codes)
            else:
                meets_value_criteria = False
            
            # CRITERIO 1: Mismo base_name + códigos correctos
            if base_name == next_base and meets_value_criteria:
                current_group.append(next_var)
                group_labels.append(meta.column_names_to_labels.get(next_var, next_var))
                consecutive_non_matches = 0
                
                # Actualizar patrón de labels si tenemos ≥3 variables
                if len(current_group) >= 3:
                    label_prefix, label_suffix = find_common_prefix_suffix(group_labels)
                
                j += 1
                continue
            
            # CRITERIO 2: Base_name diferente PERO label cumple patrón
            if meets_value_criteria and len(current_group) >= 3:
                next_label = meta.column_names_to_labels.get(next_var, next_var)
                
                if label_matches_pattern(next_label, label_prefix, label_suffix):
                    current_group.append(next_var)
                    group_labels.append(next_label)
                    consecutive_non_matches = 0
                    
                    label_prefix, label_suffix = find_common_prefix_suffix(group_labels)
                    
                    j += 1
                    continue
            
            # No cumple ningún criterio
            consecutive_non_matches += 1
            
            if consecutive_non_matches >= 3:
                break
            
            j += 1
        
        # Si encontramos grupo válido (≥3 variables), guardarlo
        if len(current_group) >= 3:
            multiresponse_groups.append(current_group)
            i = j
        else:
            i += 1
    
    return multiresponse_groups

def transform_multiresponse_to_list(df: pd.DataFrame, group: List[str], meta) -> tuple:
    """
    Transforma un grupo de variables multirespuesta en una sola columna tipo lista
    El mapeo de valores usa la PARTE DIFERENCIAL del label de cada variable
    """
    
    def extract_differential_part(labels: List[str]) -> Dict[str, str]:
        """Extrae la parte diferencial de cada label"""
        if not labels:
            return {}
        
        normalized = [' '.join(label.split()) for label in labels]
        differential_parts = {}
        
        for label in normalized:
            # 1. Entre corchetes: [VOX], [PSOE]
            match = re.search(r'\[([^\]]+)\]', label)
            if match:
                differential_parts[label] = match.group(1).strip()
                continue
            
            # 2. Entre paréntesis: (VOX), (PSOE)
            match = re.search(r'\(([^)]+)\)(?!.*\()', label)
            if match:
                content = match.group(1).strip()
                if content.upper() not in ['ESPONTÁNEA', 'MULTIRRESPUESTA', 'RESPUESTA ESPONTÁNEA',
                                            'NO LEER', 'LEER', 'MARCAR', 'ÚNICA']:
                    differential_parts[label] = content
                    continue
            
            # 3. Después de dos puntos o guión
            match = re.search(r'[:\-]\s*([^:\-]+)$', label)
            if match:
                differential_parts[label] = match.group(1).strip()
                continue
            
            # 4. Última palabra/frase significativa (fallback)
            words = label.split()
            if words:
                differential_parts[label] = words[-1].strip('.,;:!?')
        
        # Si el método 1 no funcionó para todos, usar método 2: prefijo común
        if len(differential_parts) < len(normalized):
            if len(normalized) > 1:
                common_prefix = normalized[0]
                for label in normalized[1:]:
                    min_len = min(len(common_prefix), len(label))
                    i = 0
                    while i < min_len and common_prefix[i] == label[i]:
                        i += 1
                    common_prefix = common_prefix[:i]
                
                for label in normalized:
                    if label not in differential_parts:
                        diff = label[len(common_prefix):].strip()
                        diff = diff.lstrip('.,;:!?-–—[] ')
                        differential_parts[label] = diff if diff else label
        
        return differential_parts
    
    def clean_differential_value(value: str) -> str:
        """Limpia el valor diferencial extraído"""
        value = value.strip('[](){}')
        value = re.sub(r'^\d+[.)]\s*', '', value)
        
        if value.isupper() and len(value) > 3:
            value = value.capitalize()
        
        return value.strip()
    
    # Obtener labels de cada variable del grupo
    labels = []
    label_to_var = {}
    
    for var in group:
        label = meta.column_names_to_labels.get(var, var)
        labels.append(label)
        label_to_var[label] = var
    
    # Extraer partes diferenciales
    differential_parts = extract_differential_part(labels)
    
    # Crear mapeo: variable → valor diferencial
    var_to_differential = {}
    for label, diff in differential_parts.items():
        var = label_to_var[label]
        cleaned_diff = clean_differential_value(diff)
        var_to_differential[var] = cleaned_diff
    
    # Crear nueva columna tipo lista
    def create_list_for_row(row):
        selected = []
        for var in group:
            if pd.notna(row[var]):
                value = row[var]
                if value == 1.0 or value == True or value == 1:
                    diff_value = var_to_differential.get(var, var)
                    selected.append(diff_value)
        
        return selected if selected else None
    
    # Aplicar transformación
    new_column = df[group].apply(create_list_for_row, axis=1)
    
    # Nombre de la nueva columna
    if labels:
        first_label = labels[0]
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
        'transformation_type': 'multiresponse_to_list',
        'value_mappings': {var: var_to_differential.get(var, var) for var in group}
    }
    
    return new_column, new_column_name, transformation_metadata

def verify_multiresponse_transformation(df: pd.DataFrame, 
                                        original_columns: List[str],
                                        new_column_name: str,
                                        value_mappings: Dict[str, str],
                                        n_samples: int = 3) -> Dict:
    """
    Verifica que la transformación multirespuesta se hizo correctamente
    Retorna dict con ejemplos de transformación
    """
    verification = {
        'new_column': new_column_name,
        'original_count': len(original_columns),
        'value_mappings': value_mappings,
        'examples': []
    }
    
    # Mostrar ejemplos de filas
    sample_rows = df[df[new_column_name].notna()].head(n_samples)
    
    for idx, row in sample_rows.iterrows():
        example = {
            'row_index': int(idx),
            'original_values': {},
            'transformed_value': row[new_column_name]
        }
        
        for col in original_columns:
            if col in df.columns:
                val = row[col]
                if pd.notna(val) and val in [1.0, 1, True]:
                    example['original_values'][col] = val
        
        verification['examples'].append(example)
    
    return verification

# ============================================================================
# CLASE: SAVImporter (COMPLETO)
# ============================================================================

@dataclass
class FileSource:
    name: str
    source_type: str
    path: str
    date: Optional[str] = None

class SAVImporter:
    """Importa y procesa archivos SAV"""
    
    def __init__(self):
        self.datasets = {}
        self.metadata_collection = {}
        self.multiresponse_transformations = {}
        self.multiresponse_verifications = {}
    
    def read_sav_file(self, file_bytes, filename: str) -> Tuple[pd.DataFrame, any]:
        """Lee un archivo SAV desde bytes"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        df, meta = pyreadstat.read_sav(tmp_path)
        Path(tmp_path).unlink()
        return df, meta
    
    def process_multiresponse(self, df: pd.DataFrame, meta: any, dataset_id: str):
        """Detecta y transforma variables multirespuesta"""
        multiresponse_groups = detect_multiresponse_variables(df, meta)
        
        if not multiresponse_groups:
            return df, meta
        
        transformations = []
        verifications = []
        
        for group in multiresponse_groups:
            new_column, new_column_name, transformation_meta = transform_multiresponse_to_list(df, group, meta)
            df[new_column_name] = new_column
            
            # Verificar transformación
            verification = verify_multiresponse_transformation(
                df, group, new_column_name, 
                transformation_meta['value_mappings']
            )
            verifications.append(verification)
            
            df = df.drop(columns=group)
            transformations.append(transformation_meta)
        
        self.multiresponse_transformations[dataset_id] = transformations
        self.multiresponse_verifications[dataset_id] = verifications
        
        return df, meta
    
    def extract_metadata(self, df: pd.DataFrame, meta: any, dataset_name: str) -> Dict:
        """Extrae toda la metadata relevante"""
        multiresponse_labels = {}
        if dataset_name in self.multiresponse_transformations:
            for transformation in self.multiresponse_transformations[dataset_name]:
                col_name = transformation['column_name']
                col_label = transformation['column_name']
                cleaned_label = col_label.strip().rstrip(':').strip()
                multiresponse_labels[col_name] = cleaned_label
        
        column_names_to_labels = {}
        for col in df.columns:
            if col in meta.column_names_to_labels:
                column_names_to_labels[col] = meta.column_names_to_labels[col]
            elif col in multiresponse_labels:
                column_names_to_labels[col] = multiresponse_labels[col]
            else:
                column_names_to_labels[col] = col
        
        metadata = {
            'dataset_name': dataset_name,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'column_names': list(df.columns),
            'column_labels': [column_names_to_labels.get(col, col) for col in df.columns],
            'column_names_to_labels': column_names_to_labels,
            'variable_value_labels': {
                col: meta.variable_value_labels.get(col, {}) 
                for col in df.columns 
                if col in meta.variable_value_labels
            }
        }
        
        return metadata
    
    def import_file(self, file_bytes, filename: str):
        """Importa un archivo SAV"""
        df, meta = self.read_sav_file(file_bytes, filename)
        
        dataset_id = filename.replace('.sav', '').replace('.SAV', '')
        date = extract_date_from_filename(filename)
        
        df, meta = self.process_multiresponse(df, meta, dataset_id)
        
        self.datasets[dataset_id] = {
            'dataframe': df,
            'metadata': meta,
            'file_info': {'name': filename, 'date': date}
        }
        
        self.metadata_collection[dataset_id] = self.extract_metadata(df, meta, dataset_id)

# ============================================================================
# CLASE: VariableInventoryGenerator (COMPLETO)
# ============================================================================

class VariableInventoryGenerator:
    """Genera inventario de variables para análisis con IA"""
    
    def __init__(self, metadata_collection: Dict, multiresponse_transformations: Dict):
        self.metadata_collection = metadata_collection
        self.multiresponse_transformations = multiresponse_transformations
        self.variable_inventory = {}
    
    def generate_variable_inventory(self) -> Dict:
        """Genera inventario de variables INCLUYENDO multirespuesta"""
        label_to_codes = defaultdict(lambda: defaultdict(list))
        
        for dataset_id, meta in self.metadata_collection.items():
            dataset_name = meta['dataset_name']
            
            for var_code, var_label in meta['column_names_to_labels'].items():
                if var_label is None or var_label.strip() == '':
                    var_label = f"[VAR_SIN_ETIQUETA: {var_code}]"
                
                is_multiresponse = (
                    var_code.endswith('_multirespuesta') or 
                    var_code.endswith('_')
                )
                
                if dataset_id in self.multiresponse_transformations:
                    for transformation in self.multiresponse_transformations[dataset_id]:
                        if transformation['column_name'] == var_code:
                            is_multiresponse = True
                            break
                
                if is_multiresponse:
                    var_label = f"{var_label} [MULTIRESPUESTA]"
                
                normalized_label = ' '.join(var_label.split())
                label_to_codes[normalized_label][var_code].append(dataset_name)
        
        self.variable_inventory = {
            label: dict(codes) 
            for label, codes in sorted(label_to_codes.items())
        }
        
        return self.variable_inventory
    
    def _get_value_labels_preview(self, var_code: str, dataset_name: str) -> str:
        """Obtiene vista previa de value_labels"""
        if dataset_name not in self.metadata_collection:
            return "Dataset no encontrado"
        
        meta = self.metadata_collection[dataset_name]
        
        if var_code not in meta['variable_value_labels']:
            return "Variable numérica sin labels"
        
        value_labels = meta['variable_value_labels'][var_code]
        
        if not value_labels:
            return "Variable numérica sin labels"
        
        preview_parts = []
        
        for code_value in [1.0, 2.0]:
            if code_value in value_labels:
                label_text = value_labels[code_value]
                preview_parts.append(f"{int(code_value)}='{label_text}'")
        
        if not preview_parts:
            sorted_keys = sorted(value_labels.keys())[:2]
            for key in sorted_keys:
                label_text = value_labels[key]
                if isinstance(key, float) and key.is_integer():
                    key_str = str(int(key))
                else:
                    key_str = str(key)
                preview_parts.append(f"{key_str}='{label_text}'")
        
        if not preview_parts:
            return "Variable numérica sin labels"
        
        return " | ".join(preview_parts)
    
    def generate_summary_for_ai(self) -> str:
        """Genera resumen compacto para IA"""
        lines = [
            "# INVENTARIO DE VARIABLES - RESUMEN PARA ANÁLISIS",
            "# Formato: [Label] → {Código: [Datasets]} | Value Labels Preview",
            "# IMPORTANTE - Variables Multirespuesta:",
            "# Las variables marcadas con [MULTIRESPUESTA] contienen LISTAS de valores",
            "# IMPORTANTE - Value Labels Preview:",
            "# Se muestran los valores correspondientes a códigos 1 y 2 (cuando existen)",
            ""
        ]
        
        for label, codes in sorted(self.variable_inventory.items()):
            lines.append(f'"{label}"')
            
            for code, datasets in codes.items():
                first_dataset = datasets[0] if datasets else None
                
                if first_dataset:
                    value_preview = self._get_value_labels_preview(code, first_dataset)
                else:
                    value_preview = "Sin datasets"
                
                datasets_str = ', '.join(datasets)
                lines.append(f"   → {code}: [{datasets_str}] → {value_preview}")
            
            lines.append("")
        
        return "\n".join(lines)

# ============================================================================
# CLASE: VariableMappingGenerator (con OpenAI)
# ============================================================================

class VariableMappingGenerator:
    """Genera VARIABLE_MAPPING usando IA"""
    
    def __init__(self, api_key: str, inventory_summary: str):
        self.client = OpenAI(api_key=api_key)
        self.inventory_summary = inventory_summary
        self.variable_mapping = None
        self.raw_response = None
    
    def generate_mapping(self, model: str = "gpt-5.2") -> str:
        """Genera mapping con OpenAI"""
        
        # Usar prompts desde archivos o defaults
        system_prompt = VARIABLE_MAPPING_SYSTEM_PROMPT
        user_prompt = VARIABLE_MAPPING_USER_TEMPLATE.format(inventory=self.inventory_summary)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        self.raw_response = response.choices[0].message.content
        return self.raw_response
    
    def extract_python_code(self) -> str:
        """Extrae código Python de la respuesta"""
        if not self.raw_response:
            return ""
        
        code_blocks = re.findall(r'```python\n(.*?)\n```', self.raw_response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        if "VARIABLE_MAPPING" in self.raw_response:
            start_idx = self.raw_response.find("VARIABLE_MAPPING")
            return self.raw_response[start_idx:]
        
        return self.raw_response

# ============================================================================
# FUNCIÓN: Validación de VARIABLE_MAPPING
# ============================================================================

def validate_variable_mapping(variable_mapping: Dict, metadata_collection: Dict) -> Dict:
    """
    Valida VARIABLE_MAPPING y detecta problemas
    
    Returns:
        Dict con problemas encontrados
    """
    problems_found = []
    total_entries = len(variable_mapping)
    
    for unified_name, codes_dict in variable_mapping.items():
        # Contar códigos (items)
        total_codes = len(codes_dict)
        
        # Contar datasets únicos
        all_datasets = []
        for code, datasets_list in codes_dict.items():
            if isinstance(datasets_list, str):
                all_datasets.append(datasets_list)
            elif isinstance(datasets_list, (list, tuple)):
                all_datasets.extend(datasets_list)
        
        unique_datasets = set(all_datasets)
        num_unique_datasets = len(unique_datasets)
        
        # Detectar problema: más códigos que datasets únicos
        if total_codes > num_unique_datasets:
            # Identificar qué datasets están duplicados
            dataset_counts = Counter(all_datasets)
            duplicated_datasets = {ds: count for ds, count in dataset_counts.items() if count > 1}
            
            problems_found.append({
                'unified_name': unified_name,
                'total_codes': total_codes,
                'unique_datasets': num_unique_datasets,
                'duplicated_datasets': duplicated_datasets,
                'codes_dict': codes_dict
            })
    
    return {
        'total_entries': total_entries,
        'problems_count': len(problems_found),
        'problems_details': problems_found
    }

def annotate_variable_mapping_with_warnings(variable_mapping: Dict, validation_results: Dict) -> str:
    """
    Genera versión del VARIABLE_MAPPING con comentarios #REVISAR
    donde hay duplicados
    """
    problems_set = set(p['unified_name'] for p in validation_results['problems_details'])
    
    lines = ["VARIABLE_MAPPING = {"]
    
    for unified_name, codes_dict in variable_mapping.items():
        # Determinar si necesita advertencia
        needs_warning = unified_name in problems_set
        
        # Añadir entrada
        lines.append(f'    "{unified_name}": {{')
        
        for code, datasets in codes_dict.items():
            if isinstance(datasets, str):
                datasets_formatted = f'"{datasets}"'
            else:
                datasets_formatted = '[\n            ' + ',\n            '.join(f'"{ds}"' for ds in datasets) + '\n        ]'
            
            lines.append(f'        "{code}": {datasets_formatted},')
        
        # Añadir advertencia si es necesario
        if needs_warning:
            problem = next(p for p in validation_results['problems_details'] if p['unified_name'] == unified_name)
            lines.append(f'    }},  # REVISAR: {problem["total_codes"]} códigos, {problem["unique_datasets"]} datasets únicos')
        else:
            lines.append(f'    }},')
    
    lines.append("}")
    
    return '\n'.join(lines)

# ============================================================================
# FUNCIÓN: Auto-Mapping con Similitud
# ============================================================================

def auto_map_unmapped_variables(importer: SAVImporter, variable_mapping: Dict, 
                                similarity_threshold: float = 0.985) -> Dict:
    """
    Identifica variables que NO están en VARIABLE_MAPPING y las añade
    con agrupación por similitud
    """
    
    # Obtener variables ya mapeadas
    mapped_codes = set()
    for unified_name, codes_dict in variable_mapping.items():
        for code, datasets in codes_dict.items():
            mapped_codes.add(code)
    
    # Recopilar variables NO mapeadas
    unmapped_vars_raw = []
    
    for dataset_id, data in importer.datasets.items():
        df = data['dataframe']
        meta = importer.metadata_collection[dataset_id]
        
        for col in df.columns:
            if col not in mapped_codes:
                label = meta['column_names_to_labels'].get(col, None)
                has_real_label = (label is not None and label != col and label.strip() != '')
                
                unmapped_vars_raw.append({
                    'code': col,
                    'label': label if has_real_label else None,
                    'dataset': dataset_id,
                    'has_label': has_real_label
                })
    
    if len(unmapped_vars_raw) == 0:
        return variable_mapping
    
    # Agrupar por similitud de labels
    label_groups = {}
    vars_sin_label = []
    
    for var_info in unmapped_vars_raw:
        if not var_info['has_label']:
            vars_sin_label.append(var_info)
            continue
        
        label = var_info['label']
        
        # Buscar label similar existente
        matched = False
        
        for existing_label, group in label_groups.items():
            should_group, details = should_group_by_similarity(label, existing_label, similarity_threshold)
            
            if should_group:
                group.append(var_info)
                matched = True
                break
        
        if not matched:
            label_groups[label] = [var_info]
    
    # Añadir al VARIABLE_MAPPING
    added_count = 0
    
    # Grupos de labels
    for representative_label, group in label_groups.items():
        if representative_label not in variable_mapping:
            variable_mapping[representative_label] = {}
        
        for var_info in group:
            code = var_info['code']
            dataset = var_info['dataset']
            
            if code not in variable_mapping[representative_label]:
                variable_mapping[representative_label][code] = []
            
            variable_mapping[representative_label][code].append(dataset)
        
        added_count += 1
    
    # Variables sin label (independientes)
    for var_info in vars_sin_label:
        code = var_info['code']
        dataset = var_info['dataset']
        
        final_name = f"{code} [{dataset}]"
        
        original_name = final_name
        counter = 1
        while final_name in variable_mapping:
            counter += 1
            final_name = f"{original_name} (#{counter})"
        
        variable_mapping[final_name] = {code: [dataset]}
        added_count += 1
    
    return variable_mapping

# ============================================================================
# CLASE: DatasetUnifier (COMPLETO CON TODAS LAS FUNCIONES)
# ============================================================================

class DatasetUnifier:
    """Unifica múltiples datasets aplicando labels y mapeo de variables"""
    
    def __init__(self, importer: SAVImporter, variable_mapping: Dict):
        self.importer = importer
        self.variable_mapping = variable_mapping
        self.unified_df = None
        self.column_name_mapping = {}
        self.multiresponse_columns = set()
        self.nulls_report_pre = None
        self.nulls_report_post = None
    
    def _get_final_column_name(self, unified_name: str) -> str:
        """Obtiene el nombre final de la columna"""
        return unified_name
    
    def apply_value_labels_only(self, dataset_id: str) -> pd.DataFrame:
        """Aplica SOLO las etiquetas de VALORES (no cambia nombres de columnas)"""
        dataset_data = self.importer.datasets[dataset_id]
        df = dataset_data['dataframe'].copy()
        meta = self.importer.metadata_collection[dataset_id]
        
        for col in df.columns:
            if col in meta['variable_value_labels']:
                value_labels = meta['variable_value_labels'][col]
                df[col] = df[col].map(lambda x: value_labels.get(x, x) if pd.notna(x) else x)
        
        return df
    
    def _clean_no_contesta_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores 'No contesta' por NaN en todo el dataframe
        """
        no_contesta_patterns = [
            'No contesta',
            'NO CONTESTA',
            'No Contesta',
            'NC',
            'N.C.',
            'No sabe',
            'NO SABE',
            'No Sabe',
            'NS',
            'N.S.',
            'NS/NC',
            'N.S./N.C.'
        ]
        
        # Reemplazar en todas las columnas excepto dataset_origen
        for col in df.columns:
            if col != 'dataset_origen':
                df[col] = df[col].replace(no_contesta_patterns, pd.NA)
        
        return df
    
    def generate_nulls_report_before_unification(self) -> pd.DataFrame:
        """
        Genera reporte de NULLs ANTES de unificar
        """
        all_nulls_data = []
        
        for dataset_id in self.importer.datasets.keys():
            df = self.importer.datasets[dataset_id]['dataframe']
            
            for col in df.columns:
                null_count = df[col].isna().sum()
                null_percentage = (null_count / len(df)) * 100
                
                all_nulls_data.append({
                    'Dataset': dataset_id,
                    'Variable': col,
                    'Null_Count': null_count,
                    'Total_Rows': len(df),
                    'Null_Percentage': round(null_percentage, 2)
                })
        
        # Crear DataFrame
        df_nulls = pd.DataFrame(all_nulls_data)
        
        # Ordenar solo si hay datos
        if len(df_nulls) > 0:
            df_nulls = df_nulls.sort_values(['Dataset', 'Null_Count'], ascending=[True, False])
        
        self.nulls_report_pre = df_nulls
        
        return df_nulls
    
    def generate_nulls_report_post_unification(self) -> pd.DataFrame:
        """
        Genera reporte de NULLs DESPUÉS de unificar
        """
        nulls_data = []
        
        for col in self.unified_df.columns:
            if col in ['dataset_origen', 'tipo_encuesta', 'fecha']:
                continue
                
            null_count = self.unified_df[col].isna().sum()
            null_percentage = (null_count / len(self.unified_df)) * 100
            
            nulls_data.append({
                'Variable': col,
                'Null_Count': null_count,
                'Total_Rows': len(self.unified_df),
                'Null_Percentage': round(null_percentage, 2)
            })
        
        # Crear DataFrame y ordenar (de menos a más nulls)
        df_nulls = pd.DataFrame(nulls_data)
        df_nulls = df_nulls.sort_values('Null_Count')
        
        self.nulls_report_post = df_nulls
        
        return df_nulls
    
    def diagnose_variable_mapping(self) -> Dict:
        """
        Diagnostica problemas potenciales en VARIABLE_MAPPING
        ANTES de aplicarlo
        """
        issues = {
            'duplicates_in_same_dataset': [],
            'missing_codes': [],
            'total_issues': 0
        }
        
        # Verificar cada dataset
        for dataset_id in self.importer.datasets.keys():
            dataset_data = self.importer.datasets[dataset_id]
            available_codes = set(dataset_data['dataframe'].columns)
            
            # Códigos que se van a renombrar en este dataset
            final_names = {}
            
            for unified_name, codes_dict in self.variable_mapping.items():
                for code, datasets in codes_dict.items():
                    if dataset_id in datasets:
                        final_name = self._get_final_column_name(unified_name)
                        
                        # Verificar si el código existe
                        if code not in available_codes:
                            issues['missing_codes'].append({
                                'dataset': dataset_id,
                                'code': code,
                                'final_name': final_name
                            })
                            issues['total_issues'] += 1
                        
                        # Verificar duplicados
                        if final_name in final_names:
                            issues['duplicates_in_same_dataset'].append({
                                'dataset': dataset_id,
                                'final_name': final_name,
                                'code1': final_names[final_name],
                                'code2': code
                            })
                            issues['total_issues'] += 1
                        else:
                            final_names[final_name] = code
        
        return issues
    
    def unify_datasets(self) -> pd.DataFrame:
        """Unifica todos los datasets según VARIABLE_MAPPING"""
        
        # Generar reporte de nulls PRE-unificación
        self.generate_nulls_report_before_unification()
        
        unified_dfs = []
        
        for dataset_id in self.importer.datasets.keys():
            # PASO 1: Aplicar labels SOLO a valores
            df_with_value_labels = self.apply_value_labels_only(dataset_id)
            
            # PASO 2: Crear diccionario de renombrado
            rename_dict = {}
            
            for unified_name, codes_dict in self.variable_mapping.items():
                for code, datasets in codes_dict.items():
                    if dataset_id in datasets and code in df_with_value_labels.columns:
                        final_name = self._get_final_column_name(unified_name)
                        rename_dict[code] = final_name
                        
                        if unified_name not in self.column_name_mapping:
                            self.column_name_mapping[unified_name] = final_name
            
            # PASO 3: Detectar y resolver duplicados
            name_to_codes = {}
            for code, name in rename_dict.items():
                if name not in name_to_codes:
                    name_to_codes[name] = []
                name_to_codes[name].append(code)
            
            duplicates = {name: codes for name, codes in name_to_codes.items() if len(codes) > 1}
            
            if duplicates:
                for final_name, codes in duplicates.items():
                    for idx, code in enumerate(codes):
                        if idx > 0:
                            new_name = f"{final_name} (var_{idx})"
                            rename_dict[code] = new_name
            
            # PASO 4: Renombrar columnas
            df_renamed = df_with_value_labels.rename(columns=rename_dict)
            
            # PASO 5: Verificar duplicados después del renombrado
            duplicate_cols = df_renamed.columns[df_renamed.columns.duplicated()].tolist()
            
            if duplicate_cols:
                cols = pd.Series(df_renamed.columns)
                for dup in cols[cols.duplicated()].unique():
                    dup_indices = cols[cols == dup].index.tolist()
                    for idx, pos in enumerate(dup_indices):
                        if idx > 0:
                            cols[pos] = f"{dup}_dup{idx}"
                df_renamed.columns = cols
            
            # PASO 6: Añadir columnas de metadatos
            df_renamed['dataset_origen'] = dataset_id
            
            dataset_info = parse_dataset_name(dataset_id)
            df_renamed['tipo_encuesta'] = dataset_info['tipo']
            
            fecha_timestamp = create_timestamp(dataset_info['año'], dataset_info['mes'])
            df_renamed['fecha'] = fecha_timestamp
            
            unified_dfs.append(df_renamed)
        
        # Concatenar todos los datasets
        self.unified_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
        
        # Limpiar valores "No contesta"
        self.unified_df = self._clean_no_contesta_values(self.unified_df)
        
        # Generar reporte de nulls POST-unificación
        self.generate_nulls_report_post_unification()
        
        return self.unified_df

# ============================================================================
# CLASE: DiscrepancyAnalyzer (COMPLETO)
# ============================================================================

class DiscrepancyAnalyzer:
    """Analiza discrepancias en valores entre datasets"""
    
    def __init__(self, unified_df: pd.DataFrame, column_name_mapping: Dict, multiresponse_columns: set):
        self.unified_df = unified_df
        self.column_name_mapping = column_name_mapping
        self.multiresponse_columns = multiresponse_columns
        self.discrepancies = {}
    
    def _is_list_column(self, series: pd.Series) -> bool:
        """Detecta si una columna contiene listas"""
        non_null = series.dropna().head(10)
        if len(non_null) == 0:
            return False
        return any(isinstance(val, list) for val in non_null)
    
    def _extract_unique_values_from_lists(self, series: pd.Series) -> set:
        """Extrae todos los valores únicos de una columna que contiene listas"""
        unique_values = set()
        
        for val_list in series.dropna():
            if isinstance(val_list, (list, tuple)): 
                for item in val_list:
                    if pd.notna(item) and str(item).strip() != '' and str(item).lower() != 'nan':
                        unique_values.add(str(item))
        
        return unique_values
    
    def _extract_unique_values(self, series: pd.Series) -> set:
        """Extrae valores únicos, detectando automáticamente si es lista o valor simple"""
        if self._is_list_column(series):
            return self._extract_unique_values_from_lists(series)
        else:
            unique_values = set()
            for val in series.unique():
                if pd.notna(val) and str(val).strip() != '':  # ✅ CORRECTO
                    unique_values.add(str(val))
            return unique_values
    
    def _clean_dotted_values(self, df: pd.DataFrame, columns_to_clean: List[str]) -> pd.DataFrame:
        """Limpia valores con .0 SOLO en columnas unificadas especificadas"""
        cleaned_df = df.copy()
        
        for col in columns_to_clean:
            if col not in cleaned_df.columns or col == 'dataset_origen':
                continue
            
            if self._is_list_column(cleaned_df[col]):
                continue
            
            if cleaned_df[col].dtype == 'object':
                try:
                    mask = cleaned_df[col].notna() & cleaned_df[col].astype(str).str.contains('.', regex=False, na=False)
                    
                    if mask.any():
                        cleaned_df.loc[mask, col] = (
                            cleaned_df.loc[mask, col]
                            .astype(str)
                            .str.rstrip('0')
                            .str.rstrip('.')
                        )
                except Exception:
                    continue
        
        return cleaned_df
    
    def _is_numeric_variable(self, values_set: set) -> bool:
        """Determina si una variable es numérica"""
        numeric_count = 0
        total_count = len(values_set)
        
        if total_count == 0:
            return False
        
        for val in values_set:
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        return (numeric_count / total_count) > 0.8
    
    def analyze_discrepancies(self, max_different_values: int = 30) -> Dict:
        """Analiza discrepancias en valores entre datasets"""
        
        # Identificar columnas unificadas
        unified_columns = list(self.column_name_mapping.values())
        unified_columns = [col for col in unified_columns if col in self.unified_df.columns]
        
        # Limpiar valores con .0
        self.unified_df = self._clean_dotted_values(self.unified_df, unified_columns)
        
        datasets = self.unified_df['dataset_origen'].unique()
        
        for unified_name, column_label in self.column_name_mapping.items():
            if column_label not in self.unified_df.columns:
                continue
            
            is_multiresponse = self._is_list_column(self.unified_df[column_label])
            
            values_by_dataset = {}
            all_values_combined = set()
            
            for dataset in datasets:
                dataset_mask = self.unified_df['dataset_origen'] == dataset
                dataset_values = self.unified_df.loc[dataset_mask, column_label]
                
                unique_values = self._extract_unique_values(dataset_values)
                
                if unique_values:
                    values_by_dataset[dataset] = unique_values
                    all_values_combined.update(unique_values)
            
            if len(values_by_dataset) <= 1:
                continue
            
            if not is_multiresponse and self._is_numeric_variable(all_values_combined):
                continue
            
            all_sets = list(values_by_dataset.values())
            common_values = set.intersection(*all_sets)
            
            unique_per_dataset = {}
            has_differences = False
            
            for dataset, values in values_by_dataset.items():
                unique = values - common_values
                if unique:
                    unique_per_dataset[dataset] = sorted(list(unique))
                    has_differences = True
            
            if has_differences:
                all_different_values = set()
                for unique_vals in unique_per_dataset.values():
                    all_different_values.update(unique_vals)
                
                if len(all_different_values) > max_different_values:
                    continue
                
                example_dataset = list(values_by_dataset.keys())[0]
                example_values = sorted(list(values_by_dataset[example_dataset]))
                
                self.discrepancies[column_label] = {
                    'unified_name': unified_name,
                    'different_values': sorted(list(all_different_values)),
                    'example_dataset': example_dataset,
                    'example_values': example_values,
                    'common_values_count': len(common_values),
                    'is_multiresponse': is_multiresponse,
                    'num_datasets_with_differences': len(unique_per_dataset)
                }
        
        return self.discrepancies
    
    def generate_summary_for_ai(self) -> str:
        """Genera resumen de discrepancias para IA"""
        lines = [
            "# INVENTARIO DE DISCREPANCIAS EN VALORES",
            "# Objetivo: Unificar valores que difieren solo tipográficamente",
            "# IMPORTANTE - Variables Multirespuesta:",
            "# Las variables marcadas con [MULTIRESPUESTA] contienen LISTAS de valores",
            ""
        ]
        
        normal_vars = []
        multiresponse_vars = []
        
        for column_label, info in sorted(self.discrepancies.items()):
            if info['is_multiresponse']:
                multiresponse_vars.append((column_label, info))
            else:
                normal_vars.append((column_label, info))
        
        if normal_vars:
            lines.append("="*80)
            lines.append("VARIABLES NORMALES")
            lines.append("="*80)
            lines.append("")
            
            for column_label, info in normal_vars:
                lines.append(f'"{column_label}"')
                lines.append(f"   Valores comunes: {info['common_values_count']}")
                lines.append(f"   Valores diferentes: {info['different_values']}")
                lines.append(f"   Datasets con diferencias: {info['num_datasets_with_differences']}")
                lines.append(f"   Ejemplo (dataset '{info['example_dataset']}'):")
                lines.append(f"      Valores únicos: {info['example_values']}")
                lines.append("")
        
        if multiresponse_vars:
            lines.append("="*80)
            lines.append("VARIABLES MULTIRESPUESTA ⚠️")
            lines.append("="*80)
            lines.append("")
            
            for column_label, info in multiresponse_vars:
                lines.append(f'"{column_label}" ⚠️ MULTIRESPUESTA')
                lines.append(f"   Valores comunes: {info['common_values_count']}")
                lines.append(f"   Valores diferentes: {info['different_values']}")
                lines.append(f"   Datasets con diferencias: {info['num_datasets_with_differences']}")
                lines.append(f"   Ejemplo (dataset '{info['example_dataset']}'):")
                lines.append(f"      Elementos únicos en listas: {info['example_values']}")
                lines.append("")
        
        return "\n".join(lines)

# ============================================================================
# CLASE: ValueMappingGenerator (COMPLETO)
# ============================================================================

class ValueMappingGenerator:
    """Genera VALUE_MAPPING usando IA para unificar valores discrepantes"""
    
    def __init__(self, api_key: str, discrepancies_summary: str):
        self.client = OpenAI(api_key=api_key)
        self.discrepancies_summary = discrepancies_summary
        self.value_mapping = None
        self.raw_response = None
    
    def generate_mapping(self, model: str = "gpt-5.2") -> str:
        """Genera mapping con OpenAI"""
        
        # Usar prompts desde archivos o defaults
        system_prompt = VALUE_MAPPING_SYSTEM_PROMPT
        user_prompt = VALUE_MAPPING_USER_TEMPLATE.format(discrepancies=self.discrepancies_summary)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        self.raw_response = response.choices[0].message.content
        return self.raw_response
    
    def extract_python_code(self) -> str:
        """Extrae código Python de la respuesta"""
        if not self.raw_response:
            return ""
        
        code_blocks = re.findall(r'```python\n(.*?)\n```', self.raw_response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        if "VALUE_MAPPING" in self.raw_response:
            start_idx = self.raw_response.find("VALUE_MAPPING")
            return self.raw_response[start_idx:]
        
        return self.raw_response

# ============================================================================
# FUNCIÓN: Aplicar VALUE_MAPPING (COMPLETO)
# ============================================================================

def apply_value_mapping(df: pd.DataFrame, value_mapping: Dict) -> pd.DataFrame:
    """Aplica transformaciones de valores (soporta multirespuesta)"""
    final_df = df.copy()
    
    for variable, mapping in value_mapping.items():
        if variable not in final_df.columns:
            continue
        
        # Detectar si es columna con listas
        sample = final_df[variable].dropna().head(10)
        is_list_column = False
        if len(sample) > 0:
            is_list_column = any(isinstance(val, (list, tuple)) for val in sample.values)
        
        if is_list_column:
            # CASO 1: Columna con listas (multirespuesta)
            for idx in final_df.index:
                val = final_df.at[idx, variable]
                
                if val is not None and isinstance(val, (list, tuple)):
                    nueva_lista = []
                    for item in val:
                        str_item = str(item)
                        if str_item in mapping:
                            nueva_lista.append(mapping[str_item])
                        else:
                            nueva_lista.append(item)
                    
                    final_df.at[idx, variable] = nueva_lista if nueva_lista else None
        
        else:
            # CASO 2: Columna normal (valores string)
            final_df[variable] = final_df[variable].apply(
                lambda x: mapping.get(str(x), x) if x is not None else x
            )
    
    return final_df

# ============================================================================
# FUNCIÓN: Ordenar columnas por presencia
# ============================================================================

def order_columns_by_dataset_presence(df: pd.DataFrame, metadata_cols: List[str] = METADATA_COLS) -> pd.DataFrame:
    """
    Ordena las columnas del DataFrame por presencia en datasets
    Las columnas que aparecen en más datasets van primero
    """
    # Separar columnas de metadata de las demás
    meta_cols_present = [col for col in metadata_cols if col in df.columns]
    data_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Calcular presencia de cada columna en datasets
    column_presence = {}
    
    for col in data_cols:
        # Contar en cuántos datasets únicos aparece (con valores no nulos)
        datasets_with_data = set()
        
        for dataset in df['dataset_origen'].unique():
            mask = df['dataset_origen'] == dataset
            if df.loc[mask, col].notna().any():
                datasets_with_data.add(dataset)
        
        column_presence[col] = len(datasets_with_data)
    
    # Ordenar columnas por presencia (descendente)
    sorted_data_cols = sorted(data_cols, key=lambda x: column_presence[x], reverse=True)
    
    # Construir orden final: metadata primero, luego datos ordenados
    final_column_order = meta_cols_present + sorted_data_cols
    
    return df[final_column_order]

# ============================================================================
# FUNCIONES: Generar archivos adicionales
# ============================================================================

def generate_tabla_mapeo_variables(importer: SAVImporter, variable_mapping: Dict) -> pd.DataFrame:
    """Genera tabla de mapeo de variables (formato long)"""
    
    mapping_records = []
    
    for dataset_id in importer.datasets.keys():
        df = importer.datasets[dataset_id]['dataframe']
        meta = importer.metadata_collection[dataset_id]
        
        # Parsear info del dataset
        dataset_info = parse_dataset_name(dataset_id)
        fecha = create_timestamp(dataset_info['año'], dataset_info['mes'])
        tipo = dataset_info['tipo']
        
        for original_col in df.columns:
            # Buscar si fue mapeada
            mapped_to = None
            
            for unified_var, codes_dict in variable_mapping.items():
                if isinstance(codes_dict, dict):
                    if original_col in codes_dict:
                        datasets_list = codes_dict[original_col]
                        
                        if isinstance(datasets_list, str):
                            datasets_list = [datasets_list]
                        elif not isinstance(datasets_list, (list, tuple)):
                            datasets_list = []
                        
                        if dataset_id in datasets_list:
                            mapped_to = unified_var
                            break
            
            # Si no fue mapeada, se mantiene el nombre original
            if not mapped_to:
                mapped_to = original_col
            
            # Obtener label original
            original_label = meta['column_names_to_labels'].get(original_col, original_col)
            
            mapping_records.append({
                'dataset_origen': dataset_id,
                'fecha': fecha,
                'tipo_encuesta': tipo,
                'variable_original': original_col,
                'variable_unificada': mapped_to,
                'label_original': original_label,
                'fue_unificada': mapped_to != original_col
            })
    
    # Crear DataFrame
    df_mapping = pd.DataFrame(mapping_records)
    
    # Ordenar por variable unificada y dataset
    df_mapping = df_mapping.sort_values(['variable_unificada', 'fecha'])
    
    return df_mapping

def generate_matriz_presencia_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Genera matriz de presencia de variables (formato wide)"""
    
    # Obtener todas las variables del dataset unificado (excluyendo metadata)
    all_vars = [col for col in df.columns if col not in METADATA_COLS]
    
    # Crear DataFrame de presencia (True/False) para todas las variables
    presence_df = df[all_vars].notna()
    presence_df['dataset_origen'] = df['dataset_origen']
    
    # Agrupar por dataset y calcular presencia
    grouped = presence_df.groupby('dataset_origen')
    
    # Calcular matriz de presencia (1/0 si hay al menos un dato)
    presence_matrix_dict = grouped[all_vars].any().astype(int).T.to_dict()
    
    # Calcular conteo de registros con dato por variable y dataset
    count_matrix_dict = df.groupby('dataset_origen')[all_vars].apply(
        lambda x: x.notna().sum()
    ).T.to_dict()
    
    # Construir lista de resultados
    presence_matrix = []
    datasets_list = df['dataset_origen'].unique()
    
    for var in all_vars:
        row_data = {'variable': var}
        
        # Añadir presencia por dataset (1/0)
        for dataset_id in datasets_list:
            row_data[dataset_id] = presence_matrix_dict.get(dataset_id, {}).get(var, 0)
        
        # Calcular totales
        row_data['total_datasets'] = sum(
            presence_matrix_dict.get(ds, {}).get(var, 0) for ds in datasets_list
        )
        row_data['total_registros_con_dato'] = sum(
            count_matrix_dict.get(ds, {}).get(var, 0) for ds in datasets_list
        )
        
        presence_matrix.append(row_data)
    
    # Crear DataFrame
    df_presence = pd.DataFrame(presence_matrix)
    
    # Ordenar por total_datasets (descendente)
    df_presence = df_presence.sort_values('total_datasets', ascending=False)
    
    # Reorganizar columnas
    cols_order = ['variable', 'total_datasets', 'total_registros_con_dato'] + \
                 [col for col in df_presence.columns 
                  if col not in ['variable', 'total_datasets', 'total_registros_con_dato']]
    df_presence = df_presence[cols_order]
    
    return df_presence

def generate_indice_variables_long(df: pd.DataFrame, importer: SAVImporter, 
                                   variable_mapping: Dict, unifier: DatasetUnifier) -> pd.DataFrame:
    """Genera índice de variables en formato long"""
    
    # Construir mapeo inverso
    final_to_original = {}
    
    for unified_name, codes_dict in variable_mapping.items():
        final_name = unifier.column_name_mapping.get(unified_name, unified_name)
        final_to_original[final_name] = {}
        
        for original_code, datasets_list in codes_dict.items():
            if isinstance(datasets_list, str):
                datasets_list = [datasets_list]
            elif not isinstance(datasets_list, (list, tuple)):
                datasets_list = []
            
            for dataset_id in datasets_list:
                final_to_original[final_name][dataset_id] = original_code
    
    # Obtener todas las variables
    all_vars = [col for col in df.columns if col not in METADATA_COLS]
    
    # Pre-calcular presencia por dataset
    datasets_por_variable = df.groupby('dataset_origen')[all_vars].apply(
        lambda x: x.notna().any()
    ).T
    
    # Generar registros en formato long
    index_records = []
    
    for var in all_vars:
        # Obtener datasets donde la variable tiene datos
        if var in datasets_por_variable.index:
            datasets_con_datos = datasets_por_variable.loc[var]
            datasets_con_var = [ds for ds, tiene_datos in datasets_con_datos.items() if tiene_datos]
        else:
            datasets_con_var = []
        
        # Determinar tipo de variable
        try:
            sample = df[var].dropna()
            
            if len(sample) == 0:
                tipo_var = 'Sin datos'
            else:
                sample_val = sample.iloc[0]
                
                if isinstance(sample_val, list):
                    tipo_var = 'Multirespuesta'
                elif isinstance(sample_val, (int, float)) and not isinstance(sample_val, bool):
                    tipo_var = 'Numérica'
                else:
                    tipo_var = 'Categórica'
        except Exception:
            tipo_var = 'Desconocido'
        
        # Si no tiene datos en ningún dataset, crear una fila vacía
        if not datasets_con_var:
            index_records.append({
                'variable': var,
                'label': var,
                'tipo': tipo_var,
                'dataset_nombre': '',
                'dataset_fecha': None
            })
            continue
        
        # Crear una fila por cada dataset
        for dataset_id in datasets_con_var:
            
            # Obtener label de metadata
            label = var
            
            if dataset_id in importer.metadata_collection:
                meta = importer.metadata_collection[dataset_id]
                
                # Determinar código original
                if var in final_to_original and dataset_id in final_to_original[var]:
                    var_original = final_to_original[var][dataset_id]
                else:
                    var_original = var
                
                # Obtener label
                label = meta['column_names_to_labels'].get(var_original, var)
            
            # Parsear fecha del dataset
            dataset_info = parse_dataset_name(dataset_id)
            fecha = create_timestamp(dataset_info['año'], dataset_info['mes'])
            
            # Crear registro
            index_records.append({
                'variable': var,
                'label': label,
                'tipo': tipo_var,
                'dataset_nombre': dataset_id,
                'dataset_fecha': fecha
            })
    
    # Crear DataFrame
    df_index = pd.DataFrame(index_records)
    
    # Ordenar por variable (alfabético) y fecha (cronológico)
    df_index = df_index.sort_values(['variable', 'dataset_fecha'], ascending=[True, True])
    
    # Resetear índice
    df_index = df_index.reset_index(drop=True)
    
    return df_index

# ============================================================================
# INTERFAZ DE STREAMLIT
# ============================================================================

def main():
    st.title("📊 Unificador de Encuestas Longitudinales")
    st.markdown("Sistema completo de procesamiento y unificación de datasets SAV")
    st.markdown("---")
    
    # SIDEBAR: Navegación
    with st.sidebar:
        st.header("🔧 Configuración")
        
        # API Key
        try:
            secret_key = st.secrets["openai_key"]
            st.session_state.api_key = secret_key
            st.success("✅ API Key cargada desde secrets")
        except (KeyError, FileNotFoundError):
            # Si no hay secret, permitir input manual
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_key,
                help="Necesaria para generar mappings automáticamente con IA"
            )
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.header("📍 Progreso")
        
        steps = [
            "1️⃣ Subir archivos SAV",
            "2️⃣ Generar VARIABLE_MAPPING",
            "3️⃣ Editar VARIABLE_MAPPING",
            "4️⃣ Generar VALUE_MAPPING",
            "5️⃣ Editar VALUE_MAPPING",
            "6️⃣ Descargar resultado"
        ]
        
        for i, step in enumerate(steps, 1):
            if i < st.session_state.step:
                st.success(step)
            elif i == st.session_state.step:
                st.info(step)
            else:
                st.text(step)
        
        st.markdown("---")
        
        if st.button("🔄 Reiniciar proceso"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ========================================================================
    # PASO 1: SUBIR ARCHIVOS
    # ========================================================================
    
    if st.session_state.step == 1:
        st.header("1️⃣ Subir archivos SAV")
        
        st.info("""
        **¿Qué hace este paso?**
        - Importa archivos .sav (SPSS)
        - Detecta y transforma automáticamente variables multirespuesta
        - Extrae metadata completa (labels, value labels, etc.)
        - Genera inventario de variables
        """)
        
        uploaded_files = st.file_uploader(
            "Selecciona uno o más archivos .sav",
            type=['sav', 'SAV'],
            accept_multiple_files=True,
            help="Puedes seleccionar múltiples archivos a la vez"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            st.success(f"✅ {len(uploaded_files)} archivos cargados")
            
            # Mostrar lista
            with st.expander("📋 Ver archivos"):
                for file in uploaded_files:
                    date = extract_date_from_filename(file.name)
                    st.text(f"• {file.name} ({date})")
            
            if st.button("➡️ Procesar archivos", type="primary"):
                with st.spinner("Procesando archivos SAV..."):
                    # Crear importador
                    importer = SAVImporter()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"📂 Procesando: {file.name}")
                        file_bytes = file.read()
                        importer.import_file(file_bytes, file.name)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Guardar en session state
                    st.session_state.importer = importer
                    st.session_state.metadata_collection = importer.metadata_collection
                    
                    # Generar inventario
                    generator = VariableInventoryGenerator(
                        importer.metadata_collection,
                        importer.multiresponse_transformations
                    )
                    st.session_state.variable_inventory = generator.generate_variable_inventory()
                    
                    status_text.text("✅ Procesamiento completado")
                    progress_bar.progress(1.0)
                    
                    # Estadísticas
                    total_vars = sum(len(ds['dataframe'].columns) for ds in importer.datasets.values())
                    total_multiresponse = sum(len(v) for v in importer.multiresponse_transformations.values())
                    
                    st.success(f"""
                    ✅ **Archivos procesados exitosamente:**
                    - 📊 Datasets: {len(importer.datasets)}
                    - 📋 Total variables: {total_vars}
                    - 🔗 Variables multirespuesta detectadas: {total_multiresponse}
                    - 🏷️ Variables únicas en inventario: {len(st.session_state.variable_inventory)}
                    """)
                    
                    # Mostrar verificaciones de multirespuesta
                    if importer.multiresponse_verifications:
                        with st.expander("🔍 Ver verificaciones de transformaciones multirespuesta"):
                            for dataset_id, verifications in importer.multiresponse_verifications.items():
                                st.markdown(f"**Dataset: {dataset_id}**")
                                for verif in verifications:
                                    st.markdown(f"- {verif['new_column']}: {verif['original_count']} variables originales")
                    
                    st.session_state.step = 2
                    st.rerun()
    
    # ========================================================================
    # PASO 2: GENERAR VARIABLE_MAPPING
    # ========================================================================
    
    elif st.session_state.step == 2:
        st.header("2️⃣ Generar VARIABLE_MAPPING")
        
        st.info("""
        **¿Qué es VARIABLE_MAPPING?**
        
        Agrupa variables de diferentes datasets que miden **lo mismo**.
        
        Ejemplo: `"C3. Sexo"` y `"PC3. Sexo"` se unifican en `"¿Cuál es su sexo?"`
        
        **Funcionalidades:**
        - 🤖 Generación automática con IA (gpt-5.2)
        - ✍️ Escritura manual del mapping
        - 🔍 Detección de variables multirespuesta
        - 🎯 Auto-mapping por similitud (≥98.5%)
        """)
        
        tab1, tab2 = st.tabs(["🤖 Generar con IA", "✍️ Manual"])
        
        with tab1:
            if not st.session_state.api_key:
                st.warning("⚠️ Por favor, ingresa tu OpenAI API Key en el sidebar")
            else:
                st.markdown("**Vista previa del inventario:**")
                
                # Usar el inventario ya generado en Paso 1
                generator = VariableInventoryGenerator(
                    st.session_state.metadata_collection,
                    st.session_state.importer.multiresponse_transformations
                )
                generator.variable_inventory = st.session_state.variable_inventory
                summary = generator.generate_summary_for_ai()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    with st.expander("📋 Ver inventario completo"):
                        st.text(summary[:5000] + "..." if len(summary) > 5000 else summary)
                
                with col2:
                    st.metric("Variables únicas", len(st.session_state.variable_inventory))
                    st.metric("Datasets", len(st.session_state.importer.datasets))
                
                if st.button("🚀 Generar VARIABLE_MAPPING con IA", type="primary"):
                    with st.spinner("Generando mapping con gpt-5.2... (esto puede tardar entre 1 y 5 minutos...)"):
                        try:
                            mapping_generator = VariableMappingGenerator(
                                st.session_state.api_key,
                                summary
                            )
                            
                            mapping_generator.generate_mapping()
                            code = mapping_generator.extract_python_code()
                            
                            st.session_state.variable_mapping = code
                            st.session_state.step = 3
                            st.success("✅ VARIABLE_MAPPING generado exitosamente")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error al generar mapping: {str(e)}")
                            st.info("💡 Verifica tu API key o intenta usar el modo manual")
        
        with tab2:
            st.markdown("**Escribe el VARIABLE_MAPPING manualmente:**")
            
            st.code("""# Ejemplo de estructura:
VARIABLE_MAPPING = {
    "¿Cuál es su sexo?": {
        "C3": ["Dataset1", "Dataset2"],
        "PC3": ["Dataset3"]
    },
    "Edad": {
        "C4": ["Dataset1"],
        "PC4": ["Dataset2", "Dataset3"]
    }
}""", language="python")
            
            manual_mapping = st.text_area(
                "VARIABLE_MAPPING (Python dict)",
                height=400,
                placeholder="VARIABLE_MAPPING = {...}"
            )
            
            if st.button("➡️ Usar este mapping", type="primary"):
                if manual_mapping.strip():
                    st.session_state.variable_mapping = manual_mapping
                    st.session_state.step = 3
                    st.rerun()
                else:
                    st.error("❌ El mapping no puede estar vacío")
    
    # ========================================================================
    # PASO 3: EDITAR VARIABLE_MAPPING
    # ========================================================================
    
    elif st.session_state.step == 3:
        st.header("3️⃣ Revisar y Editar VARIABLE_MAPPING")
        
        st.info("""
        **Revisa el mapping antes de aplicarlo:**
        - ✅ Verifica que las agrupaciones sean correctas
        - ✏️ Haz ajustes si es necesario
        - 🔍 Asegúrate de que entidades diferentes estén separadas
        - ⚠️ Detecta duplicados automáticamente
        """)
        
        edited_mapping = st.text_area(
            "VARIABLE_MAPPING",
            value=st.session_state.variable_mapping,
            height=500,
            help="Puedes editar el código directamente"
        )
        
        # Validación sintáctica y de duplicados
        try:
            namespace = {}
            exec(edited_mapping, namespace)
            var_mapping = namespace.get('VARIABLE_MAPPING', {})
            
            if var_mapping:
                # Validar duplicados
                validation_results = validate_variable_mapping(var_mapping, st.session_state.metadata_collection)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Sintaxis válida: {len(var_mapping)} grupos de variables")
                with col2:
                    if validation_results['problems_count'] > 0:
                        st.warning(f"⚠️ {validation_results['problems_count']} problemas detectados")
                    else:
                        st.success("✅ Sin problemas detectados")
                
                # Mostrar problemas si existen
                if validation_results['problems_count'] > 0:
                    with st.expander("🔍 Ver problemas detectados"):
                        for problem in validation_results['problems_details'][:5]:
                            st.markdown(f"**{problem['unified_name']}**")
                            st.markdown(f"- Total códigos: {problem['total_codes']}")
                            st.markdown(f"- Datasets únicos: {problem['unique_datasets']}")
                            st.markdown("- Datasets duplicados:")
                            for ds, count in problem['duplicated_datasets'].items():
                                st.markdown(f"  - {ds}: {count} veces")
                        
                        if validation_results['problems_count'] > 5:
                            st.info(f"... y {validation_results['problems_count'] - 5} problemas más")
                        
                        # Opción de descargar versión anotada
                        annotated = annotate_variable_mapping_with_warnings(var_mapping, validation_results)
                        st.download_button(
                            label="📥 Descargar versión con comentarios #REVISAR",
                            data=annotated,
                            file_name="variable_mapping_annotated.py",
                            mime="text/plain"
                        )
                
                st.session_state.variable_mapping_validated = validation_results
            else:
                st.warning("⚠️ No se encontró VARIABLE_MAPPING en el código")
        except Exception as e:
            st.error(f"❌ Error de sintaxis: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ Volver", type="secondary"):
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            if st.button("✅ Aplicar VARIABLE_MAPPING", type="primary"):
                with st.spinner("Aplicando mapping y unificando datasets..."):
                    try:
                        # Ejecutar código
                        namespace = {}
                        exec(edited_mapping, namespace)
                        variable_mapping = namespace.get('VARIABLE_MAPPING', {})
                        
                        if not variable_mapping:
                            st.error("❌ No se encontró VARIABLE_MAPPING en el código")
                        else:
                            # Auto-mapping de variables no incluidas
                            variable_mapping = auto_map_unmapped_variables(
                                st.session_state.importer,
                                variable_mapping,
                                similarity_threshold=0.985
                            )
                            # Diagnóstico antes de unificar
                            unifier = DatasetUnifier(
                                st.session_state.importer,
                                variable_mapping
                            )
                            
                            diagnosis = unifier.diagnose_variable_mapping()
                            
                            if diagnosis['total_issues'] > 0:
                                st.warning(f"⚠️ Se detectaron {diagnosis['total_issues']} problemas. La unificación continuará pero puede haber errores.")
                                with st.expander("Ver diagnóstico"):
                                    st.json(diagnosis)
                            
                            # Unificar datasets
                            unified_df = unifier.unify_datasets()
                            
                            st.session_state.unified_dataset = unified_df
                            st.session_state.unifier = unifier
                            st.session_state.variable_mapping_edited = edited_mapping
                            
                            st.success(f"""
                            ✅ **Datasets unificados exitosamente:**
                            - 📊 Filas: {len(unified_df):,}
                            - 📋 Columnas: {len(unified_df.columns)}
                            - 📅 Datasets originales: {unified_df['dataset_origen'].nunique()}
                            """)
                            
                            # Mostrar reportes de nulls
                            with st.expander("📊 Ver reporte de NULLs"):
                                tab1, tab2 = st.tabs(["Pre-unificación", "Post-unificación"])
                                
                                with tab1:
                                    if unifier.nulls_report_pre is not None:
                                        st.dataframe(unifier.nulls_report_pre.head(20), use_container_width=True)
                                
                                with tab2:
                                    if unifier.nulls_report_post is not None:
                                        st.dataframe(unifier.nulls_report_post.head(20), use_container_width=True)
                            
                            st.session_state.step = 4
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error al aplicar mapping: {str(e)}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # ========================================================================
    # PASO 4: GENERAR VALUE_MAPPING
    # ========================================================================
    
    elif st.session_state.step == 4:
        st.header("4️⃣ Generar VALUE_MAPPING")
        
        st.info("""
        **¿Qué es VALUE_MAPPING?**
        
        Unifica valores discrepantes de la misma variable:
        - "HOMBRE" → "Hombre"
        - "1 Extrema izquierda" → "1"
        - "No sabe" → null
        
        **Funcionalidades:**
        - 🔍 Detección automática de discrepancias
        - 🤖 Generación con IA
        - ✍️ Edición manual
        - 🔗 Soporte para variables multirespuesta
        """)
        
        tab1, tab2 = st.tabs(["🤖 Generar con IA", "✍️ Manual / Saltar"])
        
        with tab1:
            if not st.session_state.api_key:
                st.warning("⚠️ Por favor, ingresa tu OpenAI API Key en el sidebar")
            else:
                if st.button("🔍 Analizar discrepancias", type="primary"):
                    with st.spinner("Analizando discrepancias en valores (esto puede tardar entre 1 y 5 minutos...)"):
                        # Analizar discrepancias
                        analyzer = DiscrepancyAnalyzer(
                            st.session_state.unified_dataset,
                            st.session_state.unifier.column_name_mapping,
                            st.session_state.unifier.multiresponse_columns
                        )
                        discrepancies = analyzer.analyze_discrepancies(max_different_values=30)
                        
                        st.session_state.discrepancies = discrepancies
                        
                        if not discrepancies:
                            st.success("✅ No se encontraron discrepancias. Puedes continuar al paso final.")
                            st.session_state.value_mapping = "VALUE_MAPPING = {}"
                            st.session_state.step = 5
                            st.rerun()
                        else:
                            st.info(f"📊 Se encontraron discrepancias en **{len(discrepancies)}** variables")
                
                if st.session_state.discrepancies:
                    # Generar resumen
                    analyzer = DiscrepancyAnalyzer(
                        st.session_state.unified_dataset,
                        st.session_state.unifier.column_name_mapping,
                        st.session_state.unifier.multiresponse_columns
                    )
                    analyzer.discrepancies = st.session_state.discrepancies
                    summary = analyzer.generate_summary_for_ai()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        with st.expander("📋 Ver discrepancias detectadas"):
                            st.text(summary[:3000] + "..." if len(summary) > 3000 else summary)
                    
                    with col2:
                        st.metric("Variables con discrepancias", len(st.session_state.discrepancies))
                        multiresponse_count = sum(
                            1 for d in st.session_state.discrepancies.values() 
                            if d.get('is_multiresponse', False)
                        )
                        st.metric("Variables multirespuesta", multiresponse_count)
                    
                    if st.button("🚀 Generar VALUE_MAPPING con IA"):
                        with st.spinner("Generando mapping con gpt-5.2..."):
                            try:
                                value_generator = ValueMappingGenerator(
                                    st.session_state.api_key,
                                    summary
                                )
                                
                                value_generator.generate_mapping()
                                code = value_generator.extract_python_code()
                                
                                st.session_state.value_mapping = code
                                st.session_state.step = 5
                                st.success("✅ VALUE_MAPPING generado exitosamente")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        with tab2:
            st.markdown("**Opciones manuales:**")
            
            option = st.radio(
                "¿Qué deseas hacer?",
                [
                    "Saltar este paso (no aplicar transformaciones de valores)",
                    "Escribir VALUE_MAPPING manualmente"
                ]
            )
            
            if option == "Saltar este paso (no aplicar transformaciones de valores)":
                if st.button("➡️ Continuar sin VALUE_MAPPING", type="primary"):
                    st.session_state.value_mapping = "VALUE_MAPPING = {}"
                    st.session_state.step = 5
                    st.rerun()
            else:
                st.code("""# Ejemplo de estructura:
VALUE_MAPPING = {
    "Sexo": {
        "HOMBRE": "Hombre",
        "MUJER": "Mujer",
        "No sabe": null
    }
}""", language="python")
                
                manual_value_mapping = st.text_area(
                    "VALUE_MAPPING (Python dict)",
                    height=400,
                    placeholder="VALUE_MAPPING = {...}"
                )
                
                if st.button("➡️ Usar este mapping", type="primary"):
                    if manual_value_mapping.strip():
                        st.session_state.value_mapping = manual_value_mapping
                        st.session_state.step = 5
                        st.rerun()
                    else:
                        st.error("❌ El mapping no puede estar vacío")
    
    # ========================================================================
    # PASO 5: EDITAR VALUE_MAPPING
    # ========================================================================
    
    elif st.session_state.step == 5:
        st.header("5️⃣ Revisar y Editar VALUE_MAPPING")
        
        st.info("""
        **Último paso antes de generar el dataset final:**
        - ✅ Verifica las transformaciones de valores
        - ✏️ Haz ajustes finales si es necesario
        - 🎯 Asegúrate de que "No sabe"/"No contesta" mapean a null
        """)
        
        edited_value_mapping = st.text_area(
            "VALUE_MAPPING",
            value=st.session_state.value_mapping,
            height=500
        )
        
        # Validación sintáctica
        try:
            namespace = {'null': None}
            exec(edited_value_mapping, namespace)
            val_mapping = namespace.get('VALUE_MAPPING', {})
            
            if val_mapping:
                total_transformations = sum(len(v) for v in val_mapping.values())
                st.success(f"✅ Sintaxis válida: {len(val_mapping)} variables, {total_transformations} transformaciones")
            else:
                st.info("ℹ️ VALUE_MAPPING vacío (no se aplicarán transformaciones)")
        except Exception as e:
            st.error(f"❌ Error de sintaxis: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ Volver", type="secondary"):
                st.session_state.step = 4
                st.rerun()
        
        with col2:
            if st.button("✅ Aplicar VALUE_MAPPING y Finalizar", type="primary"):
                with st.spinner("Aplicando transformaciones finales..."):
                    try:
                        # Ejecutar código
                        namespace = {'null': None}
                        exec(edited_value_mapping, namespace)
                        value_mapping = namespace.get('VALUE_MAPPING', {})
                        
                        # Aplicar transformaciones
                        final_df = apply_value_mapping(
                            st.session_state.unified_dataset,
                            value_mapping
                        )
                        
                        # Ordenar columnas por presencia
                        final_df = order_columns_by_dataset_presence(final_df, METADATA_COLS)
                        
                        st.session_state.final_dataset = final_df
                        st.session_state.value_mapping_edited = edited_value_mapping
                        
                        st.success(f"""
                        ✅ **Proceso completado exitosamente:**
                        - 📊 Dataset final: {len(final_df):,} filas × {len(final_df.columns)} columnas
                        - 🔄 Transformaciones aplicadas: {sum(len(v) for v in value_mapping.values())}
                        """)
                        
                        st.session_state.step = 6
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("Ver detalles del error"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # ========================================================================
    # PASO 6: DESCARGAR RESULTADO
    # ========================================================================
    
    elif st.session_state.step == 6:
        st.header("6️⃣ ¡Proceso Completado! 🎉")
        
        st.success("✅ El dataset longitudinal unificado está listo para análisis")
        
        # Estadísticas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Registros", f"{len(st.session_state.final_dataset):,}")
        
        with col2:
            st.metric("📋 Variables", len(st.session_state.final_dataset.columns))
        
        with col3:
            st.metric("📅 Datasets", st.session_state.final_dataset['dataset_origen'].nunique())
        
        with col4:
            fecha_min = st.session_state.final_dataset['fecha'].min()
            fecha_max = st.session_state.final_dataset['fecha'].max()
            st.metric("📆 Rango temporal", f"{fecha_min.year}-{fecha_max.year}")
        
        st.markdown("---")
        
        # Información adicional
        with st.expander("📊 Estadísticas detalladas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Por dataset:**")
                dataset_counts = st.session_state.final_dataset['dataset_origen'].value_counts()
                for dataset, count in dataset_counts.items():
                    st.text(f"• {dataset}: {count:,} registros")
            
            with col2:
                st.markdown("**Por tipo de encuesta:**")
                tipo_counts = st.session_state.final_dataset['tipo_encuesta'].value_counts()
                for tipo, count in tipo_counts.items():
                    st.text(f"• {tipo}: {count:,} registros")
        
        # Vista previa del dataset
        st.subheader("📊 Vista previa del dataset")
        st.dataframe(
            st.session_state.final_dataset.head(100),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Sección de descargas
        st.subheader("💾 Descargar archivos")
        
        # ARCHIVOS PRINCIPALES
        st.markdown("### 📌 Archivos Principales")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Dataset final CSV
            st.markdown("**📊 Dataset Final**")
            csv_buffer = io.StringIO()
            st.session_state.final_dataset.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 dataset_completo_final.csv",
                data=csv_buffer.getvalue(),
                file_name="dataset_completo_final.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            
            file_size_mb = len(csv_buffer.getvalue()) / (1024 * 1024)
            st.caption(f"Tamaño: {file_size_mb:.2f} MB")
        
        with col2:
            # VARIABLE_MAPPING
            st.markdown("**🗂️ Variable Mapping**")
            st.download_button(
                label="📥 variable_mapping.py",
                data=st.session_state.variable_mapping_edited,
                file_name="variable_mapping.py",
                mime="text/plain",
                use_container_width=True
            )
            
            # Contar grupos
            try:
                namespace = {}
                exec(st.session_state.variable_mapping_edited, namespace)
                var_map = namespace.get('VARIABLE_MAPPING', {})
                st.caption(f"Grupos: {len(var_map)}")
            except:
                st.caption("Mapping personalizado")
        
        with col3:
            # VALUE_MAPPING
            st.markdown("**🔄 Value Mapping**")
            st.download_button(
                label="📥 value_mapping.py",
                data=st.session_state.value_mapping_edited,
                file_name="value_mapping.py",
                mime="text/plain",
                use_container_width=True
            )
            
            # Contar transformaciones
            try:
                namespace = {'null': None}
                exec(st.session_state.value_mapping_edited, namespace)
                val_map = namespace.get('VALUE_MAPPING', {})
                total = sum(len(v) for v in val_map.values())
                st.caption(f"Transformaciones: {total}")
            except:
                st.caption("Mapping personalizado")
        
        st.markdown("---")
        
        # ARCHIVOS ADICIONALES (OPCIONALES)
        st.markdown("### 📎 Archivos Adicionales (Análisis)")
        
        with st.expander("🔧 Generar archivos adicionales"):
            st.info("Estos archivos son útiles para análisis detallado pero no son necesarios para el uso básico.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generar Tabla Mapeo Variables", use_container_width=True):
                    with st.spinner("Generando..."):
                        try:
                            # Ejecutar VARIABLE_MAPPING
                            namespace = {}
                            exec(st.session_state.variable_mapping_edited, namespace)
                            var_map = namespace.get('VARIABLE_MAPPING', {})
                            
                            tabla_mapeo = generate_tabla_mapeo_variables(
                                st.session_state.importer,
                                var_map
                            )
                            
                            csv_buf = io.StringIO()
                            tabla_mapeo.to_csv(csv_buf, index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label="📥 tabla_mapeo_variables.csv",
                                data=csv_buf.getvalue(),
                                file_name="tabla_mapeo_variables.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success(f"✅ {len(tabla_mapeo):,} registros")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("📊 Generar Matriz Presencia", use_container_width=True):
                    with st.spinner("Generando..."):
                        try:
                            matriz_presencia = generate_matriz_presencia_variables(
                                st.session_state.final_dataset
                            )
                            
                            csv_buf = io.StringIO()
                            matriz_presencia.to_csv(csv_buf, index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label="📥 matriz_presencia_variables.csv",
                                data=csv_buf.getvalue(),
                                file_name="matriz_presencia_variables.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success(f"✅ {len(matriz_presencia):,} variables")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col3:
                if st.button("📚 Generar Índice Variables", use_container_width=True):
                    with st.spinner("Generando..."):
                        try:
                            # Ejecutar VARIABLE_MAPPING
                            namespace = {}
                            exec(st.session_state.variable_mapping_edited, namespace)
                            var_map = namespace.get('VARIABLE_MAPPING', {})
                            
                            indice = generate_indice_variables_long(
                                st.session_state.final_dataset,
                                st.session_state.importer,
                                var_map,
                                st.session_state.unifier
                            )
                            
                            csv_buf = io.StringIO()
                            indice.to_csv(csv_buf, index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label="📥 indice_variables_long.csv",
                                data=csv_buf.getvalue(),
                                file_name="indice_variables_long.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success(f"✅ {len(indice):,} registros")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Reportes de nulls
            st.markdown("**📊 Reportes de Calidad de Datos**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.unifier and st.session_state.unifier.nulls_report_pre is not None:
                    csv_buf = io.StringIO()
                    st.session_state.unifier.nulls_report_pre.to_csv(csv_buf, index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 nulls_report_pre.csv",
                        data=csv_buf.getvalue(),
                        file_name="nulls_report_pre.csv",
                        mime="text/csv",
                        help="Reporte de NULLs antes de unificación",
                        use_container_width=True
                    )
            
            with col2:
                if st.session_state.unifier and st.session_state.unifier.nulls_report_post is not None:
                    csv_buf = io.StringIO()
                    st.session_state.unifier.nulls_report_post.to_csv(csv_buf, index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 nulls_report_post.csv",
                        data=csv_buf.getvalue(),
                        file_name="nulls_report_post.csv",
                        mime="text/csv",
                        help="Reporte de NULLs después de unificación",
                        use_container_width=True
                    )
        
        st.markdown("---")
        
        # Botón para nuevo proceso
        if st.button("🔄 Procesar nuevos archivos", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Mensaje final
        st.success("""
        ### ✅ ¡Dataset listo para análisis!
        
        **Próximos pasos sugeridos:**
        1. Importa el CSV en tu software de análisis (Python, R, SPSS, etc.)
        2. Explora las variables unificadas
        3. Analiza tendencias temporales usando la columna `fecha`
        4. Compara entre tipos de encuesta usando `tipo_encuesta`
        """)

if __name__ == "__main__":
    main()
