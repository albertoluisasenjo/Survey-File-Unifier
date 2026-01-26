"""
===============================================================================
SURVEY FILE UNIFIER - VERSIÓN WEB
===============================================================================
Aplicación web para unificar archivos de encuestas SPSS (.sav)
"""

import streamlit as st
import pandas as pd
import pyreadstat
from pathlib import Path
import tempfile
from io import BytesIO
from typing import List, Dict, Tuple
import re

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Survey File Unifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_spss_file(uploaded_file) -> Tuple[pd.DataFrame, dict]:
    """
    Carga un archivo SPSS desde un objeto UploadedFile
    
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
        
        return df, meta
    
    except Exception as e:
        st.error(f"Error al cargar {uploaded_file.name}: {str(e)}")
        return None, None


def get_all_columns_from_files(files_data: List[Tuple[str, pd.DataFrame, dict]]) -> List[str]:
    """
    Obtiene todas las columnas únicas de todos los archivos
    
    Args:
        files_data: Lista de tuplas (nombre_archivo, dataframe, metadata)
    
    Returns:
        Lista ordenada de columnas únicas
    """
    all_columns = set()
    
    for name, df, meta in files_data:
        all_columns.update(df.columns.tolist())
    
    return sorted(list(all_columns))


def unify_datasets(files_data: List[Tuple[str, pd.DataFrame, dict]], 
                   add_source_column: bool = True) -> pd.DataFrame:
    """
    Unifica múltiples datasets en uno solo
    
    Args:
        files_data: Lista de tuplas (nombre_archivo, dataframe, metadata)
        add_source_column: Si True, añade columna 'dataset_origen'
    
    Returns:
        DataFrame unificado
    """
    unified_dfs = []
    
    for file_name, df, meta in files_data:
        # Crear copia
        df_copy = df.copy()
        
        # Añadir columna de origen si se solicita
        if add_source_column:
            df_copy['dataset_origen'] = file_name
        
        unified_dfs.append(df_copy)
    
    # Concatenar todos los dataframes
    # outer join = mantiene todas las columnas de todos los archivos
    result = pd.concat(unified_dfs, axis=0, ignore_index=True, sort=False)
    
    return result


def generate_column_report(files_data: List[Tuple[str, pd.DataFrame, dict]]) -> pd.DataFrame:
    """
    Genera un reporte de qué columnas aparecen en qué archivos
    
    Returns:
        DataFrame con columnas: variable, label, archivo_1, archivo_2, ...
    """
    all_columns = get_all_columns_from_files(files_data)
    
    report_data = []
    
    for col in all_columns:
        row = {'variable': col}
        
        # Obtener label del primer archivo donde aparece
        label = None
        for name, df, meta in files_data:
            if col in df.columns and meta.column_names_to_labels:
                label = meta.column_names_to_labels.get(col, '')
                if label:
                    break
        
        row['label'] = label if label else ''
        
        # Marcar en qué archivos aparece
        for name, df, meta in files_data:
            if col in df.columns:
                row[name] = '✓'
            else:
                row[name] = ''
        
        report_data.append(row)
    
    return pd.DataFrame(report_data)


def export_to_spss(df: pd.DataFrame) -> BytesIO:
    """
    Exporta un DataFrame a formato SPSS (.sav) en memoria
    
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


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.title("📊 Survey File Unifier")
    st.markdown("**Unifica múltiples archivos de encuestas SPSS en un único dataset**")
    st.markdown("---")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        add_source = st.checkbox(
            "Añadir columna 'dataset_origen'",
            value=True,
            help="Añade una columna indicando de qué archivo proviene cada registro"
        )
        
        st.markdown("---")
        st.markdown("### 📝 Instrucciones")
        st.markdown("""
        1. Sube tus archivos SPSS (.sav)
        2. Revisa el resumen de archivos
        3. Descarga el archivo unificado
        
        **Nota:** El archivo unificado contendrá:
        - Todas las filas de todos los archivos
        - Todas las columnas que aparezcan en cualquier archivo
        - NaN donde una columna no existía en un archivo
        """)
    
    # Área principal - Upload
    st.header("1️⃣ Sube tus archivos")
    
    uploaded_files = st.file_uploader(
        "Selecciona archivos SPSS (.sav)",
        type=['sav'],
        accept_multiple_files=True,
        help="Puedes seleccionar múltiples archivos a la vez"
    )
    
    if not uploaded_files:
        st.info("👆 Sube al menos un archivo para comenzar")
        return
    
    # Procesar archivos
    st.header("2️⃣ Procesando archivos...")
    
    with st.spinner("Cargando archivos SPSS..."):
        files_data = []
        
        for uploaded_file in uploaded_files:
            df, meta = load_spss_file(uploaded_file)
            
            if df is not None:
                files_data.append((uploaded_file.name, df, meta))
    
    if not files_data:
        st.error("❌ No se pudo cargar ningún archivo correctamente")
        return
    
    # Mostrar resumen
    st.success(f"✅ {len(files_data)} archivos cargados correctamente")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📊 Resumen", "📋 Reporte de Columnas", "💾 Descargar"])
    
    with tab1:
        st.subheader("Resumen de archivos")
        
        # Tabla resumen
        summary_data = []
        for name, df, meta in files_data:
            summary_data.append({
                'Archivo': name,
                'Filas': len(df),
                'Columnas': len(df.columns)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Estadísticas globales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_rows = sum(row['Filas'] for row in summary_data)
            st.metric("Total de filas", f"{total_rows:,}")
        
        with col2:
            all_cols = get_all_columns_from_files(files_data)
            st.metric("Columnas únicas", len(all_cols))
        
        with col3:
            st.metric("Archivos procesados", len(files_data))
    
    with tab2:
        st.subheader("Reporte de columnas")
        st.markdown("Muestra qué columnas aparecen en cada archivo")
        
        with st.spinner("Generando reporte..."):
            report_df = generate_column_report(files_data)
        
        st.dataframe(report_df, use_container_width=True, height=400)
        
        # Botón para descargar reporte
        csv_report = report_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Descargar reporte (CSV)",
            data=csv_report,
            file_name="reporte_columnas.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Descargar archivo unificado")
        
        # Unificar datasets
        with st.spinner("Unificando datasets..."):
            unified_df = unify_datasets(files_data, add_source_column=add_source)
        
        st.success(f"✅ Dataset unificado creado: {len(unified_df):,} filas × {len(unified_df.columns)} columnas")
        
        # Preview
        with st.expander("👀 Ver preview del dataset unificado"):
            st.dataframe(unified_df.head(20), use_container_width=True)
        
        # Opciones de descarga
        st.markdown("### Formato de descarga")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Descargar como SPSS
            if st.button("📥 Preparar descarga SPSS (.sav)", type="primary"):
                with st.spinner("Generando archivo SPSS..."):
                    try:
                        spss_file = export_to_spss(unified_df)
                        
                        st.download_button(
                            label="💾 Descargar archivo SPSS",
                            data=spss_file,
                            file_name="dataset_unificado.sav",
                            mime="application/x-spss-sav"
                        )
                        st.success("✅ Archivo SPSS listo para descargar")
                    except Exception as e:
                        st.error(f"❌ Error al generar SPSS: {str(e)}")
        
        with col2:
            # Descargar como CSV
            csv_data = unified_df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name="dataset_unificado.csv",
                mime="text/csv"
            )


# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    main()
