"""
===============================================================================
SURVEY FILE UNIFIER - VERSIÓN WEB
===============================================================================
Aplicación web para unificar archivos de encuestas SPSS (.sav)

Funcionalidades:
1. Upload de múltiples archivos SPSS
2. Detección automática de multirespuesta
3. Mapeo manual de variables (con sugerencias)
4. Unificación y descarga
"""

import streamlit as st
import pandas as pd
from typing import List
import json

# Importar módulos propios
from utils.file_handler import load_spss_file, export_to_spss
from utils.multiresponse import (
    detect_multiresponse_variables,
    apply_multiresponse_transformations
)
from utils.variable_mapping import (
    create_variable_inventory,
    suggest_variable_mappings,
    create_mapping_dataframe,
    apply_variable_mapping,
    get_mapping_stats
)

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
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

if 'files_data' not in st.session_state:
    st.session_state.files_data = []

if 'multiresponse_transformations' not in st.session_state:
    st.session_state.multiresponse_transformations = {}

if 'variable_inventory' not in st.session_state:
    st.session_state.variable_inventory = {}

if 'mapping_df' not in st.session_state:
    st.session_state.mapping_df = None

if 'unified_df' not in st.session_state:
    st.session_state.unified_df = None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def reset_from_step(step: int):
    """Reinicia el proceso desde un paso específico"""
    if step <= 1:
        st.session_state.files_data = []
        st.session_state.multiresponse_transformations = {}
        st.session_state.variable_inventory = {}
        st.session_state.mapping_df = None
        st.session_state.unified_df = None
    elif step <= 2:
        st.session_state.multiresponse_transformations = {}
        st.session_state.variable_inventory = {}
        st.session_state.mapping_df = None
        st.session_state.unified_df = None
    elif step <= 3:
        st.session_state.variable_inventory = {}
        st.session_state.mapping_df = None
        st.session_state.unified_df = None
    elif step <= 4:
        st.session_state.unified_df = None

# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================

with st.sidebar:
    st.title("📊 Survey File Unifier")
    st.markdown("---")
    
    st.subheader("Progreso del proceso")
    
    steps = {
        1: "📁 Cargar archivos",
        2: "🔍 Detectar multirespuesta",
        3: "🗂️ Mapear variables",
        4: "💾 Descargar resultado"
    }
    
    for step_num, step_name in steps.items():
        if step_num < st.session_state.current_step:
            st.success(f"✓ {step_name}")
        elif step_num == st.session_state.current_step:
            st.info(f"→ {step_name}")
        else:
            st.text(f"  {step_name}")
    
    st.markdown("---")
    
    # Botón de reinicio
    if st.button("🔄 Reiniciar proceso", type="secondary"):
        reset_from_step(1)
        st.session_state.current_step = 1
        st.rerun()
    
    # Información
    st.markdown("---")
    st.caption("**Survey File Unifier** v1.0")
    st.caption("Desarrollado para análisis de encuestas")

# ============================================================================
# PASO 1: CARGAR ARCHIVOS
# ============================================================================

if st.session_state.current_step == 1:
    st.title("Paso 1: Cargar archivos SPSS")
    st.markdown("Sube los archivos de encuestas que deseas unificar")
    
    uploaded_files = st.file_uploader(
        "Selecciona archivos SPSS (.sav)",
        type=['sav'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("Procesar archivos", type="primary"):
            with st.spinner("Cargando archivos..."):
                files_data = []
                errors = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Procesando: {uploaded_file.name}")
                        df, metadata = load_spss_file(uploaded_file)
                        files_data.append((uploaded_file.name, df, metadata))
                        
                    except Exception as e:
                        errors.append(f"{uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                if files_data:
                    st.session_state.files_data = files_data
                    st.success(f"✅ {len(files_data)} archivos cargados correctamente")
                    
                    # Mostrar resumen
                    summary_data = []
                    for name, df, meta in files_data:
                        summary_data.append({
                            'Archivo': name,
                            'Filas': len(df),
                            'Columnas': len(df.columns)
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    
                    if errors:
                        with st.expander("⚠️ Errores encontrados"):
                            for error in errors:
                                st.error(error)
                    
                    if st.button("Continuar al siguiente paso →", type="primary"):
                        st.session_state.current_step = 2
                        st.rerun()
                else:
                    st.error("❌ No se pudo cargar ningún archivo")
    
    elif st.session_state.files_data:
        st.info("Ya tienes archivos cargados. Puedes continuar al siguiente paso.")
        if st.button("Continuar al siguiente paso →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

# ============================================================================
# PASO 2: DETECTAR MULTIRESPUESTA
# ============================================================================

elif st.session_state.current_step == 2:
    st.title("Paso 2: Detección de multirespuesta")
    st.markdown("Detectaremos automáticamente variables multirespuesta en tus archivos")
    
    if not st.session_state.files_data:
        st.error("No hay archivos cargados. Vuelve al paso 1.")
        if st.button("← Volver al paso 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        st.info(f"📁 Archivos cargados: {len(st.session_state.files_data)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Detectar multirespuesta", type="primary"):
                with st.spinner("Detectando variables multirespuesta..."):
                    all_transformations = {}
                    processed_files_data = []
                    
                    for file_name, df, metadata in st.session_state.files_data:
                        st.text(f"Procesando: {file_name}")
                        
                        # Detectar grupos
                        groups = detect_multiresponse_variables(df, metadata)
                        
                        if groups:
                            # Aplicar transformaciones
                            df_transformed, transformations = apply_multiresponse_transformations(
                                df, metadata, groups
                            )
                            all_transformations[file_name] = transformations
                            processed_files_data.append((file_name, df_transformed, metadata))
                            
                            st.success(f"✓ {file_name}: {len(groups)} grupos detectados")
                        else:
                            # No hay multirespuesta, mantener original
                            processed_files_data.append((file_name, df, metadata))
                            st.info(f"ℹ️ {file_name}: Sin multirespuesta")
                    
                    # Guardar resultados
                    st.session_state.files_data = processed_files_data
                    st.session_state.multiresponse_transformations = all_transformations
                    
                    total_groups = sum(len(t) for t in all_transformations.values())
                    st.success(f"✅ Total: {total_groups} grupos de multirespuesta transformados")
        
        with col2:
            if st.button("⏭️ Saltar este paso"):
                st.session_state.multiresponse_transformations = {}
                st.info("Sin transformaciones de multirespuesta")
        
        # Mostrar resultados si ya se detectaron
        if st.session_state.multiresponse_transformations:
            st.markdown("---")
            st.subheader("Transformaciones realizadas")
            
            for file_name, transformations in st.session_state.multiresponse_transformations.items():
                with st.expander(f"📄 {file_name} ({len(transformations)} grupos)"):
                    for idx, trans in enumerate(transformations, 1):
                        st.markdown(f"**Grupo {idx}: {trans['column_name']}**")
                        st.caption(f"Variables originales: {len(trans['original_columns'])}")
                        
                        # Mostrar mapeo de valores
                        mapping_df = pd.DataFrame([
                            {'Variable original': var, 'Valor en lista': val}
                            for var, val in trans['value_mappings'].items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True)
        
        # Botón para continuar
        st.markdown("---")
        if st.button("Continuar al siguiente paso →", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
        
        if st.button("← Volver al paso anterior"):
            st.session_state.current_step = 1
            st.rerun()

# ============================================================================
# PASO 3: MAPEAR VARIABLES
# ============================================================================

elif st.session_state.current_step == 3:
    st.title("Paso 3: Mapeo de variables")
    st.markdown("Unifica variables similares entre diferentes archivos")
    
    if not st.session_state.files_data:
        st.error("No hay archivos cargados. Vuelve al paso 1.")
        if st.button("← Volver al paso 1"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        # Crear inventario si no existe
        if not st.session_state.variable_inventory:
            with st.spinner("Creando inventario de variables..."):
                inventory = create_variable_inventory(st.session_state.files_data)
                st.session_state.variable_inventory = inventory
                
                # Crear sugerencias
                suggestions = suggest_variable_mappings(inventory)
                
                # Crear DataFrame de mapeo
                mapping_df = create_mapping_dataframe(inventory, suggestions)
                st.session_state.mapping_df = mapping_df
        
        # Mostrar estadísticas
        stats = get_mapping_stats(st.session_state.mapping_df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Variables totales", stats['total_variables'])
        col2.metric("Variables únicas", stats['variables_unificadas'])
        col3.metric("Variables agrupadas", stats['variables_agrupadas'])
        col4.metric("Sin cambios", stats['variables_sin_cambios'])
        
        st.markdown("---")
        
        # Tabs para diferentes vistas
        tab1, tab2 = st.tabs(["📝 Editar mapeo", "📊 Vista previa"])
        
        with tab1:
            st.subheader("Editar mapeo de variables")
            st.markdown("""
            **Instrucciones:**
            - Edita la columna `nombre_unificado` para agrupar variables
            - Variables con el mismo `nombre_unificado` se fusionarán
            - Puedes descargar, editar en Excel y volver a subir
            """)
            
            # Editor de datos
            edited_df = st.data_editor(
                st.session_state.mapping_df,
                use_container_width=True,
                num_rows="dynamic",
                height=400,
                column_config={
                    "variable_original": st.column_config.TextColumn("Variable original", disabled=True),
                    "label": st.column_config.TextColumn("Label", disabled=True),
                    "datasets": st.column_config.TextColumn("Datasets", disabled=True),
                    "nombre_unificado": st.column_config.TextColumn("Nombre unificado", required=True)
                }
            )
            
            # Guardar cambios
            if st.button("💾 Guardar cambios", type="primary"):
                st.session_state.mapping_df = edited_df
                st.success("✅ Cambios guardados")
                st.rerun()
            
            # Opciones de exportación/importación
            col1, col2 = st.columns(2)
            
            with col1:
                # Descargar como CSV
                csv = edited_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Descargar mapeo (CSV)",
                    data=csv,
                    file_name="variable_mapping.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Cargar CSV editado
                uploaded_mapping = st.file_uploader(
                    "📤 Cargar mapeo editado (CSV)",
                    type=['csv'],
                    key="mapping_upload"
                )
                
                if uploaded_mapping:
                    try:
                        loaded_df = pd.read_csv(uploaded_mapping)
                        st.session_state.mapping_df = loaded_df
                        st.success("✅ Mapeo cargado correctamente")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al cargar: {str(e)}")
        
        with tab2:
            st.subheader("Vista previa de agrupaciones")
            
            # Agrupar por nombre_unificado
            grouped = edited_df.groupby('nombre_unificado')
            
            # Mostrar solo grupos con más de una variable
            multi_var_groups = {name: group for name, group in grouped if len(group) > 1}
            
            if multi_var_groups:
                st.info(f"Se fusionarán {len(multi_var_groups)} grupos de variables")
                
                for name, group in list(multi_var_groups.items())[:10]:  # Mostrar primeros 10
                    with st.expander(f"📋 {name} ({len(group)} variables)"):
                        st.dataframe(
                            group[['variable_original', 'label', 'datasets']],
                            use_container_width=True
                        )
                
                if len(multi_var_groups) > 10:
                    st.caption(f"... y {len(multi_var_groups) - 10} grupos más")
            else:
                st.warning("No hay variables agrupadas. Todas mantienen su nombre original.")
        
        # Botones de navegación
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Volver al paso anterior"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col2:
            if st.button("Continuar al siguiente paso →", type="primary"):
                # Aplicar mapeo y unificar
                with st.spinner("Aplicando mapeo y unificando datasets..."):
                    try:
                        unified_df = apply_variable_mapping(
                            st.session_state.files_data,
                            st.session_state.mapping_df
                        )
                        st.session_state.unified_df = unified_df
                        st.session_state.current_step = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al unificar: {str(e)}")

# ============================================================================
# PASO 4: DESCARGAR RESULTADO
# ============================================================================

elif st.session_state.current_step == 4:
    st.title("Paso 4: Resultado y descarga")
    st.markdown("Tu dataset unificado está listo")
    
    if st.session_state.unified_df is None:
        st.error("No hay dataset unificado. Vuelve al paso anterior.")
        if st.button("← Volver al paso anterior"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        df = st.session_state.unified_df
        
        # Estadísticas
        st.subheader("📊 Estadísticas del dataset unificado")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de filas", f"{len(df):,}")
        col2.metric("Total de columnas", len(df.columns))
        col3.metric("Archivos unidos", df['dataset_origen'].nunique())
        
        # Preview
        st.markdown("---")
        st.subheader("👀 Preview del dataset")
        
        with st.expander("Ver primeras filas", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)
        
        with st.expander("Ver distribución por archivo origen"):
            dist = df['dataset_origen'].value_counts().reset_index()
            dist.columns = ['Archivo', 'Número de registros']
            st.dataframe(dist, use_container_width=True)
        
        # Descargas
        st.markdown("---")
        st.subheader("💾 Descargar resultado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Formato SPSS (.sav)**")
            st.caption("Para seguir trabajando en SPSS")
            
            if st.button("📥 Preparar descarga SPSS", type="primary"):
                with st.spinner("Generando archivo SPSS..."):
                    try:
                        spss_file = export_to_spss(df)
                        
                        st.download_button(
                            label="💾 Descargar archivo SPSS",
                            data=spss_file,
                            file_name="dataset_unificado.sav",
                            mime="application/x-spss-sav",
                            use_container_width=True
                        )
                        st.success("✅ Archivo SPSS listo")
                    except Exception as e:
                        st.error(f"Error al generar SPSS: {str(e)}")
        
        with col2:
            st.markdown("**Formato CSV**")
            st.caption("Para Excel, R, Python, etc.")
            
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name="dataset_unificado.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Metadata
        st.markdown("---")
        st.subheader("📋 Metadata del proceso")
        
        metadata = {
            'archivos_procesados': [name for name, _, _ in st.session_state.files_data],
            'transformaciones_multirespuesta': {
                file: len(trans) 
                for file, trans in st.session_state.multiresponse_transformations.items()
            },
            'estadisticas_mapeo': get_mapping_stats(st.session_state.mapping_df),
            'estadisticas_finales': {
                'total_filas': len(df),
                'total_columnas': len(df.columns),
                'archivos_unidos': int(df['dataset_origen'].nunique())
            }
        }
        
        st.json(metadata)
        
        # Descargar metadata
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Descargar metadata (JSON)",
            data=metadata_json,
            file_name="metadata_proceso.json",
            mime="application/json"
        )
        
        # Botones de navegación
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Volver al paso anterior"):
                st.session_state.current_step = 3
                st.rerun()
        
        with col2:
            if st.button("🔄 Iniciar nuevo proceso", type="primary"):
                reset_from_step(1)
                st.session_state.current_step = 1
                st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Survey File Unifier v1.0 | Desarrollado para análisis de encuestas longitudinales")
