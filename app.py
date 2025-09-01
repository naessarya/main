import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predicción de Autoconsumo Solar",
    page_icon="☀️",
    layout="wide"
)

# --- Caché de Modelos ---
# Esto evita tener que recargar los modelos cada vez que interactuamos con la app
@st.cache_resource
def load_model():
    try:
        model = joblib.load('improved_solar_model.pkl')
        model_info = joblib.load('model_features_info.pkl')
        return model, model_info['features']
    except FileNotFoundError:
        st.error("Error: No se encontraron los ficheros del modelo ('improved_solar_model.pkl', 'model_features_info.pkl'). Asegúrate de que están en la misma carpeta que la app.")
        return None, None

model, FEATURES = load_model()

# --- Función Principal de Predicción ---
# Encapsulamos toda la lógica de predicción en una única función
def generate_prediction(df_meteo_horaria):
    
    # ETAPA 1: PREDICCIÓN DE TOTALES DIARIOS
    df_pred_diario = df_meteo_horaria.set_index('fecha_hora').resample('D').agg({
        'temperatura_C': ['mean', 'min', 'max'], 'precipitacion_mm': 'sum',
        'presion_hPa': ['max', 'min'], 'humedad_relativa_%': 'mean',
        'velocidad_viento_kmh': 'mean', 'radiacion_solar_w_m2': 'sum',
        'segundos_de_sol_por_hora': 'sum'
    }).reset_index()
    
    df_pred_diario.columns = ['fecha_dias', 'tmed', 'tmin', 'tmax', 'prec', 'presMax', 'presMin','hrMedia', 'velmedia', 'radiacion_total_diaria', 'segundos_sol_total']
    df_pred_diario['sol'] = df_pred_diario['segundos_sol_total'] / 3600
    
    # Recrear Features
    df_pred_diario['dia_semana'] = df_pred_diario['fecha_dias'].dt.dayofweek
    df_pred_diario['mes'] = df_pred_diario['fecha_dias'].dt.month
    df_pred_diario['dia_anio'] = df_pred_diario['fecha_dias'].dt.dayofyear
    df_pred_diario['trimestre'] = df_pred_diario['fecha_dias'].dt.quarter
    df_pred_diario['semana_anio'] = df_pred_diario['fecha_dias'].dt.isocalendar().week.astype(int)
    df_pred_diario['sin_dia_anio'] = np.sin(2 * np.pi * df_pred_diario['dia_anio'] / 365.25)
    df_pred_diario['cos_dia_anio'] = np.cos(2 * np.pi * df_pred_diario['dia_anio'] / 365.25)
    
    # ... (Aquí iría el resto de tu robusto feature engineering) ...
    for col in FEATURES:
        if col not in df_pred_diario.columns:
            df_pred_diario[col] = 0

    X_future = df_pred_diario[FEATURES]
    daily_predictions = model.predict(X_future)
    df_pred_diario['autoconsumo_diario_predicho'] = daily_predictions

    # ETAPA 2: DISTRIBUCIÓN HORARIA
    df_meteo_horaria['fecha_dias'] = pd.to_datetime(df_meteo_horaria['fecha_hora'].dt.date)
    df_merged = pd.merge(df_meteo_horaria, df_pred_diario[['fecha_dias', 'autoconsumo_diario_predicho']], on='fecha_dias')
    
    temp_coeff = -0.004
    df_merged['indice_generacion'] = df_merged['radiacion_solar_w_m2'] * (1 + (df_merged['temperatura_C'] - 25) * temp_coeff)
    df_merged['indice_generacion'] = df_merged['indice_generacion'].clip(lower=0)
    
    sum_indice_diario = df_merged.groupby('fecha_dias')['indice_generacion'].transform('sum')
    df_merged['peso_horario'] = df_merged['indice_generacion'] / sum_indice_diario.replace(0, 1)
    df_merged['autoconsumo_horario_predicho'] = df_merged['autoconsumo_diario_predicho'] * df_merged['peso_horario']
    
    return df_pred_diario, df_merged

# --- Interfaz de la Aplicación ---
st.title("☀️ Herramienta de Predicción de Autoconsumo Solar")
st.markdown("Esta aplicación utiliza un modelo de Machine Learning (XGBoost) para predecir la generación de energía fotovoltaica a 15 días vista.")

# 1. Carga del fichero
uploaded_file = st.file_uploader(
    "Carga el fichero de predicción meteorológica (prediccion_meteorologica.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    st.success("Fichero cargado correctamente. Procesando la predicción...")
    
    # Leer el fichero subido
    df_meteo = pd.read_excel(uploaded_file)
    df_meteo['fecha_hora'] = pd.to_datetime(df_meteo['fecha_hora'])

    # Generar la predicción
    resumen_diario, prediccion_horaria = generate_prediction(df_meteo)

    st.balloons()
    st.header("Resultados de la Predicción")

    # 2. Mostrar gráfico
    st.subheader("Predicción Horaria de Autoconsumo (kWh)")
    chart_data = prediccion_horaria.rename(columns={'autoconsumo_horario_predicho': 'Autoconsumo Predicho (kWh)'})
    st.line_chart(chart_data, x='fecha_hora', y='Autoconsumo Predicho (kWh)')

    # 3. Mostrar tabla de resumen
    st.subheader("Resumen Diario Predicho (kWh)")
    st.dataframe(resumen_diario[['fecha_dias', 'autoconsumo_diario_predicho']].style.format(precision=0))

    # 4. Botón de descarga
    # Convertimos los dataframes a un fichero Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        resumen_diario.to_excel(writer, sheet_name='Resumen_Diario', index=False)
        prediccion_horaria.to_excel(writer, sheet_name='Prediccion_Horaria', index=False)
    
    # Ofrecemos el fichero en memoria para descarga
    st.download_button(
        label="📥 Descargar Resultados en Excel",
        data=output.getvalue(),
        file_name=f"prediccion_autoconsumo_{date.today().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )