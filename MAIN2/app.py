import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import io
import requests # Necesario para la descarga de Open-Meteo

# --- 1. Configuración de la Página y Título ---
st.set_page_config(
    page_title="Predicción de Autoconsumo Solar",
    page_icon="☀️",
    layout="wide"
)

st.image("Ipsom-Logo.png", width=200) # Asegúrate de tener el logo en la misma carpeta
st.title("Panel de Predicción de Autoconsumo Solar")
st.markdown("Herramienta interna para generar pronósticos de generación fotovoltaica a 15 días vista.")

# --- Autenticación Simple (puedes cambiar la contraseña) ---
CORRECT_PASSWORD = "ipsom" 

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        password = st.text_input("Introduce la contraseña para continuar", type="password")
        if st.button("Acceder"):
            if password == CORRECT_PASSWORD:
                st.session_state["password_correct"] = True
                st.experimental_rerun() # Recarga la app tras el login exitoso
            else:
                st.error("La contraseña es incorrecta.")
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- Funciones Principales (Caché para mayor eficiencia) ---

@st.cache_resource
def load_model():
    """Carga el modelo y la lista de features una sola vez."""
    try:
        model = joblib.load('improved_solar_model.pkl')
        model_info = joblib.load('model_features_info.pkl')
        return model, model_info['features']
    except FileNotFoundError:
        st.error("Error crítico: Faltan los ficheros del modelo ('improved_solar_model.pkl', 'model_features_info.pkl').")
        return None, None

# [NUEVO] Función para descargar los datos de Open-Meteo
@st.cache_data # Usamos caché para no descargar los datos repetidamente
def get_openmeteo_data():
    """Descarga los datos de Open-Meteo y los devuelve como un DataFrame."""
    url = "https://api.open-meteo.com/v1/forecast?latitude=41.15&longitude=1.17&hourly=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,shortwave_radiation,sunshine_duration&timezone=Europe/Madrid&forecast_days=15"
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza un error si la petición falla
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df.rename(columns={
            'time': 'fecha_hora', 'temperature_2m': 'temperatura_C', 
            'relative_humidity_2m': 'humedad_relativa_%', 'precipitation': 'precipitacion_mm',
            'surface_pressure': 'presion_hPa', 'wind_speed_10m': 'velocidad_viento_kmh',
            'shortwave_radiation': 'radiacion_solar_w_m2', 'sunshine_duration': 'segundos_de_sol_por_hora'
        }, inplace=True)
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        return df
    except requests.RequestException as e:
        st.error(f"Error al contactar con Open-Meteo: {e}")
        return None

def generate_prediction(df_meteo_horaria, model, features):
    """Ejecuta el pipeline de predicción completo (sin cambios)."""
    # (Toda tu lógica de predicción de 2 etapas va aquí, igual que antes)
    df_pred_diario = df_meteo_horaria.set_index('fecha_hora').resample('D').agg({
        'temperatura_C': ['mean', 'min', 'max'], 'precipitacion_mm': 'sum',
        'presion_hPa': ['max', 'min'], 'humedad_relativa_%': 'mean',
        'velocidad_viento_kmh': 'mean', 'radiacion_solar_w_m2': 'sum',
        'segundos_de_sol_por_hora': 'sum'
    }).reset_index()
    df_pred_diario.columns = ['fecha_dias', 'tmed', 'tmin', 'tmax', 'prec', 'presMax', 'presMin','hrMedia', 'velmedia', 'radiacion_total_diaria', 'segundos_sol_total']
    df_pred_diario['sol'] = df_pred_diario['segundos_sol_total'] / 3600
    df_pred_diario['dia_semana'] = df_pred_diario['fecha_dias'].dt.dayofweek
    df_pred_diario['mes'] = df_pred_diario['fecha_dias'].dt.month
    df_pred_diario['dia_anio'] = df_pred_diario['fecha_dias'].dt.dayofyear
    df_pred_diario['cos_dia_anio'] = np.cos(2 * np.pi * df_pred_diario['dia_anio'] / 365.25)
    
    for col in features:
        if col not in df_pred_diario.columns:
            df_pred_diario[col] = 0
    
    X_future = df_pred_diario[features]
    daily_predictions = model.predict(X_future)
    df_pred_diario['autoconsumo_diario_predicho'] = daily_predictions

    df_meteo_horaria['fecha_dias'] = pd.to_datetime(df_meteo_horaria['fecha_hora'].dt.date)
    df_merged = pd.merge(df_meteo_horaria, df_pred_diario[['fecha_dias', 'autoconsumo_diario_predicho']], on='fecha_dias')
    
    temp_coeff = -0.004
    df_merged['indice_generacion'] = df_merged['radiacion_solar_w_m2'] * (1 + (df_merged['temperatura_C'] - 25) * temp_coeff)
    df_merged['indice_generacion'] = df_merged['indice_generacion'].clip(lower=0)
    
    sum_indice_diario = df_merged.groupby('fecha_dias')['indice_generacion'].transform('sum')
    df_merged['peso_horario'] = df_merged['indice_generacion'] / sum_indice_diario.replace(0, 1)
    df_merged['autoconsumo_horario_predicho'] = df_merged['autoconsumo_diario_predicho'] * df_merged['peso_horario']
    
    return df_pred_diario, df_merged


# --- Cuerpo de la Aplicación con el Nuevo Flujo de Trabajo ---

# Cargar modelo y features
model, FEATURES = load_model()
if model is None:
    st.stop() # Detiene la app si no se pueden cargar los modelos

# Dividir la interfaz en columnas para un diseño más limpio
col1, col2 = st.columns(2)

with col1:
    st.subheader("Paso 1: Obtener Datos Meteorológicos")
    if st.button("📡 Actualizar Pronóstico desde Open-Meteo"):
        with st.spinner("Descargando los últimos datos..."):
            # Guardamos los datos en el estado de la sesión para que persistan
            st.session_state['df_meteo'] = get_openmeteo_data()
            if st.session_state['df_meteo'] is not None:
                st.success("Pronóstico meteorológico actualizado.")

# Si ya tenemos los datos meteorológicos, mostramos el siguiente paso
if 'df_meteo' in st.session_state:
    with col2:
        st.subheader("Paso 2: Generar Predicción")
        if st.button("🤖 Ejecutar Modelo de Predicción"):
            with st.spinner("Calculando la predicción..."):
                # Usamos los datos guardados en la sesión
                resumen_diario, prediccion_horaria = generate_prediction(st.session_state['df_meteo'], model, FEATURES)
                st.session_state['resumen_diario'] = resumen_diario
                st.session_state['prediccion_horaria'] = prediccion_horaria
                st.success("¡Predicción generada con éxito!")
                st.balloons()

# Si ya se ha generado una predicción, mostramos los resultados
if 'prediccion_horaria' in st.session_state:
    st.divider()
    st.header("Resultados de la Predicción")

    # Gráfico
    chart_data = st.session_state['prediccion_horaria'].rename(columns={'autoconsumo_horario_predicho': 'Autoconsumo Predicho (kWh)'})
    st.line_chart(chart_data, x='fecha_hora', y='Autoconsumo Predicho (kWh)')

    # Tablas de datos con pestañas
    tab_diaria, tab_horaria = st.tabs(["Resumen Diario", "Detalle Horario"])
    with tab_diaria:
        st.dataframe(st.session_state['resumen_diario'][['fecha_dias', 'autoconsumo_diario_predicho']])
    with tab_horaria:
        st.dataframe(st.session_state['prediccion_horaria'][['fecha_hora', 'autoconsumo_horario_predicho', 'radiacion_solar_w_m2']])
    
    # Botón de descarga final
    output_pred = io.BytesIO()
    with pd.ExcelWriter(output_pred, engine='openpyxl') as writer:
        st.session_state['resumen_diario'].to_excel(writer, sheet_name='Resumen_Diario', index=False)
        st.session_state['prediccion_horaria'].to_excel(writer, sheet_name='Prediccion_Horaria', index=False)
    
    st.download_button(
        label="📥 Descargar Predicción Completa en Excel",
        data=output_pred.getvalue(),
        file_name=f"prediccion_autoconsumo_{date.today().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )