import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
import os

st.set_page_config(page_title="SkyProb Dashboard", layout="wide")

# --- DATA LOADER ---
@st.cache_data
def load_data():
    if os.path.exists("test_verisi.csv"):
        return pd.read_csv("test_verisi.csv")
    else:
        return pd.DataFrame({'unit_number':[1], 'RUL':[100], 'sensor_4_mean':[600], 'sensor_11_std':[0.1]})

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    if os.path.exists("motor_modeli.json"):
        model.load_model("motor_modeli.json")
    return model

try:
    df = load_data()
    model = load_model()
    
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Mng_airlines_logo.png/640px-Mng_airlines_logo.png", width=200)
    st.sidebar.title("Kontrol Paneli")
    motor_id = st.sidebar.selectbox("Motor ID", df['unit_number'].unique())
    
    subset = df[df['unit_number'] == motor_id]
    
    st.title(f"✈️ Motor {motor_id} Analizi")
    col1, col2 = st.columns(2)
    col1.metric("Kalan Ömür (RUL)", f"{subset['RUL'].iloc[-1]:.1f}")
    col2.metric("Sağlık Durumu", "Stabil" if subset['RUL'].iloc[-1] > 50 else "Riskli")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=subset['RUL'], mode='lines', name='RUL', line=dict(color='orange', width=3)))
    fig.update_layout(title="Ömür Grafiği", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Veri yüklenemedi, ancak site çalışıyor. Hata: {e}")
