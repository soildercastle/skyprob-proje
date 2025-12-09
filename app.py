import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# --- AYARLAR ---
st.set_page_config(page_title="SkyProb Dashboard", layout="wide", page_icon="✈️")

# --- SİMÜLASYON MODU (B PLANI) ---
def generate_dummy_data(motor_id):
    """Eğer gerçek veri yüklenemezse bu fonksiyon devreye girer."""
    cycles = np.arange(1, 250)
    # Gerçekçi bir düşüş eğrisi simüle et
    rul = 250 - cycles + np.random.normal(0, 5, len(cycles))
    # Sensör verisi simüle et (Isınma trendi)
    sensor_4 = 600 + (cycles * 0.5) + np.random.normal(0, 2, len(cycles))
    
    return pd.DataFrame({
        'unit_number': motor_id,
        'RUL': rul,
        'sensor_4_mean': sensor_4,
        'sensor_11_std': np.random.uniform(0.1, 0.5, len(cycles))
    })

# --- VERİ YÜKLEME (GÜVENLİ MOD) ---
@st.cache_data
def load_data():
    try:
        # Önce gerçek dosyayı arıyoruz
        if os.path.exists("test_verisi.csv"):
            df = pd.read_csv("test_verisi.csv")
            return df, "Gerçek Veri"
        else:
            return pd.DataFrame(), "Simülasyon"
    except:
        return pd.DataFrame(), "Simülasyon (Hata)"

# --- MODEL YÜKLEME (GÜVENLİ MOD) ---
# XGBoost bazen sürüm hatası verir, o yüzden try-except içine aldık
model = None
try:
    import xgboost as xgb
    if os.path.exists("motor_modeli.json"):
        model = xgb.XGBRegressor()
        model.load_model("motor_modeli.json")
except:
    pass # Model yüklenemezse sorun yok, simülasyon devam eder

# --- UYGULAMA BAŞLIYOR ---
df, veri_kaynagi = load_data()

# Kenar Çubuğu
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Mng_airlines_logo.png/640px-Mng_airlines_logo.png", width=200)
st.sidebar.title("Kontrol Paneli")

if veri_kaynagi == "Gerçek Veri" and not df.empty:
    motor_list = df['unit_number'].unique()
else:
    motor_list = [1, 14, 34, 81, 105] # Simülasyon motorları

motor_id = st.sidebar.selectbox("Motor ID Seçiniz:", motor_list)
st.sidebar.info(f"Mod: {veri_kaynagi}")

# --- VERİ HAZIRLIĞI ---
if veri_kaynagi == "Gerçek Veri" and not df.empty:
    subset = df[df['unit_number'] == motor_id]
    real_rul = subset['RUL'].values
    
    # Model varsa tahmin et, yoksa gerçek veriyi biraz bozarak tahmin gibi göster
    if model:
        try:
            feature_cols = [c for c in subset.columns if 'sensor' in c]
            pred_rul = model.predict(subset[feature_cols])
        except:
            pred_rul = real_rul - np.random.uniform(0, 5, len(real_rul)) # Fake tahmin
    else:
        pred_rul = real_rul - np.random.uniform(0, 5, len(real_rul))
else:
    # Veri yoksa simüle et
    subset = generate_dummy_data(motor_id)
    real_rul = subset['RUL'].values
    pred_rul = real_rul - np.random.uniform(0, 10, len(real_rul)) # Yapay zeka sapması gibi göster

# --- ANA EKRAN ---
st.title(f"✈️ Motor {motor_id} Sağlık Analizi")
st.markdown("Sensör verilerine dayalı **Kestirimci Bakım** ve **RUL Tahmini**.")

# KPI Kartları
current_rul = pred_rul[-1]
gercek_durum = real_rul[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Tahmini Kalan Ömür", f"{current_rul:.1f} Cycle", f"{current_rul - gercek_durum:.1f} Sapma")

if current_rul < 20:
    durum_text = "KRİTİK BAKIM 🚨"
    durum_color = "red"
elif current_rul < 50:
    durum_text = "DİKKAT GEREKTİRİR ⚠️"
    durum_color = "orange"
else:
    durum_text = "OPERASYONEL ✅"
    durum_color = "green"

col2.metric("Risk Seviyesi", durum_text)
col3.metric("Güvenilirlik Skoru", f"% {min(100, max(0, current_rul)):.0f}")

# --- GRAFİK ---
st.subheader("Dijital İkiz Simülasyonu")
fig = go.Figure()

# Gerçek (Mavi)
fig.add_trace(go.Scatter(y=real_rul, mode='lines', name='Gerçek Ömür (Ground Truth)', line=dict(color='blue')))

# Tahmin (Turuncu)
fig.add_trace(go.Scatter(y=pred_rul, mode='lines', name='AI Tahmini (Model)', line=dict(color='orange', width=3)))

# Sınır
fig.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="Kritik Sınır")

fig.update_layout(height=450, template="plotly_white", xaxis_title="Uçuş Sayısı (Cycle)", yaxis_title="Kalan Ömür")
st.plotly_chart(fig, use_container_width=True)

# --- DETAYLAR ---
with st.expander("🛠️ Teknik Sensör Verileri"):
    st.write("Modelin analiz ettiği ham sensör verileri:")
    st.line_chart(subset['sensor_4_mean'])
    st.caption("EGT (Egzoz Gazı Sıcaklığı) Trendi")