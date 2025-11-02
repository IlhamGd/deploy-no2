import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi NO2 Gresik",
    page_icon="☁️",
    layout="wide"
)

# --- Fungsi Caching untuk Load Model & Data ---
# Ini agar Streamlit tidak me-load ulang model setiap kali ada interaksi

@st.cache_resource
def load_model():
    """Memuat model K-NN yang sudah dilatih."""
    model = joblib.load('knn_model.joblib')
    return model

@st.cache_resource
def load_scaler():
    """Memuat scaler yang sudah di-fit."""
    scaler = joblib.load('scaler.joblib')
    return scaler

@st.cache_data
def load_data():
    """Memuat data historis NO2."""
    # parse_dates=['date'] penting agar tanggal dibaca sebagai datetime
    data = pd.read_csv('no2_data.csv', parse_dates=['date'])
    return data

# --- Load Aset ---
try:
    model = load_model()
    scaler = load_scaler()
    data = load_data()
    
    # --- Judul Aplikasi ---
    st.title('☁️ Dashboard Prediksi Konsentrasi NO2')
    st.write("Aplikasi ini memprediksi konsentrasi kolom NO2 (mol/m^2) untuk hari berikutnya di area Gresik.")
    st.write("Model terbaik yang digunakan adalah **K-NN (lag 3)** dengan akurasi **MAPE 0.1852%**.")

    # --- Garis Pemisah ---
    st.markdown("---")

    # --- Logika Prediksi ---
    
    # 1. Ambil 3 data NO2 terakhir (sesuai model h1-h3)
    last_3_values = data['NO2'].values[-3:]
    
    # 2. Tentukan nama fitur (HARUS SAMA DENGAN SAAT LATIHAN)
    feature_names = ['h1', 'h2', 'h3']
    
    # 3. Buat DataFrame untuk input (1 baris, 3 kolom)
    input_df = pd.DataFrame([last_3_values], columns=feature_names)
    
    # 4. Scale input
    input_scaled = scaler.transform(input_df)
    
    # 5. Lakukan prediksi
    prediction = model.predict(input_scaled)
    
    # --- Tampilkan Hasil Prediksi ---
    st.header("📈 Hasil Prediksi untuk Hari Berikutnya")
    
    col1, col2 = st.columns(2)
    
    # Kolom untuk metrik prediksi
    with col1:
        # Tampilkan prediksi dengan format yang bagus
        st.metric(
            label="Prediksi NO2 (mol/m^2)",
            value=f"{prediction[0]:.8f}"
        )
        
        # Analisis Relatif
        median_val = data['NO2'].quantile(0.50)
        if prediction[0] <= median_val:
            st.success("**Kategori: Baik (Rendah)**")
            st.write(f"Prediksi ini di bawah median historis ({median_val:.8f}).")
        else:
            st.warning("**Kategori: Tinggi (Relatif)**")
            st.write(f"Prediksi ini di atas median historis ({median_val:.8f}).")

    # Kolom untuk data input
    with col2:
        st.subheader("Data yang Digunakan (3 Hari Terakhir)")
        # Tampilkan 3 data terakhir yang dipakai sebagai input
        st.dataframe(input_df.style.format("{:.8f}"))

    # --- Tampilkan Data Historis ---
    st.markdown("---")
    st.header("📊 Grafik Data Historis NO2")
    st.write("Data ini digunakan untuk melatih model.")

    # Buat chart
    chart_data = data.set_index('date')['NO2']
    st.line_chart(chart_data)

    # Tampilkan data mentah (opsional)
    if st.checkbox("Tampilkan data historis mentah"):
        st.dataframe(data)

except FileNotFoundError:
    st.error("ERROR: File Aset Tidak Ditemukan!")
    st.info("Pastikan file 'knn_model.joblib', 'scaler.joblib', dan 'no2_data.csv' berada di folder yang sama dengan 'app.py'")
except Exception as e:
    st.error(f"Terjadi error saat melakukan prediksi: {e}")