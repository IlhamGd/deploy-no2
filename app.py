import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi NO2 Gresik",
    page_icon="☁️",
    layout="wide"
)

# --- Fungsi Caching (Tidak Berubah) ---
@st.cache_resource
def load_all_assets():
    models = {}
    scalers = {}
    model_names = ['h1-h2', 'h1-h3', 'h1-h4', 'h1-h5']
    
    for name in model_names:
        lag_num = name.split('-')[1] # h2, h3, h4, h5
        models[name] = joblib.load(f'knn_model_{lag_num}.joblib')
        scalers[name] = joblib.load(f'scaler_{lag_num}.joblib')
        
    data = pd.read_csv('no2_data.csv', parse_dates=['date'])
    return models, scalers, data

# --- Load Aset ---
try:
    models, scalers, data = load_all_assets()
    
    st.title('☁️ Dashboard Interaktif Prediksi Konsentrasi NO2')
    st.write("Aplikasi ini berisi dua mode prediksi: (1) Prediksi Otomatis untuk Gresik, dan (2) Prediksi Manual Interaktif.")

    # --- Tentukan Model Terbaik & Median ---
    BEST_MODEL_NAME = 'h1-h3'
    best_model = models[BEST_MODEL_NAME]
    best_scaler = scalers[BEST_MODEL_NAME]
    median_val = data['NO2'].quantile(0.50)

    st.markdown("---")
    
    # ==========================================================
    # BAGIAN 3: GRAFIK DATA (DIPINDAHKAN KE ATAS)
    # ==========================================================
    st.header("📊 Grafik Data Historis NO2")
    st.write("Grafik ini menunjukkan data historis yang digunakan untuk melatih semua model prediksi di bawah.")
    
    chart_data = data.set_index('date')['NO2']
    st.line_chart(chart_data)

    if st.checkbox("Tampilkan data historis mentah"):
        st.dataframe(data)

    st.markdown("---")

    # ==========================================================
    # BAGIAN 1: PREDIKSI OTOMATIS (Gresik)
    # ==========================================================
    with st.expander("Panel 1: Prediksi Otomatis (Gresik)", expanded=True):
        st.subheader("📈 Prediksi Otomatis Hari Berikutnya (Gresik)")
        st.write(f"Fitur ini secara otomatis mengambil 3 data terakhir dari dataset di atas dan memprediksi menggunakan model terbaik ({BEST_MODEL_NAME}).")

        # --- Logika Prediksi Otomatis ---
        last_3_values = data['NO2'].values[-3:]
        feature_names_h3 = ['h1', 'h2', 'h3']
        input_df = pd.DataFrame([last_3_values], columns=feature_names_h3)
        input_scaled = best_scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)
        pred_value = prediction[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"Prediksi NO2 (mol/m^2) - Model {BEST_MODEL_NAME}", value=f"{pred_value:.8f}")
            
            # Logika Warna Dinamis
            if pred_value <= median_val:
                st.success(f"**Kategori: Baik (Rendah)** (Prediksi: {pred_value:.8f} <= Median Historis: {median_val:.8f})")
            else:
                st.warning(f"**Kategori: Tinggi (Relatif)** (Prediksi: {pred_value:.8f} > Median Historis: {median_val:.8f})")
        
        with col2:
            st.write("**Data Input yang Digunakan:**")
            st.dataframe(input_df.style.format("{:.8f}"))


    st.markdown("---")

    # ==========================================================
    # BAGIAN 2: PREDIKSI MANUAL (Interaktif)
    # ==========================================================
    st.subheader("🔮 Panel 2: Prediksi Manual Interaktif")
    
    model_names = ['h1-h2', 'h1-h3', 'h1-h4', 'h1-h5']
    selected_name = st.selectbox('Langkah 1: Pilih Model (Jumlah Lag) yang Ingin Digunakan', model_names, index=1)
    
    lag_count_map = {'h1-h2': 2, 'h1-h3': 3, 'h1-h4': 4, 'h1-h5': 5}
    n_features = lag_count_map[selected_name]
    feature_names = [f'h{i+1}' for i in range(n_features)]
    
    st.info(f"Anda memilih model **{selected_name}**. Model ini membutuhkan **{n_features}** data input (data {n_features} hari terakhir).")

    with st.form(key='manual_prediction_form'):
        st.write(f"**Langkah 2: Masukkan {n_features} Nilai NO2 (mol/m^2)**")
        
        input_values = []
        cols = st.columns(3)
        for i in range(n_features):
            val = cols[i % 3].number_input(
                label=f"Data {n_features - i} Hari Lalu (h{i+1})", 
                key=f"val_{i}",
                format="%.8f"
            )
            input_values.append(val)
        
        submit_button = st.form_submit_button(label='Buat Prediksi Manual')

    # Logika Prediksi Manual
    if submit_button:
        model = models[selected_name]
        scaler = scalers[selected_name]
        
        input_df_manual = pd.DataFrame([input_values], columns=feature_names)
        input_scaled_manual = scaler.transform(input_df_manual)
        prediction_manual = model.predict(input_scaled_manual)
        pred_value_manual = prediction_manual[0]
        
        st.header(f"Hasil Prediksi Manual (Model {selected_name})")
        st.metric(label=f"Prediksi NO2 (mol/m^2)", value=f"{pred_value_manual:.8f}")

        # Logika Warna Dinamis
        if pred_value_manual <= median_val:
            st.success(f"**Kategori: Baik (Rendah)** (Prediksi: {pred_value_manual:.8f} <= Median Historis: {median_val:.8f})")
        else:
            st.warning(f"**Kategori: Tinggi (Relatif)** (Prediksi: {pred_value_manual:.8f} > Median Historis: {median_val:.8f})")

        st.info("Catatan: Kategori 'Baik' atau 'Tinggi' bersifat relatif terhadap data historis, bukan standar ISPU.")
        
        st.write("**Data Input yang Anda Masukkan:**")
        st.dataframe(input_df_manual.style.format("{:.8f}"))


except FileNotFoundError as e:
    st.error(f"ERROR: File Aset Tidak Ditemukan! ({e})")
    st.info("Pastikan Anda sudah mengekspor SEMUA file (knn_model_h2.joblib, scaler_h2.joblib, dst.)")
    st.info("Dan pastikan Anda sudah melakukan 'Commit & Push' ke GitHub.")
except Exception as e:
    st.error(f"Terjadi error: {e}")
