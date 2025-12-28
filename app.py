import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN (DESAIN)
# ==========================================
st.set_page_config(
    page_title="Warung Adi Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD MODEL
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('Data_Penjualan_Warung_Adi.joblib')
        scaler = joblib.load('Data_Penjualan_Warung_Adi_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# ==========================================
# 3. SIDEBAR (INPUT DATA)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/263/263142.png", width=100)
    st.title("Parameter Prediksi")
    st.markdown("Masukkan data historis warung untuk memprediksi keuntungan minggu depan.")
    st.markdown("---")

    # Input User (Sesuai Fitur Training)
    minggu_ke = st.slider("📆 Prediksi Minggu Ke-", 1, 52, 12)
    
    st.subheader("💰 Data Keuangan")
    keuntungan_minggu_lalu = st.number_input("Keuntungan Minggu Lalu (Rp)", min_value=0, value=1500000, step=50000)
    keuntungan_2_minggu_lalu = st.number_input("Keuntungan 2 Minggu Lalu (Rp)", min_value=0, value=1400000, step=50000)
    rata2_sebulan = st.number_input("Rata-rata Omset Bulan Ini (Rp)", min_value=0, value=1450000, step=50000)

    st.markdown("---")
    predict_btn = st.button("🚀 PREDIKSI SEKARANG")

# ==========================================
# 4. HALAMAN UTAMA (DASHBOARD)
# ==========================================
st.title("🏪 Warung Adi Dashboard")
st.markdown("Sistem Peramalan Keuntungan Berbasis Artificial Intelligence")

# Cek apakah model ada
if model is None:
    st.error("⚠️ File Model (.joblib) tidak ditemukan! Harap upload file model dan scaler ke folder yang sama dengan app.py.")
else:
    if predict_btn:
        # --- PROSES PREDIKSI ---
        # 1. Susun Data
        input_data = [[keuntungan_minggu_lalu, keuntungan_2_minggu_lalu, rata2_sebulan, minggu_ke]]
        
        # 2. Scaling
        input_scaled = scaler.transform(input_data)
        
        # 3. Prediksi
        prediksi_raw = model.predict(input_scaled)[0]
        prediksi_clean = max(0, prediksi_raw) # Mencegah hasil minus
        
        # 4. Hitung Pertumbuhan
        selisih = prediksi_clean - keuntungan_minggu_lalu
        persentase = (selisih / keuntungan_minggu_lalu) * 100 if keuntungan_minggu_lalu != 0 else 0
        
        # --- TAMPILAN HASIL (METRICS) ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Keuntungan Minggu Lalu</h3>
                <h2 style="color: #6c757d;">Rp {keuntungan_minggu_lalu:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            color = "#28a745" if selisih >= 0 else "#dc3545"
            arrow = "⬆️" if selisih >= 0 else "⬇️"
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid {color};">
                <h3>Prediksi Minggu Depan</h3>
                <h1 style="color: {color};">Rp {prediksi_clean:,.0f}</h1>
                <p style="color: {color}; font-weight: bold;">{arrow} {persentase:.1f}% vs Minggu Lalu</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            status = "Target Tercapai" if prediksi_clean > rata2_sebulan else "Perlu Dorongan"
            icon_status = "✅" if prediksi_clean > rata2_sebulan else "⚠️"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Status Performa</h3>
                <h2 style="color: #17a2b8;">{icon_status} {status}</h2>
            </div>
            """, unsafe_allow_html=True)

        # --- TAMPILAN GRAFIK (PLOTLY) ---
        st.write("---")
        st.subheader("📊 Visualisasi Tren")
        
        # Buat Data Dummy untuk Grafik
        weeks = ['2 Minggu Lalu', 'Minggu Lalu', 'Prediksi (Minggu Depan)']
        values = [keuntungan_2_minggu_lalu, keuntungan_minggu_lalu, prediksi_clean]
        colors = ['#adb5bd', '#adb5bd', '#28a745'] # Abu-abu untuk history, Hijau untuk prediksi

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks, 
            y=values, 
            mode='lines+markers+text',
            line=dict(color='#2c3e50', width=3, dash='dash'),
            marker=dict(size=12, color=colors),
            text=[f"Rp {v:,.0f}" for v in values],
            textposition="top center"
        ))
        
        fig.update_layout(
            title="Proyeksi Pertumbuhan Keuntungan",
            yaxis_title="Keuntungan (Rupiah)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- REKOMENDASI BISNIS ---
        st.write("---")
        st.subheader("💡 Rekomendasi AI")
        if selisih > 0:
            st.success("**Tren Positif:** Prediksi menunjukkan kenaikan omset. Pastikan stok barang (terutama best seller) tersedia lebih banyak dari minggu lalu agar tidak kehabisan barang.")
        else:
            st.warning("**Tren Menurun:** Prediksi menunjukkan sedikit penurunan. Disarankan untuk membuat promo paket hemat atau diskon akhir pekan untuk menarik pelanggan kembali.")

    else:
        # Tampilan Awal (Belum tekan tombol)
        st.info("👈 Silakan masukkan data di menu sebelah kiri dan tekan tombol 'PREDIKSI SEKARANG'.")
        
        # Placeholder Grafik Kosong agar tidak sepi
        dummy_df = pd.DataFrame({
            'Minggu': ['M1', 'M2', 'M3', 'M4'],
            'Keuntungan': [100, 120, 110, 130]
        })
        st.markdown("### Contoh Data Historis (Demo)")
        st.area_chart(dummy_df.set_index('Minggu'))