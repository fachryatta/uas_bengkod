
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('final_churn_prediction_model.pkl')

# Load the scaler, feature columns, and feature statistics
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('X_train_final_columns.pkl')
# ===============================
# Mapping kategori -> nilai scaling
# ===============================

level_map = {
    "Sangat Rendah": -2.0,
    "Rendah": -1.0,
    "Sedang": 0.0,
    "Tinggi": 1.0,
    "Sangat Tinggi": 2.0
}
feature_stats = joblib.load('feature_stats.pkl')

st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Prediksi Churn Pelanggan")

st.markdown("""
### Sistem Prediksi Churn Pelanggan

Aplikasi ini membantu perusahaan **mengidentifikasi pelanggan yang berpotensi berhenti menggunakan layanan (churn)** berdasarkan aktivitas dan interaksi pelanggan menggunakan model **Machine Learning**.

Silakan masukkan informasi pelanggan pada panel sebelah kiri, kemudian tekan tombol **Prediksi Churn**.
""")

st.divider()

# Collect user input for features (simplified for demonstration)
st.sidebar.title("📝 Input Data Pelanggan")

st.sidebar.markdown("### 💰 Aktivitas Pelanggan")

def user_input_features():
    # Use loaded feature_stats for slider ranges and default values
    st.sidebar.subheader("💰 Aktivitas Pelanggan")

    spent_level = st.sidebar.select_slider(
        "Total Pengeluaran",
        options=list(level_map.keys()),
        value="Sedang",
        help="Total pengeluaran pelanggan."
)

    total_spent = level_map[spent_level]
    st.sidebar.markdown("### 😊 Kepuasan Pelanggan")
    st.sidebar.subheader("😊 Kepuasan Pelanggan")

    satisfaction_level = st.sidebar.select_slider(
        "Tingkat Kepuasan",
        options=[
            "Sangat Tidak Puas",
            "Tidak Puas",
            "Netral",
            "Puas",
            "Sangat Puas"
        ],
        value="Netral"
    )

    satisfaction_map = {
        "Sangat Tidak Puas": -2,
        "Tidak Puas": -1,
        "Netral": 0,
        "Puas": 1,
        "Sangat Puas": 2
    }

    satisfaction_score = satisfaction_map[satisfaction_level]
    st.sidebar.markdown("### 📞 Interaksi Pelanggan")
    st.sidebar.subheader("📞 Interaksi Pelanggan")

    support_level = st.sidebar.select_slider(
        "Jumlah Keluhan",
        options=[
            "Sangat Sedikit",
            "Sedikit",
            "Sedang",
            "Banyak",
            "Sangat Banyak"
        ],
        value="Sedang"
    )

    support_map = {
        "Sangat Sedikit": -2,
        "Sedikit": -1,
        "Sedang": 0,
        "Banyak": 1,
        "Sangat Banyak": 2
    }

    support_tickets = support_map[support_level]
    pages_level = st.sidebar.select_slider(
        "Halaman yang Dibuka per Sesi",
        options=list(level_map.keys()),
        value="Sedang",
        help="Semakin tinggi berarti pelanggan membuka lebih banyak halaman."
    )

    pages_per_session = level_map[pages_level]
    session_level = st.sidebar.select_slider(
        "Durasi Rata-rata Sesi",
        options=list(level_map.keys()),
        value="Sedang",
        help="Semakin tinggi berarti pelanggan lebih lama menggunakan layanan."
    )

    average_session_time = level_map[session_level]

    # Create a dictionary for the input for the interactive features
    data_input = {
        'total_spent': total_spent,
        'satisfaction_score': satisfaction_score,
        'support_tickets': support_tickets,
        'pages_per_session': pages_per_session,
        'avg_session_time': avg_session_time,
    }

    # Create a DataFrame with all 41 feature columns, initialized to 0
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Fill the interactive feature columns with user input
    for col, value in data_input.items():
        if col in input_df.columns:
            input_df[col] = value
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan di feature_columns. Input ini mungkin diabaikan.")

    # Customer_id should be treated carefully. For now, set to a default or 0.
    if 'customer_id' in input_df.columns:
        input_df['customer_id'] = 0 # Or some other appropriate default
    
    # Apply the loaded scaler to the input features
    # Ensure only numerical columns that were scaled are passed to the scaler
    numerical_cols_to_scale = input_df.select_dtypes(include=np.number).columns.drop('customer_id', errors='ignore')

    input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])

    return input_df

input_df_scaled = user_input_features()

if st.button("🔍 Prediksi Churn", use_container_width=True):

    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)

    st.divider()

    st.header("📈 Hasil Prediksi")

    prob = prediction_proba[0][1]

    col1, col2 = st.columns(2)

    if prediction[0] == 1:
        status = "Risiko Tinggi"
    else:
        status = "Risiko Rendah"

    col1.metric("Status Pelanggan", status)
    col2.metric("Probabilitas Churn", f"{prob*100:.2f}%")

    st.progress(float(prob))

    st.write(f"**Probabilitas Churn:** {prob*100:.2f}%")

    if prediction[0] == 1:
        st.error("🔴 Pelanggan diprediksi memiliki kemungkinan tinggi untuk berhenti menggunakan layanan.")

        st.markdown("""
### 💡 Rekomendasi

- Berikan promo atau diskon loyalitas.
- Hubungi pelanggan untuk mengetahui kendala.
- Tingkatkan kualitas pelayanan.
- Lakukan monitoring terhadap aktivitas pelanggan.
""")

    else:
        st.success("🟢 Pelanggan diprediksi akan tetap menggunakan layanan.")

        st.markdown("""
### 💡 Rekomendasi

- Pertahankan kualitas pelayanan.
- Berikan reward kepada pelanggan loyal.
- Lakukan monitoring secara berkala.
""")
    
st.divider()

st.caption("""
📌 Model : Voting Classifier

📂 Dataset : Sales & Marketing Customer Dataset

☁️ Deployment : Streamlit Cloud

© UAS Bengkel Koding Data Science
""")