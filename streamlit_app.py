
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('final_churn_prediction_model.pkl')

# Load the scaler, feature columns, and feature statistics
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('X_train_final_columns.pkl')
feature_stats = joblib.load('feature_stats.pkl')

st.title('Prediksi Churn Pelanggan')
st.write('Aplikasi ini memprediksi kemungkinan churn pelanggan berdasarkan input yang diberikan.')

# Collect user input for features (simplified for demonstration)
st.sidebar.header('Input Fitur Pelanggan')

def user_input_features():
    # Use loaded feature_stats for slider ranges and default values
    total_spent = st.sidebar.slider('Total Spent', feature_stats['total_spent']['min'], feature_stats['total_spent']['max'], feature_stats['total_spent']['mean'])
    satisfaction_score = st.sidebar.slider('Satisfaction Score', feature_stats['satisfaction_score']['min'], feature_stats['satisfaction_score']['max'], feature_stats['satisfaction_score']['mean'])
    support_tickets = st.sidebar.slider('Support Tickets', feature_stats['support_tickets']['min'], feature_stats['support_tickets']['max'], feature_stats['support_tickets']['mean'])
    pages_per_session = st.sidebar.slider('Pages Per Session', feature_stats['pages_per_session']['min'], feature_stats['pages_per_session']['max'], feature_stats['pages_per_session']['mean'])
    avg_session_time = st.sidebar.slider('Average Session Time', feature_stats['avg_session_time']['min'], feature_stats['avg_session_time']['max'], feature_stats['avg_session_time']['mean'])

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

st.subheader('Input Pelanggan (Setelah Scaling)')
st.write(input_df_scaled)

if st.button('Prediksi Churn'):
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)

    st.subheader('Hasil Prediksi')
    if prediction[0] == 1:
        st.error('Pelanggan ini **cenderung CHURN**.')
    else:
        st.success('Pelanggan ini **tidak cenderung CHURN**.')
    
    st.write(f"Probabilitas Churn: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Probabilitas Tidak Churn: {prediction_proba[0][0]*100:.2f}%")
