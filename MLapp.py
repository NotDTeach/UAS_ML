#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns

# Fungsi untuk memuat data
@st.cache
def load_data():
    data = pd.read_csv('diabetes.csv')
    return data

# Fungsi untuk melatih model Naive Bayes
def train_model(x_train, y_train):
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    return nb

# Fungsi untuk melakukan prediksi dan menghitung akurasi
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Prediksi Diabetes Menggunakan Model Naive Bayes")

    # Load dataset
    data = load_data()

    # Tampilkan data
    if st.checkbox("Tampilkan Dataset"):
        st.write(data.head())

    # Normalisasi data
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data.drop('Outcome', axis=1)), columns=data.columns[:-1])
    data_scaled['Outcome'] = data['Outcome']

    # Pisahkan fitur dan label kelas sebelum normalisasi
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Pisahkan fitur dan label kelas setelah normalisasi
    X_scaled = data_scaled.drop('Outcome', axis=1)
    y_scaled = data_scaled['Outcome']

    # Bagi dataset menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Latih model Naive Bayes tanpa normalisasi
    model = train_model(X_train, y_train)
    accuracy, _ = evaluate_model(model, X_test, y_test)
    st.subheader("Naive Bayes (Sebelum Normalisasi)")
    st.write("Akurasi:", accuracy)

    # Latih model Naive Bayes dengan normalisasi
    model_scaled = train_model(X_train_scaled, y_train_scaled)
    accuracy_scaled, _ = evaluate_model(model_scaled, X_test_scaled, y_test_scaled)
    st.subheader("Naive Bayes (Setelah Normalisasi)")
    st.write("Akurasi:", accuracy_scaled)

    # Input data pengguna
    st.sidebar.header("Input Features")
    def user_input_features():
        Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
        Glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
        BloodPressure = st.sidebar.number_input('BloodPressure', min_value=0, max_value=200, value=70)
        SkinThickness = st.sidebar.number_input('SkinThickness', min_value=0, max_value=100, value=20)
        Insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=80)
        BMI = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.5)
        Age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=25)
        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Tampilkan input pengguna
    st.subheader('Input User')
    st.write(input_df)

    # Prediksi menggunakan model sebelum normalisasi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Prediksi menggunakan model setelah normalisasi
    input_df_scaled = scaler.transform(input_df)
    prediction_scaled = model_scaled.predict(input_df_scaled)
    prediction_proba_scaled = model_scaled.predict_proba(input_df_scaled)

    st.subheader('Prediksi (Sebelum Normalisasi)')
    st.write('Diabetes' if prediction[0] else 'Tidak Diabetes')
    st.write('Probabilitas Prediksi :', prediction_proba)

    st.subheader('Prediksi (Setelah Normalisasi)')
    st.write('Diabetes' if prediction_scaled[0] else 'Tidak Diabetes')
    st.write('Probabilitas Prediksi :', prediction_proba_scaled)

if __name__ == "__main__":
    main()
