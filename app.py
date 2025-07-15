import streamlit as st
import pandas as pd
import joblib

# Load model Random Forest dari file
rf_model = joblib.load('model_penyakit_jantung.pkl')

# Judul aplikasi
st.title("Prediksi Risiko Penyakit Jantung")
st.write("Silakan isi data berikut untuk memprediksi apakah pasien berisiko terkena penyakit jantung.")

# Form input pengguna
with st.form("form_prediksi"):
    new_age = st.number_input("Usia", min_value=1, max_value=120, value=50)

    new_sex = st.selectbox(
        "Jenis Kelamin",
        options=[1, 2],
        format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan"
    )

    new_cp = st.selectbox(
        "Tipe Nyeri Dada",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Typical Angina (TA)", 1: "Atypical Angina (ATA)", 2: "Non-Anginal Pain (NAP)", 3: "Asymptomatic (ASY)"}[x]
    )

    new_restingbp = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=0, max_value=300, value=120)
    new_chol = st.number_input("Kadar Kolesterol (mg/dL)", min_value=0, max_value=700, value=200)

    new_fbs = st.selectbox(
        "Gula Darah Puasa > 120 mg/dL?",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    new_ecg = st.selectbox(
        "Hasil EKG Saat Istirahat",
        options=[0, 1, 2],
        format_func=lambda x: {0: "LVH", 1: "Normal", 2: "ST-T Wave Abnormality"}[x]
    )

    new_maxhr = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=250, value=150)

    new_angina = st.selectbox(
        "Angina Saat Olahraga",
        options=[0, 1],
        format_func=lambda x: "Tidak" if x == 0 else "Ya"
    )

    new_oldpeak = st.number_input("Nilai Oldpeak (Depresi ST)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)

    new_slope = st.selectbox(
        "Kemiringan Segmen ST",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Down", 1: "Flat", 2: "Up"}[x]
    )

    # Tombol prediksi
    submit = st.form_submit_button("Prediksi")

# Proses prediksi
if submit:
    try:
        # Buat DataFrame dari input pengguna
        new_data_df = pd.DataFrame([[
            new_age, new_sex, new_cp, new_restingbp, new_chol,
            new_fbs, new_ecg, new_maxhr, new_angina,
            new_oldpeak, new_slope
        ]], columns=[
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ])

        # Prediksi
        hasil_prediksi = rf_model.predict(new_data_df)[0]
        label_mapping = {0: 'Tidak Memiliki Penyakit Jantung', 1: 'Memiliki Penyakit Jantung'}
        hasil_label = label_mapping.get(hasil_prediksi, 'Tidak diketahui')

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.success(f"Pasien diprediksi: **{hasil_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")

