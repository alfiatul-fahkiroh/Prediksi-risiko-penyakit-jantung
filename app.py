import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_penyakit_jantung.pkl')

st.title("Prediksi Risiko Penyakit Jantung")
st.write("Silakan masukkan data pasien di bawah ini:")

with st.form("form_prediksi"):
    new_age = st.number_input("Masukkan usia", min_value=1, max_value=120, value=50)
    new_sex = st.selectbox("Jenis kelamin", options={"Laki-laki": 1, "Perempuan": 2})
    new_cp = st.selectbox("Tipe nyeri dada", options={
        "Typical Angina (TA)": 0,
        "Atypical Angina (ATA)": 1,
        "Non-Anginal Pain (NAP)": 2,
        "Asymptomatic (ASY)": 3
    })
    new_restingbp = st.number_input("Tekanan darah istirahat (mm Hg)", min_value=0, max_value=300, value=130)
    new_chol = st.number_input("Kadar kolesterol (mg/dL)", min_value=0, max_value=700, value=200)
    new_fbs = st.selectbox("Gula darah puasa > 120 mg/dL?", options={"Tidak": 0, "Ya": 1})
    new_ecg = st.selectbox("Hasil EKG saat istirahat", options={
        "LVH": 0, "Normal": 1, "ST-T Wave Abnormality (ST)": 2
    })
    new_maxhr = st.number_input("Detak jantung maksimum", min_value=60, max_value=250, value=150)
    new_angina = st.selectbox("Angina saat olahraga", options={"Tidak": 0, "Ya": 1})
    new_oldpeak = st.number_input("Nilai oldpeak (depresi ST)", min_value=-5.0, max_value=10.0, value=1.0)
    new_slope = st.selectbox("Kemiringan segmen ST", options={
        "Down": 0, "Flat": 1, "Up": 2
    })

    submit = st.form_submit_button("Prediksi")

if submit:
    try:
        new_data_df = pd.DataFrame([[
            new_age, new_sex, new_cp, new_restingbp, new_chol,
            new_fbs, new_ecg, new_maxhr, new_angina,
            new_oldpeak, new_slope
        ]], columns=[
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ])

        hasil_prediksi = model.predict(new_data_df)[0]
        label_mapping = {0: 'Tidak Memiliki Penyakit Jantung', 1: 'Memiliki Penyakit Jantung'}
        hasil_label = label_mapping.get(hasil_prediksi, 'Tidak diketahui')

        st.subheader("Hasil Prediksi:")
        st.success(f"Pasien diprediksi: **{hasil_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
