<img width="1366" height="768" alt="img" src="https://github.com/user-attachments/assets/75974cd9-6bd9-42ed-9e1e-6ecaab6fe029" />
# ⚡ Digital Twin Operations Center
Smart Maintenance Simulator

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

Aplikasi ini merupakan implementasi **Sistem Simulasi Cerdas (Hybrid Simulation)** berbasis web yang mengintegrasikan model prediktif *Machine Learning* dengan logika preskriptif *Sistem Pendukung Keputusan* (SPK) menggunakan metode **Simple Additive Weighting (SAW)**. Sistem ini juga dilengkapi dengan modul transparansi **Explainable AI (XAI)** berbasis SHAP.

Proyek ini dibangun sebagai tugas Ujian Akhir Semester (UAS) mata kuliah Pemodelan dan Simulasi, mendemonstrasikan fase *The Final Synthesis* dari pengembangan sistem cerdas.

---

## ✨ Fitur Utama

- **🧠 Predictive Inference:** Prediksi risiko kegagalan mesin secara *real-time* menggunakan model Regresi/Klasifikasi yang telah dilatih.
- **🎯 MCDM-SAW Engine:** Transformasi probabilitas risiko menjadi rekomendasi tindakan konkret (peringkat prioritas perbaikan mesin).
- **🔍 Explainable AI (SHAP):** Visualisasi *Waterfall Plot* untuk membongkar *black-box* model, menjelaskan mengapa sistem memberikan rekomendasi tertentu berdasarkan kontribusi fitur (Suhu & Getaran).
- **⚠️ Data Drift Detection:** Peringatan otomatis jika pengguna memasukkan parameter simulasi ekstrem yang menyimpang dari rentang data latih historis (*Handling Uncertainty*).
- **🛡️ PII Data Anonymization:** Fitur audit keamanan yang secara instan menyamarkan (*masking*) data sensitif pengguna (Nama & NIK) untuk mematuhi etika privasi data.

---

## 📂 Struktur Repositori

```text
├── app.py                        # Berkas kode utama antarmuka Streamlit (Front-end & Back-end)
├── requirements.txt              # Spesifikasi dependensi pustaka Python (versi terkunci)
├── README.md                     # Dokumentasi proyek
└── models/                       # Direktori model terserialisasi
    ├── model_risiko_v1.joblib    # Bobot biner model Machine Learning
    ├── scaler_risiko_v1.joblib   # Parameter penskalaan StandardScaler
    └── metrics_v1.json           # Metrik evaluasi model (RMSE, R2)
