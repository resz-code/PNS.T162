import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. PAGE INITIALIZATION & DARK THEME CONFIG
# ==========================================
st.set_page_config(
    page_title="Digital Twin | Smart Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Modern Dark Theme (Cyber-Industrial Aesthetic)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }

    .header-container {
        background: linear-gradient(135deg, #1A202C 0%, #0F172A 100%);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #334155;
        border-bottom: 3px solid #3B82F6;
        box-shadow: 0 4px 20px -2px rgba(59, 130, 246, 0.15);
        margin-bottom: 30px;
    }
    .header-title { font-size: 32px; font-weight: 700; color: #F8FAFC; margin-bottom: 5px; letter-spacing: -0.5px; }
    .header-subtitle { font-size: 15px; color: #94A3B8; font-weight: 300; }

    .modern-card {
        background-color: #111827;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.05), 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .modern-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .card-title { font-size: 13px; text-transform: uppercase; color: #94A3B8; font-weight: 600; margin-bottom: 12px; letter-spacing: 0.5px; }
    .custom-metric-val { font-size: 38px; font-weight: 700; color: #F8FAFC; line-height: 1; }
    .custom-metric-delta { font-size: 14px; font-weight: 500; margin-top: 10px; }
    .delta-up { color: #EF4444; text-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
    .delta-down { color: #10B981; text-shadow: 0 0 10px rgba(16, 185, 129, 0.3); }

    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD PRODUCTION PIPELINE + MODEL METRICS
# ==========================================
@st.cache_resource
def load_production_pipeline():
    try:
        model = joblib.load('models/model_risiko_v1.joblib')
        scaler = joblib.load('models/scaler_risiko_v1.joblib')
        return model, scaler
    except Exception:
        return None, None


@st.cache_data
def load_model_metrics():
    """
    Memuat metrik evaluasi model (RMSE, R2, dsb) dari file metadata terpisah.
    File ini WAJIB dibuat oleh mahasiswa saat proses training (lihat notebook training),
    lalu disimpan sebagai models/metrics_v1.json, contoh isi:
    {"rmse": 6.42, "r2": 0.87, "n_test": 45}
    Jika file tidak ditemukan, sistem TIDAK mengarang angka -> menampilkan status
    "belum tervalidasi" agar tidak menyesatkan pengguna (lih. Kriteria A - Fungsionalitas Prediktif).
    """
    try:
        with open('models/metrics_v1.json') as f:
            return json.load(f)
    except Exception:
        return {"rmse": None, "r2": None, "n_test": None}


model, scaler = load_production_pipeline()
metrics = load_model_metrics()

# Parameter Acuan Historis
TRAIN_MEAN_REF = 43.0
BASELINE_SUHU = 75.0
BASELINE_GETARAN = 5.0

# ==========================================
# 3. LOGIC BACKEND ENGINES
# ==========================================
def anonymize_operator_identity(name: str, nik: str) -> dict:
    """
    Anonimisasi PII secara nyata (bukan sekadar label status):
    - Nama disamarkan menjadi inisial + placeholder.
    - NIK disamarkan, hanya 4 digit pertama yang tampil, sisanya di-mask.
    Data mentah TIDAK pernah ditulis ke log/file/console di aplikasi ini.
    """
    name = (name or "").strip()
    masked_name = f"{name[0].upper()}***" if name else "N/A"

    digits = "".join(ch for ch in (nik or "") if ch.isdigit())
    if digits:
        visible = digits[:4]
        masked_nik = visible + "*" * max(len(digits) - 4, 0)
    else:
        masked_nik = "N/A"

    return {
        "masked_name": masked_name,
        "masked_nik": masked_nik,
        "status": "PII Ter-mask, Tidak Disimpan ke Log"
    }


def check_data_drift(suhu, getaran, threshold=2.0):
    new_data_mean = np.mean([suhu, getaran])
    drift_score = np.abs(new_data_mean - TRAIN_MEAN_REF)
    return drift_score > threshold, drift_score


@st.cache_resource
def build_shap_explainer(_model, _scaler, background_suhu=BASELINE_SUHU, background_getaran=BASELINE_GETARAN):
    """
    Membangun SHAP explainer sesuai instruksi Lecture Note (Langkah 2: Integrasi Modul XAI).
    Menggunakan LinearExplainer bila model linier (lebih cepat & eksak),
    fallback ke generic Explainer bila model bukan linier.
    Background data memakai titik baseline historis karena dataset training
    penuh tidak disertakan dalam repositori deployment ini.
    """
    background = _scaler.transform(np.array([[background_suhu, background_getaran]]))
    try:
        explainer = shap.LinearExplainer(_model, background)
    except Exception:
        explainer = shap.Explainer(_model.predict, background)
    return explainer


def run_saw_optimization(dynamic_risk_mesin_a):
    weights = np.array([0.5, 0.3, 0.2])
    matrix_x = np.array([
        [dynamic_risk_mesin_a, 15.0, 80.0],
        [45.0, 10.0, 60.0],
        [70.0, 25.0, 95.0]
    ], dtype=float)

    epsilon = 1e-9
    matrix_x[:, 0] = np.where(matrix_x[:, 0] == 0, epsilon, matrix_x[:, 0])
    matrix_x[:, 1] = np.where(matrix_x[:, 1] == 0, epsilon, matrix_x[:, 1])

    alternatives = ["Mesin A (Unit Kontrol)", "Mesin B (Lini Produksi 2)", "Mesin C (Gudang Utama)"]
    normalized_r = np.zeros(matrix_x.shape)

    normalized_r[:, 0] = np.min(matrix_x[:, 0]) / matrix_x[:, 0]
    normalized_r[:, 1] = np.min(matrix_x[:, 1]) / matrix_x[:, 1]
    normalized_r[:, 2] = matrix_x[:, 2] / np.max(matrix_x[:, 2])

    skor_akhir = np.dot(normalized_r, weights)

    df_res = pd.DataFrame({
        "Alternatif Mesin": alternatives,
        "Skor Prioritas (V)": skor_akhir,
        "Risiko (%)": matrix_x[:, 0],
        "Biaya (Juta)": matrix_x[:, 1],
        "Urgensi": matrix_x[:, 2]
    })
    return df_res.sort_values(by="Skor Prioritas (V)", ascending=False).reset_index(drop=True)

# ==========================================
# 4. INTERACTIVE DARK UI
# ==========================================
st.markdown("""
    <div class="header-container">
        <div class="header-title">⚡ Digital Twin Operations Center</div>
        <div class="header-subtitle">Simulator Kebijakan What-If Terintegrasi ML & Sistem Pendukung Keputusan (SAW)</div>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar: Audit Keamanan (Anonimisasi Nyata) ---
st.sidebar.markdown("<h4 style='color:#60A5FA; font-weight:600; letter-spacing:1px;'>🛡️ AUDIT KEAMANAN</h4>", unsafe_allow_html=True)
input_nama = st.sidebar.text_input("Nama Operator", value="")
input_nik = st.sidebar.text_input("NIK Petugas", value="")
anonym_meta = anonymize_operator_identity(input_nama, input_nik)
st.sidebar.markdown(
    f"""<div style='background:#1E293B; padding:10px; border-radius:6px; font-size:12px; color:#34D399; border-left:3px solid #10B981;'>
    🔒 {anonym_meta['status']}<br>
    <span style='color:#94A3B8;'>Nama: {anonym_meta['masked_name']} &nbsp;|&nbsp; NIK: {anonym_meta['masked_nik']}</span>
    </div>""",
    unsafe_allow_html=True
)

st.sidebar.markdown("<br><h4 style='color:#60A5FA; font-weight:600; letter-spacing:1px;'>🎛️ TUAS INTERVENSI</h4>", unsafe_allow_html=True)
suhu_slider = st.sidebar.slider("Suhu Operasional (°C)", 10, 200, int(BASELINE_SUHU))
getaran_slider = st.sidebar.slider("Amplitudo Getaran (mm/s)", 0, 30, int(BASELINE_GETARAN))

if model is not None and scaler is not None:
    # Perhitungan Inferensi
    baseline_scaled = scaler.transform(np.array([[BASELINE_SUHU, BASELINE_GETARAN]]))
    pred_baseline = max(0.0, min(100.0, model.predict(baseline_scaled)[0]))

    intervensi_scaled = scaler.transform(np.array([[suhu_slider, getaran_slider]]))
    pred_intervensi = max(0.0, min(100.0, model.predict(intervensi_scaled)[0]))
    delta_risiko = pred_intervensi - pred_baseline

    # Deteksi Drift
    is_drift, drift_val = check_data_drift(suhu_slider, getaran_slider)
    if is_drift:
        st.sidebar.markdown(f'''
            <div style="background-color:rgba(245, 158, 11, 0.1); padding:15px; border-radius:8px; border-left:4px solid #F59E0B; margin-top:20px;">
                <span style="color:#FCD34D; font-size:13px; font-weight:600;">⚠️ DATA DRIFT ({drift_val:.2f})</span><br>
                <span style="color:#D1D5DB; font-size:12px;">Input melampaui rentang validitas historis model. Prediksi di bawah ini berisiko kurang akurat.</span>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
            <div style="background-color:rgba(16, 185, 129, 0.1); padding:12px; border-radius:8px; border-left:4px solid #10B981; margin-top:20px;">
                <span style="color:#6EE7B7; font-size:12px; font-weight:600;">✅ SISTEM STABIL</span><br>
                <span style="color:#D1D5DB; font-size:11px;">Distribusi input konsisten dengan data latih.</span>
            </div>
        """, unsafe_allow_html=True)

    # --- Kartu Transparansi Error Model (Kriteria A: Fungsionalitas Prediktif) ---
    if metrics.get("rmse") is not None:
        rmse_display = f"± {metrics['rmse']:.2f}%"
        rmse_note = f"Berdasarkan evaluasi pada {metrics.get('n_test', 'N/A')} data uji (R² = {metrics.get('r2', 'N/A')})."
        rmse_box_color = "#3B82F6"
    else:
        rmse_display = "Belum tervalidasi"
        rmse_note = "Tambahkan file models/metrics_v1.json (hasil evaluasi test set) agar tingkat error model dapat ditampilkan di sini."
        rmse_box_color = "#F59E0B"

    st.markdown(f'''
        <div class="modern-card" style="border-left:4px solid {rmse_box_color};">
            <div class="card-title">Tingkat Ketidakpastian Model (RMSE)</div>
            <div style="font-size:20px; font-weight:700; color:#F8FAFC;">{rmse_display}</div>
            <div style="font-size:12px; color:#94A3B8; margin-top:6px;">{rmse_note}</div>
        </div>
    ''', unsafe_allow_html=True)

    # Layout Tab Utama
    tab1, tab2 = st.tabs(["🎯 Simulasi Keputusan (SPK)", "🧠 Transparansi Model (XAI - SHAP)"])

    with tab1:
        c1, c2, c3 = st.columns(3)

        with c1:
            delta_class = "delta-up" if delta_risiko >= 0 else "delta-down"
            delta_sign = "+" if delta_risiko >= 0 else ""
            st.markdown(f'''
                <div class="modern-card">
                    <div class="card-title">Risiko Kegagalan Mesin A</div>
                    <div class="custom-metric-val">{pred_intervensi:.2f}%</div>
                    <div class="custom-metric-delta {delta_class}">{delta_sign}{delta_risiko:.2f}% Deviasi (What-If)</div>
                </div>
            ''', unsafe_allow_html=True)

        with c2:
            st.markdown(f'''
                <div class="modern-card">
                    <div class="card-title">Parameter Saat Ini (Intervensi)</div>
                    <div style="font-size: 15px; color:#F8FAFC; font-weight:500; line-height:1.7;">
                        <span style="color:#94A3B8;">Suhu:</span> <b style="color:#60A5FA;">{suhu_slider} °C</b><br>
                        <span style="color:#94A3B8;">Getaran:</span> <b style="color:#60A5FA;">{getaran_slider} mm/s</b>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        with c3:
            st.markdown(f'''
                <div class="modern-card">
                    <div class="card-title">Kondisi Standar (Baseline)</div>
                    <div style="font-size: 15px; color:#94A3B8; font-weight:400; line-height:1.7;">
                        Suhu: 75.0 °C<br>
                        Getaran: 5.0 mm/s
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<h4 style='color:#F8FAFC; font-weight:600; margin-top:10px; padding-bottom:10px; border-bottom:1px solid #1E293B;'>Peringkat Prioritas Tindakan (MCDM - SAW Engine)</h4>", unsafe_allow_html=True)

        df_ranking = run_saw_optimization(pred_intervensi)

        styled_df = df_ranking.style.background_gradient(
            cmap="ocean", subset=["Skor Prioritas (V)"]
        ).format({"Skor Prioritas (V)": "{:.4f}", "Risiko (%)": "{:.1f}%", "Biaya (Juta)": "Rp {:.1f} Jt"})

        st.dataframe(styled_df, use_container_width=True)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor('#0B0F19')
        ax.set_facecolor('#0B0F19')

        colors = ['#34D399' if x == df_ranking["Skor Prioritas (V)"].max() else '#334155' for x in df_ranking["Skor Prioritas (V)"]]

        bars = ax.barh(df_ranking["Alternatif Mesin"], df_ranking["Skor Prioritas (V)"], color=colors, height=0.4)
        ax.set_xlabel("Skor Preferensi ($V_i$)", fontsize=10, fontweight='500', color='#94A3B8')

        ax.tick_params(colors='#CBD5E1', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.grid(axis='x', linestyle=':', alpha=0.3, color='#94A3B8')
        ax.invert_yaxis()

        st.pyplot(fig)

    with tab2:
        st.markdown("<h4 style='color:#F8FAFC; font-weight:600; margin-bottom:15px;'>🔍 Mengapa Hasilnya Demikian? (SHAP Explanation)</h4>", unsafe_allow_html=True)

        explainer = build_shap_explainer(model, scaler)
        shap_values_raw = explainer(intervensi_scaled)

        # Bungkus hasil SHAP dengan nama fitur agar plot mudah dibaca
        shap_explanation = shap.Explanation(
            values=shap_values_raw.values[0],
            base_values=shap_values_raw.base_values[0] if np.ndim(shap_values_raw.base_values) > 0 else shap_values_raw.base_values,
            data=intervensi_scaled[0],
            feature_names=["Suhu Operasional", "Amplitudo Getaran"]
        )

        plt.style.use('dark_background')
        fig_xai = plt.figure(figsize=(10, 3.5))
        fig_xai.patch.set_facecolor('#0B0F19')
        shap.plots.waterfall(shap_explanation, show=False)
        ax_xai = plt.gca()
        ax_xai.set_facecolor('#0B0F19')
        st.pyplot(fig_xai, bbox_inches='tight')

        st.markdown("""
            <div style="background-color:rgba(59, 130, 246, 0.1); padding:15px; border-radius:8px; border-left:4px solid #3B82F6; margin-top:20px;">
                <span style="color:#60A5FA; font-weight:600; font-size:14px;">💡 Interpretasi Logika Black-Box:</span><br>
                <span style="color:#CBD5E1; font-size:13px; line-height:1.6;">
                Nilai SHAP positif menunjukkan fitur tersebut <b>menaikkan</b> prediksi risiko dibanding nilai
                dasar (base value) model; nilai negatif berarti fitur tersebut <b>menurunkan</b> risiko.
                Berbeda dari pendekatan koefisien linier sederhana, SHAP memperhitungkan kontribusi
                setiap fitur secara konsisten dan additif terhadap prediksi akhir, sehingga tetap valid
                walau model diganti dengan algoritma non-linier di kemudian hari.
                </span>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("Kritis: File biner model/scaler tidak ditemukan. Pastikan folder `models/` berisi `model_risiko_v1.joblib` dan `scaler_risiko_v1.joblib` berada satu direktori dengan app.py.")
