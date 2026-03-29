import streamlit as st
import pandas as pd
import joblib
import re
from urllib.parse import urlparse

#konfigurasi halaman dan style
st.set_page_config(page_title="Phishing URL Analyzer", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stAlert { border-radius: 10px; }
    .main-title { color: #1E3A8A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

#load model dan assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/random_forest_model.pkl")
        scaler = joblib.load("scaler_model.pkl")
        X_train_cols = pd.read_csv("X_train.csv").columns.tolist()
        return model, scaler, X_train_cols
    except Exception as e:
        st.error(f"Gagal memuat model/data. Pastikan file tersedia di GitHub. Error: {e}")
        return None, None, None

model, scaler, X_train_columns = load_assets()

#fungsi ekstraksi 17 fitur
def extract_features(url):
    url_str = str(url).lower().strip().rstrip('/')
    parsed = urlparse(url_str)
    if not parsed.netloc and url_str:
        parsed = urlparse("http://" + url_str)
    
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Daftar TLD & Ext Suspicious
    popular_tlds = ["com", "org", "net", "edu", "gov", "id", "co.id", "ac.id", "go.id", "io", "me"]
    suspicious_ext = [".exe", ".zip", ".rar", ".bat", ".cmd", ".js"]

    f = {}
    f["url_length"] = len(url_str)
    is_ip = 1 if re.search(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain.split(':')[0]) else 0
    f["has_ip_address"] = is_ip
    f["dot_count"] = url_str.count(".")
    f["https_flag"] = 1 if url_str.startswith("https") else 0
    f["token_count"] = len([t for t in re.split(r"[./\-?=&_]", url_str) if t])
    f["subdomain_count"] = 0 if is_ip else max(0, domain.count(".") - 1)
    f["query_param_count"] = len(query.split("&")) if query else 0
    
    tld = domain.split(".")[-1] if "." in domain else ""
    f["tld_length"] = len(tld)
    f["path_length"] = len(path)
    f["has_hyphen_in_domain"] = 1 if "-" in domain else 0
    
    digit_count = sum(c.isdigit() for c in url_str)
    f["number_of_digits"] = digit_count
    f["tld_popularity"] = 1 if any(domain.endswith("." + p) for p in popular_tlds) else 0
    f["suspicious_file_extension"] = 1 if any(ext in url_str for ext in suspicious_ext) else 0
    f["domain_name_length"] = len(domain)
    f["percentage_numeric_chars"] = (digit_count / len(url_str)) * 100 if len(url_str) > 0 else 0
    
    # Fitur Kustom
    f["n_slash"] = url_str.count("/")
    f["http_in_domain"] = 1 if "http" in domain else 0

    return f

#fungsi intrepetasi 
def get_explanation(prediction, proba, f):
    # Ambil nilai probabilitas untuk kelas terpilih
    conf = proba[1] * 100 if prediction == 1 else proba[0] * 100
    
    # Analisis Kepercayaan
    if conf > 85:
        level = "Sangat Tinggi"
        summary = "Model mendeteksi pola yang sangat kuat dan konsisten dengan database serangan phishing/situs resmi."
    elif conf > 70:
        level = "Tinggi"
        summary = "Model cukup yakin, namun terdapat beberapa elemen kecil yang tidak biasa pada struktur URL."
    else:
        level = "Rendah (Ambigu)"
        summary = "URL ini memiliki karakteristik campuran. Beberapa fitur terlihat aman, namun ada elemen yang menyerupai taktik phishing."

    # Analisis Fitur Pemicu
    reasons = []
    if f['https_flag'] == 0: reasons.append(" Tidak menggunakan HTTPS")
    if f['http_in_domain'] == 1: reasons.append(" Domain mengandung teks 'http'")
    if f['n_slash'] > 5: reasons.append(" Struktur direktori terlalu dalam (banyak slash)")
    if f['percentage_numeric_chars'] > 20: reasons.append(" Persentase angka terlalu tinggi")
    if f['has_ip_address'] == 1: reasons.append(" Menggunakan IP Address sebagai pengganti Nama Domain")
    
    trigger_text = " | ".join(reasons) if reasons else "Struktur URL terlihat mengikuti standar umum."
    
    return conf, level, summary, trigger_text

#UI
st.markdown("<h1 class='main-title'>🛡️ Detektor URL Phishing</h1>", unsafe_allow_html=True)
st.write("Sistem berbasis Artificial Intelligence untuk mendeteksi potensi ancaman pada tautan website.")
st.divider()

url_input = st.text_input("🔗 Masukkan URL Website:", placeholder="https://www.example.com")

if st.button("Analisis Tautan"):
    if url_input:
        if model is not None:
            # 1. Ekstraksi
            features = extract_features(url_input)
            df = pd.DataFrame([features]).reindex(columns=X_train_columns, fill_value=0)
            
            # 2. Prediksi
            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]
            
            # 3. Interpretasi
            conf, level, summary, triggers = get_explanation(prediction, proba, features)

            # 4. Tampilkan Hasil
            st.subheader("Hasil Analisis:")
            if prediction == 0:
                st.error(" STATUS: PHISHING (BERBAHAYA)")
            else:
                st.success("  STATUS: LEGITIMATE (AMAN)")
            
            # Penjelasan Detail
            with st.container():
                col1, col2 = st.columns(2)
                col1.metric("Confidence Score", f"{conf:.2f}%")
                col2.metric("Tingkat Keyakinan", level)
                
                st.markdown(f"**Mengapa demikian?** {summary}")
                st.markdown(f"**Analisis Struktur:** {triggers}")
                
                st.info("""
                **Catatan Akademik:** Skor Confidence mencerminkan persentase 'Voting' dari 100+ pohon keputusan dalam model Random Forest. 
                Hasil di bawah 70% menandakan adanya Feature Overlap antara situs resmi dan situs phishing.
                """)
        else:
            st.error("Model tidak tersedia. Pastikan proses training sudah selesai.")
    else:
        st.warning("Silakan masukkan URL terlebih dahulu!")

st.divider()
st.caption("Penelitian Skripsi - Analisis Perbandingan Algoritma Machine Learning untuk Deteksi Phishing.")
