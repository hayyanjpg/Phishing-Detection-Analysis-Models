import pandas as pd
import joblib
import re
from urllib.parse import urlparse

# =====================================================
# LOAD MODEL DAN SCALER
# =====================================================
model = joblib.load("models/random_forest_model.pkl") 
scaler = joblib.load("scaler_model.pkl")
X_train_columns = pd.read_csv("X_train.csv").columns.tolist()

# =====================================================
# DAFTAR PENDUKUNG (TLD Populer yang Lebih Lengkap)
# =====================================================
popular_tlds = [
    "com", "org", "net", "edu", "gov", "mil", "int", # GTLD
    "id", "co.id", "ac.id", "go.id", "uk", "us", "co", "io", "me", "tv", "info" # CCTLD & Populer
]

suspicious_ext = [".exe",".zip",".rar",".scr",".bat",".cmd",".js",".jar",".vbs"]

# =====================================================
# FUNGSI EKSTRAKSI FITUR (KOREKSI DEFINISI)
# =====================================================
def extract_features(url):
    # 1. Normalisasi
    url_str = str(url).lower().strip().rstrip('/')
    parsed = urlparse(url_str)
    
    # Ambil netloc (domain), jika kosong (url tanpa http), coba parsing ulang
    if not parsed.netloc and url_str:
        parsed = urlparse("http://" + url_str)
        
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {}

    # --- 1-4. Dasar ---
    features["url_length"] = len(url_str)
    
    # Deteksi IP (Regex lebih ketat)
    is_ip = 1 if re.search(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain.split(':')[0]) else 0
    features["has_ip_address"] = is_ip
    
    features["dot_count"] = url_str.count(".")
    features["https_flag"] = 1 if url_str.startswith("https") else 0
    
    # --- 5. Token Count ---
    # Memisahkan berdasarkan simbol non-alphanumeric
    tokens = re.split(r"[./\-?=&_]", url_str)
    features["token_count"] = len([t for t in tokens if t])

    # --- 6. Subdomain Count (KOREKSI) ---
    if is_ip:
        features["subdomain_count"] = 0
    else:
        # Menghitung titik di domain. 
        # google.com -> dots=1 -> subdomain=0
        # www.google.com -> dots=2 -> subdomain=1
        dot_in_domain = domain.count(".")
        features["subdomain_count"] = max(0, dot_in_domain - 1)

    # --- 7-10. Domain & Path ---
    features["query_param_count"] = len(query.split("&")) if query else 0
    
    # Ambil TLD (Bagian terakhir setelah titik)
    parts = domain.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    
    features["tld_length"] = len(tld)
    features["path_length"] = len(path)
    features["has_hyphen_in_domain"] = 1 if "-" in domain else 0

    # --- 11-15. Statistik & Keamanan ---
    digit_count = sum(c.isdigit() for c in url_str)
    features["number_of_digits"] = digit_count
    
    # TLD Popularity (KOREKSI)
    # Mengecek apakah TLD atau akhiran domain ada di daftar populer
    features["tld_popularity"] = 1 if any(domain.endswith("." + p) for p in popular_tlds) else 0
    
    features["suspicious_file_extension"] = 1 if any(ext in url_str for ext in suspicious_ext) else 0
    features["domain_name_length"] = len(domain)
    
    # Percentage Numeric (Mendeley biasanya 0-100)
    features["percentage_numeric_chars"] = (digit_count / len(url_str)) * 100 if len(url_str) > 0 else 0

    # --- 16-17. Fitur Kustom Penelitian ---
    features["n_slash"] = url_str.count("/")
    # Penting: cek teks 'http' hanya di dalam domain (bukan di awal protokol)
    features["http_in_domain"] = 1 if "http" in domain else 0

    return features

# =====================================================
# EKSEKUSI
# =====================================================
print("\n" + "="*40)
url_input = input("🔗 Masukkan URL: ")

# Proses
features_dict = extract_features(url_input)
df_input = pd.DataFrame([features_dict])
df_input = df_input.reindex(columns=X_train_columns, fill_value=0)

# Scaling & Prediksi
X_scaled = scaler.transform(df_input)
prediction = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0]

# Tampilkan Hasil
print("-" * 40)
if prediction == 0:
    print(f"🚩 HASIL: PHISHING")
    print(f"Confidence: {proba[0]*100:.2f}%")
else:
    print(f"✅ HASIL: LEGITIMATE")
    print(f"Confidence: {proba[1]*100:.2f}%")
print("-" * 40)