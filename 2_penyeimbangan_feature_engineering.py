import pandas as pd
from urllib.parse import urlparse

#load data preprocessing
df = pd.read_csv('dataset_preprocessing_4_2.csv')

# Pisahkan kelas
df_phish = df[df['ClassLabel'] == 0]
df_legit = df[df['ClassLabel'] == 1]

# Downsamppling biar rata
df_phish_downsampled = df_phish.sample(n=len(df_legit), random_state=42)

#gabung dan acak
df_balanced = pd.concat([df_phish_downsampled, df_legit]).sample(frac=1, random_state=42).reset_index(drop=True)

# tambah 2 fitur
def extract_custom_features(url):
    url_str = str(url).lower()
    
    # jumlah sub-page (garis miring '/')
    n_slash = url_str.count('/')
    
    # deteksi 'http' atau 'https' di domain
    try:
        domain = urlparse(url_str).netloc
        http_in_domain = 1 if ('http' in domain or 'https' in domain) else 0
    except:
        http_in_domain = 0
        
    return pd.Series([n_slash, http_in_domain])

print(" Mengekstraksi fitur tambahan: n_slash dan http_in_domain...")
df_balanced[['n_slash', 'http_in_domain']] = df_balanced['URL'].apply(extract_custom_features)

print(" LAPORAN BAB 4.3: BALANCING & FEATURE ENGINEERING")
print(f"1. Jumlah Data Setelah Balancing : {len(df_balanced)} baris")
print(f"   - Phishing (0)                : {len(df_balanced[df_balanced['ClassLabel']==0])}")
print(f"   - Legitimate (1)              : {len(df_balanced[df_balanced['ClassLabel']==1])}")
print(f"2. Total Fitur Saat Ini          : {len(df_balanced.columns) - 2} fitur + URL + Label") 
print(f"   (17 Fitur Bawaan + 2 Fitur Kustom)")

#dataset final
df_balanced.to_csv('dataset_final_pemodelan_4_3.csv', index=False)
print(" Dataset final disimpan: 'dataset_final_pemodelan_4_3.csv'")