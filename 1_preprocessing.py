import pandas as pd

#load
df_raw = pd.read_csv('url_features_extracted1.csv', sep=';', low_memory=False)
total_awal = len(df_raw)

#seleksi dan normalisasi
df = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')].copy()
if 'url_entropy' in df.columns:
    df = df.drop(columns=['url_entropy'])

df['URL'] = df['URL'].astype(str).str.lower().str.strip().str.rstrip('/')

#hitung ulang persentasi
def fix_percentage(url):
    url_str = str(url)
    if len(url_str) == 0: return 0
    digit_count = sum(c.isdigit() for c in url_str)
    return round((digit_count / len(url_str)) * 100, 2)

df['percentage_numeric_chars'] = df['URL'].apply(fix_percentage)

#cleaning numerik
numerical_cols = df.columns.drop(['URL', 'percentage_numeric_chars', 'ClassLabel'])
def clean_multi_dots(val):
    val_str = str(val).strip()
    if val_str.count('.') > 1:
        return val_str.replace('.', '')
    return val_str

for col in numerical_cols:
    df[col] = df[col].apply(clean_multi_dots)
    df[col] = pd.to_numeric(df[col], errors='coerce')

#stat pembuangan data
data_setelah_konversi = len(df)
df = df.dropna()
data_setelah_dropna = len(df)

# filter class label
df = df[df['ClassLabel'].astype(str).isin(['0', '1', '0.0', '1.0'])]
data_setelah_label_fix = len(df)

#hapus duplikat 
df_final = df.drop_duplicates().copy()
total_akhir = len(df_final)


distribusi = df_final['ClassLabel'].astype(int).value_counts()

print(" LAPORAN STATISTIK PRA-PEMROSESAN DATA")
print(f"Total Data Mentah (Awal)      : {total_awal} baris")
print(f"Data Dibuang (Missing/NaN)    : {data_setelah_konversi - data_setelah_dropna} baris")
print(f"Data Dibuang (Label Tidak Sah): {data_setelah_dropna - data_setelah_label_fix} baris")
print(f"Data Dibuang (Duplikat)       : {data_setelah_label_fix - total_akhir} baris")
print(f" TOTAL DATA BERSIH (AKHIR)     : {total_akhir} baris")
print("DISTRIBUSI KELAS AKHIR:")
print(f"   - Phishing (0)    : {distribusi.get(0, 0)} baris")
print(f"   - Legitimate (1)  : {distribusi.get(1, 0)} baris")

df_final['ClassLabel'] = df_final['ClassLabel'].astype(int)
df_final.to_csv('dataset_preprocessing_4_2.csv', index=False)