import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset_final_pemodelan_4_3.csv')

#hapus url dan classlabel
X = df.drop(columns=['URL', 'ClassLabel'])
y = df['ClassLabel']

# random_state=42 agar hasil split konsisten kalau run ulang
# stratify=y agar proporsi seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)#0.20 karena akan mengambil 20 % untuk test

print(" LAPORAN BAB 4.4: PEMBAGIAN DATA")
print(f"Total Data Keseluruhan : {len(df)} baris")
print(f"Rasio Pembagian        : 80% Latih, 20% Uji")
print(f"1. DATA LATIH (Training Set)")
print(f"   - Jumlah Baris      : {len(X_train)}")
print(f"   - Phishing (0)      : {sum(y_train == 0)}")
print(f"   - Legitimate (1)    : {sum(y_train == 1)}")
print(f"2. DATA UJI (Testing Set)")
print(f"   - Jumlah Baris      : {len(X_test)}")
print(f"   - Phishing (0)      : {sum(y_test == 0)}")
print(f"   - Legitimate (1)    : {sum(y_test == 1)}")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Done")