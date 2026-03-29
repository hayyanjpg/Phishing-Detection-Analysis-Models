import pandas as pd
import time
import joblib # Pustaka untuk menyimpan model
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Load Data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# 2. Scaling Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan Scaler agar bisa dipakai saat testing URL baru nanti
joblib.dump(scaler, 'scaler_model.pkl')
print(" Scaler berhasil disimpan sebagai 'scaler_model.pkl'")

# 3. Inisialisasi Model
models = {
    "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(kernel='linear', random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Membuat folder untuk menyimpan model jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

results = []

print("\n Memulai Pelatihan dan Penyimpanan Model...")
for name, model in models.items():
    start_time = time.time()
    
    # Latih Model
    model.fit(X_train_scaled, y_train)
    
    # Simpan Model setelah dilatih
    model_filename = f'models/{name.lower()}_model.pkl'
    joblib.dump(model, model_filename)
    
    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    training_time = end_time - start_time
    
    results.append({
        "Algoritma": name,
        "Akurasi": acc,
        "Waktu (s)": training_time,
        "Saved_Path": model_filename
    })
    print(f" {name} selesai ({training_time:.2f}s) -> Tersimpan di {model_filename}")

# 4. Tampilkan Tabel Ringkas
df_results = pd.DataFrame(results)
print("\n" + "="*60)
print(df_results[['Algoritma', 'Akurasi', 'Waktu (s)']].to_string(index=False))
print("="*60)