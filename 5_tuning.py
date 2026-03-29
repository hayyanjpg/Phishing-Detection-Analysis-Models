import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Load Data (Pastikan data sudah dalam kondisi scaled)
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(" Memulai Hyperparameter Tuning (Grid Search)...")
print("Catatan: Proses ini mungkin memakan waktu beberapa menit.")

# 2. Pengaturan Parameter Grid
# Logistic Regression
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs']
}

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# SVM (Linear)
param_grid_svm = {
    'C': [0.1, 1, 10]
}

# 3. Eksekusi Tuning
tuning_configs = [
    ("Logistic Regression", LogisticRegression(max_iter=1000), param_grid_lr),
    ("Random Forest", RandomForestClassifier(random_state=42), param_grid_rf),
    ("SVM", SVC(kernel='linear', random_state=42), param_grid_svm)
]

best_models = {}

for name, model, params in tuning_configs:
    print(f" Tuning {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train_scaled, y_train)
    
    best_models[name] = grid.best_estimator_
    print(f" Best Params {name}: {grid.best_params_}")
    print(f" Best Cross-Val Score: {grid.best_score_:.4f}")
    
    # Simpan model hasil tuning
    joblib.dump(grid.best_estimator_, f'models/tuned_{name.lower().replace(" ", "_")}.pkl')

print("\n Semua model telah di-tuning dan disimpan!")