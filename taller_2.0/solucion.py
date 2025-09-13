
# Proyecto Actividad 2 - Machine Learning Supervisado

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier


# -------------------------------
# 1) Descargar y cargar dataset
# -------------------------------
print("Descargando dataset de Kaggle (o usando caché local)...")
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
print("Ruta del dataset:", path)

# Buscar el archivo CSV dentro del dataset
csv_candidates = [
    os.path.join(path, "heart.csv"),
    os.path.join(path, "Heart.csv"),
]
if not any(os.path.exists(p) for p in csv_candidates):
    found = glob.glob(os.path.join(path, "*.csv"))
    if not found:
        raise FileNotFoundError(f"No se encontró un CSV en {path}.")
    data_path = found[0]
else:
    data_path = next(p for p in csv_candidates if os.path.exists(p))

print("Archivo usado:", data_path)

# Leer dataset
df = pd.read_csv(data_path)

print("\n--- Descripción inicial del dataset ---")
print(f"Registros: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print("\nPrimeras filas:")
print(df.head())
df.info()

# -------------------------------
# 2) Variables predictoras y objetivo
# -------------------------------
target = "HeartDisease"
X = df.drop(columns=[target])
y = df[target].astype(int)

# Separar categóricas y numéricas
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nVariables categóricas: {categorical_features}")
print(f"Variables numéricas: {numerical_features}")

# -------------------------------
# 3) División en train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTamaño train: {X_train.shape[0]}")
print(f"Tamaño test: {X_test.shape[0]}")

# -------------------------------
# 4) Preprocesamiento
# -------------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------------
# 5) Entrenamiento: Random Forest
# -------------------------------
print("\n--- Entrenando modelo Random Forest ---")
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            criterion="gini",
            max_features='sqrt',

            
        )),
    ]
)

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

# -------------------------------
# 6) Evaluación
# -------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- Resultados Random Forest ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

# -------------------------------
# 7) Matriz de confusión
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.tight_layout()
plt.show()