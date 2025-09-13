# Proyecto Actividad 2 - Machine Learning Supervisado

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

tf.random.set_seed(53)
# -------------------------------
# 1) Descargar y cargar dataset
# -------------------------------
print("Descargando dataset de Kaggle (o usando caché local)...")
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
print("Ruta del dataset:", path)

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
y_pred_rf = rf_pipeline.predict(X_test)

# -------------------------------
# 6) Evaluación Random Forest
# -------------------------------
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\n--- Resultados Random Forest ---")
print(f"Accuracy : {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall   : {rec_rf:.4f}")
print(f"F1-score : {f1_rf:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rf, digits=4))

# -----------------------------------------------
# Implementación de la Red Neuronal
# -----------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
input_shape = X_train_processed.shape[1]
print(f"\nNúmero de características tras el preprocesamiento: {input_shape}")


def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

nn_model = create_nn_model(input_shape)
nn_model.summary()

print("\n--- Entrenando modelo de Red Neuronal ---")
history = nn_model.fit(
    X_train_processed,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("Entrenamiento finalizado.")

X_test_processed = preprocessor.transform(X_test)

y_pred_nn_proba = nn_model.predict(X_test_processed)
y_pred_nn = (y_pred_nn_proba > 0.5).astype("int32")

acc_nn = accuracy_score(y_test, y_pred_nn)
prec_nn = precision_score(y_test, y_pred_nn, zero_division=0)
rec_nn = recall_score(y_test, y_pred_nn, zero_division=0)
f1_nn = f1_score(y_test, y_pred_nn, zero_division=0)

print("\n--- Resultados Red Neuronal ---")
print(f"Accuracy : {acc_nn:.4f}")
print(f"Precision: {prec_nn:.4f}")
print(f"Recall   : {rec_nn:.4f}")
print(f"F1-score : {f1_nn:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_nn, digits=4))


# -------------------------------
#  Entrenamiento: Gradient Boosting (modelo investigado)
# -------------------------------
from sklearn.ensemble import GradientBoostingClassifier  # import local para no modificar cabecera

print("\n--- Entrenando modelo Gradient Boosting ---")
gb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=700,     # más árboles (pequeños) en secuencia
            learning_rate=0.005,   # tasa de aprendizaje (más pequeño = más estable)
            max_depth=2,          # profundidad de cada árbol débil
            random_state=42
        )),
    ]
)

gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_test)

# -------------------------------
# 9) Evaluación Gradient Boosting
# -------------------------------
acc_gb = accuracy_score(y_test, y_pred_gb)
prec_gb = precision_score(y_test, y_pred_gb, zero_division=0)
rec_gb = recall_score(y_test, y_pred_gb, zero_division=0)
f1_gb = f1_score(y_test, y_pred_gb, zero_division=0)

print("\n--- Resultados Gradient Boosting ---")
print(f"Accuracy : {acc_gb:.4f}")
print(f"Precision: {prec_gb:.4f}")
print(f"Recall   : {rec_gb:.4f}")
print(f"F1-score : {f1_gb:.4f}")

print("\nReporte de clasificación (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb, digits=4))

# -------------------------------
# 10) Matriz de confusión Gradient Boosting
# -------------------------------
cm_gb = confusion_matrix(y_test, y_pred_gb)

plt.figure()
plt.imshow(cm_gb, interpolation="nearest", cmap=plt.cm.Oranges)
plt.title("Matriz de Confusión - Gradient Boosting")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for i in range(cm_gb.shape[0]):
    for j in range(cm_gb.shape[1]):
        plt.text(j, i, cm_gb[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.show()
