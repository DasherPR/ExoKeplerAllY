from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import StringIO
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

app = Flask(__name__)

# Parámetros de tu repo
REPO_OWNER = "DasherPR"
REPO_NAME = "ExoKeplerAllY"

# URLs para descargar artefactos
urls = {
    "model": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/best_deep_model.h5",
    "scaler": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/scaler.pkl",
    "label_map": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/label_map.pkl",
    "dataset": f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/planets.csv"
}

# -----------------------------
# Descarga y carga inicial
# -----------------------------
def download_if_missing():
    for name, url in urls.items():
        if name == "dataset":
            continue
        filename = url.split("/")[-1]
        if not os.path.exists(filename):
            print(f"Descargando {filename} ...")
            r = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{filename} descargado.")

download_if_missing()

print("Cargando artefactos...")
model = load_model("best_deep_model.h5")
scaler = joblib.load("scaler.pkl")
label_map = joblib.load("label_map.pkl")
inv_label_map = {v: k for k, v in label_map.items()}

print("Artefactos cargados correctamente ✅")

# -----------------------------
# Cargar dataset y calcular métricas
# -----------------------------
def calcular_metricas():
    print("Descargando dataset para métricas...")
    r = requests.get(urls["dataset"])
    df = pd.read_csv(StringIO(r.text))
    df = df[df["koi_disposition"] != "CANDIDATE"]

    X_test = df.drop("koi_disposition", axis=1)
    y_test = df["koi_disposition"].map(label_map).to_numpy()

    X_test_std = scaler.transform(X_test)
    y_pred = (model.predict(X_test_std) > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }

# Cachear métricas al iniciar
metricas_globales = calcular_metricas()

# -----------------------------
# Rutas de la API
# -----------------------------
@app.route('/')
def home():
    return jsonify({
        "message": "Bienvenido a ExoKeplerAllY API",
        "status": "API activa",
        "endpoints": {
            "/metrics": "Devuelve las métricas del modelo actual",
            "/predict": "Envía 12 valores de entrada para predecir si es exoplaneta"
        }
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(metricas_globales)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validar que hay 12 entradas
        if len(data) != 12:
            return jsonify({"error": "Se esperaban 12 variables numéricas"}), 400

        df = pd.DataFrame([data])
        X_std = scaler.transform(df)
        y_pred = (model.predict(X_std) > 0.5).astype(int).flatten()[0]
        result = inv_label_map[y_pred]

        return jsonify({
            "prediction": result,
            "details": {
                "numeric_output": int(y_pred)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
