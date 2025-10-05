# api.py
import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
from io import StringIO
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# -----------------------------
# 1️⃣ Configuración básica
# -----------------------------
app = Flask(__name__)

REPO_OWNER = "DasherPR"
REPO_NAME = "ExoKeplerAllY"

MODEL_FILES = {
    "best_deep_model.h5": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/best_deep_model.h5",
    "scaler.pkl": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/scaler.pkl",
    "label_map.pkl": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/label_map.pkl",
}

# -----------------------------
# 2️⃣ Descargar artefactos si no existen
# -----------------------------
def ensure_files_downloaded():
    for filename, url in MODEL_FILES.items():
        if not os.path.exists(filename):
            print(f"Descargando {filename} desde {url} ...")
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)
            print(f"{filename} descargado correctamente")
        else:
            print(f"{filename} ya existe, omitiendo descarga")

# -----------------------------
# 3️⃣ Cargar modelo y utilidades
# -----------------------------
def load_artifacts():
    global model, scaler, label_map
    ensure_files_downloaded()

    print("Cargando modelo y utilidades...")
    model = load_model("best_deep_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_map = joblib.load("label_map.pkl")

    # Invertir mapa de etiquetas
    global inv_label_map
    inv_label_map = {v: k for k, v in label_map.items()}
    print("Modelo y escalador cargados correctamente")

# -----------------------------
# 4️⃣ Endpoint de prueba
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API activa", "message": "Bienvenido a ExoKeplerAllY API"}), 200

# -----------------------------
# 5️⃣ Endpoint para predicciones
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener datos del cuerpo de la solicitud
        data = request.get_json()
        if not data or "values" not in data:
            return jsonify({"error": "Debes enviar un JSON con la clave 'values' y una lista de 15 números."}), 400

        features = np.array(data["values"]).reshape(1, -1)

        # Escalar y predecir
        features_scaled = scaler.transform(features)
        pred = (model.predict(features_scaled) > 0.5).astype(int).flatten()[0]
        label = inv_label_map[pred]

        return jsonify({
            "prediction": int(pred),
            "label": label,
            "message": f"El objeto parece ser: {label}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 6️⃣ Iniciar servidor
# -----------------------------
if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000)
