# api.py
from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Cargar artefactos
model = load_model("best_deep_model.h5")
scaler = joblib.load("scaler.pkl")
label_map = joblib.load("label_map.pkl")

@app.route('/')
def home():
    return "✅ API Working"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    values = np.array(data["values"]).reshape(1, -1)
    values_scaled = scaler.transform(values)
    pred = (model.predict(values_scaled) > 0.5).astype(int).flatten()[0]
    label = [k for k, v in label_map.items() if v == pred][0]
    return jsonify({"prediction": label})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

