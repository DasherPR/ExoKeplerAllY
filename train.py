# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib

# ---------- Cargar datos ----------
df = pd.read_csv('planets.csv')
df = df.fillna(df.mean(numeric_only=True))
df = df.drop(df[df["koi_disposition"] == "CANDIDATE"].index)

# ---------- Separar features y etiquetas ----------
X = df.drop('koi_disposition', axis=1)
y = df['koi_disposition']

# ---------- Balancear clases ----------
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# ---------- Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# ---------- Estandarización ----------
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Guardar el scaler
joblib.dump(sc, "scaler.pkl")

# ---------- Mapear etiquetas a 0/1 ----------
label_map = {"CONFIRMED":1, "FALSE POSITIVE":0}
y_train = pd.Series(y_train).map(label_map).to_numpy()
y_test = pd.Series(y_test).map(label_map).to_numpy()

# Guardar label map
joblib.dump(label_map, "label_map.pkl")

# ---------- Crear modelo ----------
model = Sequential([
    Dense(250, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=5e-4), loss="binary_crossentropy", metrics=["accuracy"])

# ---------- Callbacks ----------
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
mc = ModelCheckpoint("best_deep_model.h5", save_best_only=True)

callbacks = [es, rlr, mc]

# ---------- Entrenamiento ----------
history = model.fit(
    X_train_std, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ---------- Evaluación ----------
loss, acc = model.evaluate(X_test_std, y_test, verbose=0)
print(f"Pérdida en test: {loss:.4f}")
print(f"Precisión en test: {acc:.4f}")

