from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from scipy.sparse import hstack, csr_matrix
import numpy as np
import os

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Charger le modèle et le vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vec = pickle.load(f)

# ✅ Schéma d’entrée API
class PredictionInput(BaseModel):
    text: str
    author: str | None = None
    orientation: str | None = None

@app.post("/predict")
def predict(input: PredictionInput):
    # 1️⃣ TEXTE → TF-IDF
    X_text = vec["tfidf"].transform([input.text])

    # 2️⃣ FEATURES NUMÉRIQUES
    # 👉 pour l’instant, on met des zéros (dummy values)
    numeric_features = np.zeros(6)
    X_num = csr_matrix(vec["scaler"].transform([numeric_features]))

    # 3️⃣ CONCATÉNATION
    X = hstack([X_text, X_num])

    # 4️⃣ PRÉDICTION
    label = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0].max())

    return {
        "label": label,
        "confidence": confidence
    }