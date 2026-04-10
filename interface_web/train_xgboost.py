"""
train_xgboost.py
----------------
Entraîne le modèle XGBoost sur le dataset LIAR (mêmes paramètres que
Notebook/03.5_Modelisation_pipeline.ipynb) et exporte :
  - model.pkl       : XGBClassifier entraîné
  - vectorizer.pkl  : {"tfidf": TfidfVectorizer, "scaler": StandardScaler}

Usage :
    python train_xgboost.py
    python train_xgboost.py --data-dir LIAR_DATA_SET/02_stg --out-dir .
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration (identique au notebook 03.5)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_fire_counts", "party_encoded",
]
TEXT_FEATURE = "clean_statement"
TARGET = "label_binary"

TFIDF_PARAMS = dict(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
)

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path):
    train = pd.read_csv(data_dir / "train_clean.csv")
    valid = pd.read_csv(data_dir / "valid_clean.csv")
    test  = pd.read_csv(data_dir / "test_clean.csv")
    print(f"Train : {train.shape} | Valid : {valid.shape} | Test : {test.shape}")
    return train, valid, test


def prepare(df: pd.DataFrame):
    df = df.copy()
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    return df


def build_features(tfidf, scaler, df: pd.DataFrame, fit: bool = False):
    text = df[TEXT_FEATURE]
    num  = df[NUMERIC_FEATURES].values
    if fit:
        X_tfidf = tfidf.fit_transform(text)
        X_num   = scaler.fit_transform(num)
    else:
        X_tfidf = tfidf.transform(text)
        X_num   = scaler.transform(num)
    return hstack([X_tfidf, csr_matrix(X_num)])


def evaluate(model, X, y, split_name: str):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="weighted")
    auc = roc_auc_score(y, y_prob)
    print(f"  [{split_name}]  Accuracy={acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    return {"accuracy": acc, "f1_weighted": f1, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    train, valid, test = load_splits(data_dir)
    train, valid, test = prepare(train), prepare(valid), prepare(test)

    y_train = train[TARGET].values
    y_valid  = valid[TARGET].values
    y_test   = test[TARGET].values

    # 2. Vectorisation
    tfidf  = TfidfVectorizer(**TFIDF_PARAMS)
    scaler = StandardScaler()

    X_train = build_features(tfidf, scaler, train, fit=True)
    X_valid  = build_features(tfidf, scaler, valid)
    X_test   = build_features(tfidf, scaler, test)
    print(f"Shape features : {X_train.shape}")

    # 3. Entraînement XGBoost
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nscale_pos_weight : {spw:.3f}")

    model = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )

    # 4. Évaluation
    print("\n--- Métriques ---")
    evaluate(model, X_valid, y_valid, "Valid")
    evaluate(model, X_test,  y_test,  "Test")

    # 5. Export
    model_path = out_dir / "model.pkl"
    vec_path   = out_dir / "vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vec_path, "wb") as f:
        pickle.dump({"tfidf": tfidf, "scaler": scaler}, f)

    print(f"\nExport OK :")
    print(f"  {model_path}")
    print(f"  {vec_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne XGBoost et exporte les artefacts.")
    parser.add_argument(
        "--data-dir",
        default="LIAR_DATA_SET/02_stg",
        help="Dossier contenant train_clean.csv, valid_clean.csv, test_clean.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Dossier de sortie pour model.pkl et vectorizer.pkl",
    )
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.out_dir))
