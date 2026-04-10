# Projet Spé 3 — Détection de Fake News Politiques (LIAR Dataset)

## Objectif

Développer un pipeline complet de détection de fake news politiques à partir du **LIAR Dataset**, en passant par l'exploration des données, le preprocessing, et la modélisation.

---

## Structure du projet

```
PROJET_SPE_3_FAKE_NEW/
├── LIAR_DATA_SET/
│   ├── 01_raw/              ← Données brutes originales (TSV)
│   ├── 02_stg/              ← Données nettoyées et preprocessées
│   └── 03_models/           ← Modèles entraînés (TF‑IDF, BERT)
│
├── Notebook/
│   ├── 01_EDA_preprocessing.ipynb       ← EDA complète + preprocessing
│   ├── 02_TF_IDF_ML.ipynb               ← Pipeline TF‑IDF non supervisé (SVD, KMeans)
│   ├── 03.5_Modelisation_pipeline.ipynb ← Modèles supervisés (LR, RF, XGBoost)
│   ├── 03_Bert_pipeline.ipynb           ← Pipeline CamemBERT binaire
│   ├── 04_model_test_01.ipynb           ← Test XGBoost sur Fake/True (data_set_random_01)
│   ├── 05_model_test_02.ipynb           ← Test XGBoost sur BuzzFeed (data_set_random_02)
│   └── 06_model_test_03.ipynb           ← Test XGBoost sur GossipCop (data_set_random_03)
│
├── data_set_random_01/
│   ├── Fake.csv                 ← 23 481 articles fake
│   └── True.csv                 ← 21 417 articles vrais
│
├── data_set_random_02/
│   ├── BuzzFeed_fake_news_content.csv  ← 91 articles fake
│   └── BuzzFeed_real_news_content.csv  ← 91 articles vrais
│
├── data_set_random_03/
│   ├── gossipcop_fake.csv       ← 5 323 articles fake
│   └── gossipcop_real.csv       ← 16 817 articles vrais
│
├── interface_web/
│   ├── train_xgboost.py     ← Script d'entraînement + export des artefacts
│   ├── model.pkl            ← XGBClassifier entraîné (généré par train_xgboost.py)
│   └── vectorizer.pkl       ← {"tfidf": TfidfVectorizer, "scaler": StandardScaler}
│
├── requirements.txt
└── README.md
```

---

## Dataset

Le **LIAR Dataset** (Wang, 2017) contient ~12 800 déclarations politiques issues de PolitiFact, annotées selon 6 niveaux de véracité :

| Label | Signification |
|---|---|
| `pants-fire` | Totalement faux |
| `false` | Faux |
| `barely-true` | À peine vrai |
| `half-true` | À moitié vrai |
| `mostly-true` | Majoritairement vrai |
| `true` | Vrai |

Chaque entrée contient 14 colonnes : identifiant, label, texte de la déclaration, sujet, orateur, titre, état, parti politique, 5 compteurs historiques, et contexte.

---

## Grandes étapes du projet

### Étape 1 — Chargement et exploration des données brutes

- Chargement des splits `train.tsv`, `test.tsv`, `valid.tsv` depuis `01_raw/`
- Assignation manuelle des 14 noms de colonnes
- Vérification des shapes, types et premières lignes

### Étape 2 — EDA complète (Exploratory Data Analysis)

- **Distribution des labels** : barplots par split, calcul du ratio de déséquilibre
- **Analyse des textes** : longueur des statements (mots/caractères), top 20 mots fréquents, wordclouds par label
- **Analyse des métadonnées** : top speakers, distribution par parti politique, heatmap parti × label, sujets fréquents
- **Compteurs historiques** : distribution des 5 compteurs de véracité passée, corrélation avec le label final
- **Valeurs manquantes** : heatmap et pourcentages de nulls par colonne

### Étape 3 — Preprocessing

- **Nettoyage du texte** : lowercase, suppression ponctuation/caractères spéciaux (regex), suppression stopwords (NLTK), lemmatisation → colonne `clean_statement`
- **Mapping des labels** :
  - `label_binary` : 0 = fake (pants-fire, false, barely-true) / 1 = real (half-true, mostly-true, true)
  - `label_3class` : 0 = faux / 1 = mixte / 2 = vrai
- **Encodage des métadonnées** : parti politique encodé numériquement
- **Vérification** : shape avant/après, contrôle des nulls, exemples comparatifs

### Étape 4 — Export vers 02_stg/

- Export des DataFrames nettoyés : `train_clean.csv`, `test_clean.csv`, `valid_clean.csv`
- Chaque fichier contient toutes les colonnes originales + `clean_statement`, `label_binary`, `label_3class`


### Étape 5 — Pipeline TF‑IDF non supervisé (Notebook 02_TF_IDF_ML)

Exploration des méthodes classiques de représentation vectorielle sans supervision :

- Vectorisation TF‑IDF + réduction SVD (LSA)
- Clustering K‑Means (k=2 et k=3)
- Similarité cosinus avec le centre des déclarations vraies
- Variante avec méta‑features (one-hot speaker/parti/sujet)

**Résultats :** ARI ≈ 0.009–0.012 dans tous les scénarios — le vocabulaire seul ne sépare pas vrai/faux. Le silhouette score monte à 0.243 avec les méta-features (clusters cohérents thématiquement) mais reste décorrélé des labels. Conclusion : la véracité n'est pas un phénomène lexical.

---

### Étape 5.5 — Modélisation supervisée (Notebook 03.5_Modelisation_pipeline)

Pipeline supervisé combinant TF‑IDF (14 777 features, bigrammes) et features méta numériques (5 compteurs historiques + `party_encoded`) :

- **Logistic Regression** (`class_weight="balanced"`)
- **Random Forest** (300 arbres, `class_weight="balanced"`)
- **XGBoost** (500 arbres, `scale_pos_weight` calculé, `tree_method="hist"`)

**Résultats sur le jeu de test :**

| Modèle | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.620 | 0.621 | 0.663 |
| Random Forest       | **0.732** | **0.731** | 0.792 |
| XGBoost             | 0.730 | 0.731 | **0.818** |

**Conclusions :**
- La LR confirme que TF‑IDF seul est insuffisant (62% accuracy)
- Random Forest et XGBoost atteignent ~73% grâce aux **compteurs de crédibilité** (`mostly_true_counts`, `false_counts`, etc.) qui dominent les feature importances — l'historique du locuteur est le signal prédictif dominant
- XGBoost a le meilleur AUC (0.818) : meilleure calibration des probabilités
- Plafond atteint à ~73% avec ces features — justifie le recours à BERT pour la sémantique contextuelle

---

### Étape 5.7 — Tests de généralisation : modèle LIAR sur datasets externes (Notebooks 04 / 05 / 06)

Évaluation de la **généralisation** du modèle XGBoost pré-entraîné sur LIAR, appliqué à trois datasets externes sans réentraînement.

**Méthode commune :**
- Chargement du modèle pré-entraîné : `interface_web/model.pkl` + `interface_web/vectorizer.pkl`
- Concaténation de toutes les colonnes texte (sauf `label`) en un seul champ
- **Preprocessing identique à LIAR** : lowercase → suppression ponctuation → suppression stopwords NLTK → lemmatisation WordNet
- `tfidf.transform()` avec le vocabulaire LIAR (sans réentraînement du TF-IDF)
- Features numériques à **zéro** (pas d'historique locuteur dans ces datasets)
- Évaluation : Accuracy, F1 weighted, ROC-AUC, matrice de confusion

| Notebook | Dataset | Articles | Labels |
|---|---|---|---|
| `04_model_test_01.ipynb` | Fake.csv / True.csv | 44 898 | 0=Fake / 1=True |
| `05_model_test_02.ipynb` | BuzzFeed | 182 | 0=Fake / 1=Real |
| `06_model_test_03.ipynb` | GossipCop | 22 140 | 0=Fake / 1=Real |

**Pourquoi les scores sont inférieurs au LIAR :**

Les résultats sur les datasets externes sont significativement plus bas que sur le LIAR test set (73% accuracy, 0.818 AUC). Plusieurs facteurs structurels expliquent cette dégradation :

1. **Domain shift** : LIAR contient des déclarations politiques courtes (~20 mots) issues de PolitiFact. Les datasets externes sont des articles complets (centaines de mots, style journalistique ou people). Le vocabulaire TF-IDF appris sur LIAR ne couvre pas les tokens des articles longs.
2. **Features numériques à zéro** : les compteurs d'historique locuteur (`barely_true_counts`, `false_counts`, etc.) sont les features les plus importantes du modèle (~top 4 des importances). Les mettre à zéro biaise massivement les prédictions — le modèle les interprète comme un locuteur "neutre inconnu" et tend à prédire systématiquement "Real".
3. **Définition des labels différente** : LIAR annote des _claims_ individuels vérifiés par fact-checkers ; les datasets externes classifient des articles entiers par source, sans vérification claim-by-claim.
4. **Domaines différents** : BuzzFeed = clickbait politique (plus proche de LIAR), GossipCop = presse people/entertainment (très éloigné du domaine politique LIAR).

---

### Étape 6 — Pipeline BERT (CamemBERT) — Classification en trois parties puis approche binaire (Notebook 03_Bert_pipeline)

- Tokenisation CamemBERT
- Dataset PyTorch personnalisé
- Entraînement avec AdamW
- Grid Search sur :
- learning rate
- batch size
- dropout
- Sauvegarde automatique du meilleur modèle dans 03_models/

---

## Installation

```bash
pip install -r requirements.txt
```

Téléchargement des ressources NLTK (exécuté automatiquement dans le notebook) :

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## Utilisation

Lancer les notebooks dans l'ordre numérique depuis le dossier `Notebook/` :

```bash
jupyter lab
```

---

## Interface web — Déploiement XGBoost

Le dossier `interface_web/` contient le script d'entraînement et les artefacts prêts à être consommés par une API.

### Générer les artefacts

```bash
python interface_web/train_xgboost.py
# ou avec des chemins custom :
python interface_web/train_xgboost.py --data-dir LIAR_DATA_SET/02_stg --out-dir interface_web
```

Cela produit :
- `model.pkl` — `XGBClassifier` entraîné (Accuracy=0.730, ROC-AUC=0.818 sur le test set)
- `vectorizer.pkl` — dictionnaire `{"tfidf": TfidfVectorizer, "scaler": StandardScaler}`

### Utilisation depuis l'API

```python
import pickle
from scipy.sparse import hstack, csr_matrix

with open("interface_web/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("interface_web/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# statement     : str   — texte de la déclaration (brut, non nettoyé)
# numeric_feats : list  — [barely_true, false, half_true, mostly_true, pants_fire, party_encoded]
X_text = vec["tfidf"].transform([statement])
X_num  = csr_matrix(vec["scaler"].transform([numeric_feats]))
proba  = model.predict_proba(hstack([X_text, X_num]))[0][1]  # probabilité d'être REAL
```

---

## Référence

Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*. ACL 2017.
