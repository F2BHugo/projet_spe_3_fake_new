# Projet Spé 3 — Détection de Fake News Politiques (LIAR Dataset)

## Objectif

Développer un pipeline complet de détection de fake news politiques à partir du **LIAR Dataset**, en passant par l'exploration des données, le preprocessing, et la modélisation.

---

## Structure du projet

```
PROJET_SPE_3_FAKE_NEW/
├── LIAR_DATA_SET/
│   ├── 01_raw/          ← Données brutes originales (TSV + CSV POS)
│   ├── 02_stg/          ← Données nettoyées et preprocessées
│   └── 03_dwh/          ← Données finales prêtes pour la modélisation
├── Notebook/
│   └── 01_EDA_preprocessing.ipynb   ← EDA complète + preprocessing
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

## Référence

Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*. ACL 2017.
