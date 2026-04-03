# Préparation du Rapport — EDA & Preprocessing
## Projet : Détection de Fake News Politiques (LIAR Dataset)

---

## Introduction

Avant de construire un modèle de classification, il est indispensable de comprendre les données sur lesquelles il va s'entraîner. Cette phase se décompose en deux grands blocs :

- **L'EDA (Exploratory Data Analysis)** : explorer, visualiser et comprendre la structure des données brutes
- **Le Preprocessing** : transformer les données brutes en données exploitables par un modèle

---

# PARTIE 1 — EDA (Analyse Exploratoire des Données)

## Étape 1 — Chargement des données

### Ce qu'on fait
On charge les trois fichiers `.tsv` (tabulation-separated values) : `train.tsv`, `test.tsv` et `valid.tsv`. Ces fichiers n'ont pas d'en-tête, donc on assigne manuellement les 14 noms de colonnes.

### Les 14 colonnes du LIAR Dataset

| Colonne | Type | Description |
|---|---|---|
| `id` | string | Identifiant unique de la déclaration |
| `label` | string | Niveau de véracité (6 catégories) |
| `statement` | string | Texte de la déclaration politique |
| `subject` | string | Thème(s) abordé(s) |
| `speaker` | string | Nom de la personne ayant fait la déclaration |
| `job_title` | string | Titre/fonction du speaker |
| `state` | string | État américain d'origine |
| `party` | string | Parti politique |
| `barely_true_counts` | int | Nombre de déclarations passées "barely-true" |
| `false_counts` | int | Nombre de déclarations passées "false" |
| `half_true_counts` | int | Nombre de déclarations passées "half-true" |
| `mostly_true_counts` | int | Nombre de déclarations passées "mostly-true" |
| `pants_fire_counts` | int | Nombre de déclarations passées "pants-fire" |
| `context` | string | Contexte/lieu où la déclaration a été faite |

### Ce qu'on vérifie
- Les **shapes** (nombre de lignes × colonnes) de chaque split
- Les **types de données** (`dtypes`) pour repérer des colonnes mal typées
- Les **5 premières lignes** pour avoir un aperçu visuel concret

### Pourquoi c'est important
Sans connaître la structure exacte des données, on risque de faire des erreurs dans toutes les étapes suivantes. C'est le socle de toute l'analyse.

---

## Étape 2.1 — Distribution des labels

### Ce qu'on fait
On trace un barplot pour chaque split (train, test, valid) montrant combien de déclarations appartiennent à chacun des 6 labels. On calcule ensuite le **ratio de déséquilibre** :

```
ratio = nb d'exemples de la classe majoritaire / nb d'exemples de la classe minoritaire
```

### Les 6 labels (du plus faux au plus vrai)

| Label | Sens |
|---|---|
| `pants-fire` | Totalement faux (comme "pants on fire") |
| `false` | Faux |
| `barely-true` | À peine vrai |
| `half-true` | À moitié vrai |
| `mostly-true` | Majoritairement vrai |
| `true` | Vrai |

### Ce qu'on cherche à comprendre
- Y a-t-il un déséquilibre entre les classes ? (problème courant en classification)
- La distribution est-elle similaire entre train, test et valid ? (important pour la validité de l'évaluation)

### Ce qu'on observe généralement
Le dataset est relativement équilibré entre les 6 classes, mais `pants-fire` est souvent la classe la moins représentée. Un ratio de déséquilibre autour de 1.5–2.0 est acceptable.

---

## Étape 2.2 — Analyse des textes

### 2.2.a — Longueur des statements

**Ce qu'on fait** : on calcule pour chaque déclaration le nombre de mots et le nombre de caractères, puis on trace des histogrammes.

**Pourquoi** : la longueur du texte est une feature implicite. Un texte très court peut être une déclaration tranchée ("Obama is a Muslim") tandis qu'un texte long est souvent plus nuancé. On veut aussi savoir si certains labels tendent à produire des déclarations plus longues.

**Ce qu'on regarde** :
- La distribution globale (est-elle normale ? asymétrique ?)
- La longueur moyenne par label (les fake news sont-elles plus courtes ou plus longues ?)

---

### 2.2.b — Top 20 mots les plus fréquents

**Ce qu'on fait** : on tokenise tous les statements du train, on retire les **stopwords** (mots vides sans sens propre : "the", "is", "a", "of"…) et on compte les occurrences de chaque mot.

**Pourquoi** : identifier les mots qui structurent le discours politique dans ce dataset. Certains mots ("Obama", "tax", "jobs", "gun") peuvent être des signaux forts.

**Ce qu'on retire avant de compter** :
- La ponctuation
- Les chiffres
- Les stopwords anglais (liste NLTK : ~180 mots)
- Les mots de moins de 3 caractères

---

### 2.2.c — Top 20 mots par label extrême

**Ce qu'on fait** : on refait la même analyse mais séparément pour les labels `pants-fire` (totalement faux) et `true` (vrai), puis on compare.

**Pourquoi** : si certains mots apparaissent beaucoup dans les fake news et peu dans les vraies déclarations (ou inversement), ce sont des features discriminantes très utiles pour le modèle.

**Ce qu'on cherche** : des patterns lexicaux différents entre les deux extrêmes. Par exemple, les fake news utilisent-elles plus de superlatifs ("never", "always", "biggest") ?

---

### 2.2.d — Wordclouds par label

**Ce qu'on fait** : pour chaque label, on génère un wordcloud où la taille du mot est proportionnelle à sa fréquence.

**Pourquoi** : une représentation visuelle immédiate et intuitive des mots dominants dans chaque catégorie de véracité. Utile pour la présentation et pour détecter rapidement des patterns.

**Note technique** : si la librairie `wordcloud` n'est pas installée, on affiche à la place un barplot horizontal des 10 mots les plus fréquents.

---

## Étape 2.3 — Analyse des métadonnées

### 2.3.a — Top 20 speakers

**Ce qu'on fait** : on compte le nombre de déclarations par speaker et on affiche les 20 plus prolifiques.

**Pourquoi** : certains speakers dominent le dataset (Barack Obama, Donald Trump, Hillary Clinton…). Une forte surreprésentation d'un speaker peut introduire un biais dans le modèle — le modèle risque d'apprendre le style de la personne plutôt que la véracité de la déclaration.

---

### 2.3.b — Distribution des labels par parti politique

**Ce qu'on fait** : on regroupe les partis en 3 catégories (Republican, Democrat, Autre) et on trace un barplot groupé montrant la distribution des 6 labels pour chaque groupe.

**Pourquoi** : vérifier s'il y a un biais partisan dans les annotations. Si les déclarations républicaines reçoivent systématiquement plus de labels "false" que les démocrates (ou inversement), cela dit quelque chose sur les données — et le modèle pourrait apprendre ce biais plutôt que la véracité réelle.

---

### 2.3.c — Heatmap parti × label

**Ce qu'on fait** : on croise les partis et les labels dans un tableau de contingence, puis on normalise en pourcentage (chaque ligne somme à 100%) et on affiche une heatmap couleur.

**Pourquoi** : la heatmap permet de voir d'un coup d'œil si la distribution des labels est homogène entre les partis. Une couleur très différente entre Democrat et Republican sur `pants-fire` révèle un biais fort.

---

### 2.3.d — Sujets fréquents

**Ce qu'on fait** : la colonne `subject` peut contenir plusieurs sujets séparés par des virgules. On les éclate, on compte et on affiche les 20 plus fréquents.

**Pourquoi** : comprendre les thématiques dominantes du dataset (économie, santé, immigration…). Certains sujets peuvent être plus sujets aux fake news que d'autres.

---

## Étape 2.4 — Analyse des compteurs historiques

### Ce que sont ces compteurs

Les 5 colonnes `barely_true_counts`, `false_counts`, `half_true_counts`, `mostly_true_counts`, `pants_fire_counts` indiquent combien de fois un speaker a reçu chaque label **dans le passé** (sur ses déclarations précédentes analysées par PolitiFact).

C'est une information extrêmement riche : l'historique de fiabilité d'un speaker est une feature prédictive forte.

### Ce qu'on fait

1. **Distribution de chaque compteur** : histogrammes pour voir si les distributions sont asymétriques (beaucoup de zéros, quelques speakers très prolifiques)
2. **Moyennes par label** : est-ce que les speakers dont les déclarations sont `pants-fire` ont un historique plus chargé en `false` et `pants-fire` passés ?
3. **Matrice de corrélation** : corrélation entre les compteurs et le label final (encodé numériquement de 0 à 5)

### Ce qu'on cherche

Un speaker avec un fort `pants_fire_counts` historique a-t-il plus de chances de voir sa déclaration actuelle classée `pants-fire` ? Si oui, ces compteurs sont des features prédictives directes et très importantes pour le modèle.

---

## Étape 2.5 — Valeurs manquantes

### Ce qu'on fait

1. **Pourcentage de nulls par colonne** pour chaque split : tableau synthétique
2. **Heatmap des nulls** : visualisation des lignes manquantes (blanc = présent, rouge = manquant)

### Colonnes typiquement manquantes dans LIAR

| Colonne | Taux de nulls typique | Raison |
|---|---|---|
| `job_title` | ~5–10% | Pas toujours renseigné |
| `state` | ~10–15% | Non renseigné pour les organisations |
| `context` | ~5% | Contexte non documenté |

### Pourquoi c'est important

Les valeurs manquantes doivent être traitées avant la modélisation. Selon leur fréquence et leur nature, on choisira de les imputer (remplacer par une valeur) ou de les conserver comme information à part (colonne binaire "cette valeur est-elle manquante ?").

---

# PARTIE 2 — Preprocessing

## Étape 3.1 — Nettoyage du texte

### Objectif

Transformer la colonne `statement` (texte brut) en une colonne `clean_statement` (texte normalisé et réduit aux mots porteurs de sens).

### Pipeline de nettoyage (dans l'ordre)

#### 1. Conversion en minuscules (lowercase)
```
"The President Said He Would NEVER Raise Taxes"
→ "the president said he would never raise taxes"
```
**Pourquoi** : "Tax" et "tax" doivent être considérés comme le même mot. Sans lowercase, le vocabulaire est artificellement multiplié.

---

#### 2. Suppression de la ponctuation et caractères spéciaux (regex)
```
"he would never raise taxes!"
→ "he would never raise taxes"
```
**Règle appliquée** : on ne conserve que les lettres a–z et les espaces (`[^a-z\s]` → supprimé).

**Pourquoi** : les signes `!`, `?`, `-`, `'` ne portent pas de sens sémantique pour un modèle de type bag-of-words ou TF-IDF. Pour les modèles de transformers (BERT), c'est différent, mais ici on reste sur du preprocessing classique.

---

#### 3. Suppression des stopwords
```
"the president said he would never raise taxes"
→ "president said would never raise taxes"
```
**Stopwords** = mots très fréquents qui n'apportent pas de sens discriminant : "the", "a", "is", "in", "of", "he", "she", "it"…

On utilise la liste NLTK (environ 180 mots anglais). On supprime aussi les mots de moins de 3 caractères.

**Pourquoi** : réduire le bruit et la dimensionnalité du vocabulaire. Ces mots seraient sinon les plus fréquents dans tous les labels sans distinction.

---

#### 4. Lemmatisation
```
"president said would never raising taxes"
→ "president say would never raise tax"
```
La **lemmatisation** ramène chaque mot à sa forme de base (son lemme) :
- "taxes" → "tax"
- "raising" → "raise"
- "said" → "say"
- "better" → "good"

**Différence avec le stemming** : le stemming coupe mécaniquement les suffixes ("running" → "run", "studies" → "studi"). La lemmatisation utilise un dictionnaire et respecte la langue ("studies" → "study").

**Pourquoi** : "raise", "raised", "raising" sont le même concept. Les regrouper réduit le vocabulaire et améliore la généralisation du modèle.

---

### Résultat final

| | Texte |
|---|---|
| **Original** | `"The President said he would never raise taxes on the middle-class families!"` |
| **Nettoyé** | `"president say would never raise tax middle class family"` |

---

## Étape 3.2 — Mapping des labels

### Pourquoi créer de nouvelles colonnes de labels ?

Le dataset original a **6 classes**. Entraîner un modèle en classification 6 classes est plus difficile (moins d'exemples par classe, frontières floues entre classes adjacentes). On crée deux variantes simplifiées selon le niveau de granularité souhaité.

---

### `label_binary` — 2 classes

| Classes originales | → | `label_binary` |
|---|---|---|
| `pants-fire`, `false`, `barely-true` | → | **0** (fake) |
| `half-true`, `mostly-true`, `true` | → | **1** (real) |

**Logique du seuil** : la frontière est tracée entre `barely-true` et `half-true`. Tout ce qui est en dessous du seuil de la moitié est considéré comme faux.

**Usage** : classification binaire simple, meilleure pour obtenir une bonne baseline et pour un déploiement en production.

---

### `label_3class` — 3 classes

| Classes originales | → | `label_3class` |
|---|---|---|
| `pants-fire`, `false` | → | **0** (faux) |
| `barely-true`, `half-true` | → | **1** (mixte / ambigu) |
| `mostly-true`, `true` | → | **2** (vrai) |

**Logique** : on fusionne les labels 2 par 2. La classe "mixte" (1) capture les déclarations ambiguës, ce qui est réaliste — le fact-checking est rarement tout noir ou tout blanc.

**Usage** : compromis entre précision de l'annotation et facilité de classification.

---

### Les 3 colonnes coexistent

Le fichier exporté contient `label` (original), `label_binary` et `label_3class`. On choisit la cible selon le modèle qu'on veut entraîner.

---

## Étape 3.3 — Encodage des métadonnées

### Parti politique

On crée deux colonnes :

| Colonne | Contenu | Exemple |
|---|---|---|
| `party_group` | Catégorie simplifiée (string) | "Republican" |
| `party_encoded` | Code numérique | 0 |

**Encodage** :
- Republican → 0
- Democrat → 1
- Autre → 2

**Pourquoi simplifier** : la colonne `party` originale contient des valeurs très diverses ("republican", "none", "organization", "libertarian"…). On regroupe pour ne garder que les deux grands partis et une catégorie fourre-tout.

**Pourquoi garder `speaker` brut** : le speaker sera utilisé pour analyser les biais individuels. On ne l'encode pas ici car il y a des centaines de speakers — un encodage ordinal n'aurait pas de sens, et un one-hot serait trop large.

---

## Étape 3.4 — Vérification finale

### Ce qu'on vérifie

1. **Shape avant/après** : le nombre de lignes ne doit pas avoir changé, seul le nombre de colonnes augmente
2. **Nulls dans `clean_statement`** : s'assurer qu'aucune ligne n'a produit une chaîne vide ou nulle après nettoyage (peut arriver si le statement original ne contenait que de la ponctuation ou des stopwords)
3. **5 exemples côte à côte** : inspection visuelle pour s'assurer que le nettoyage est cohérent et ne détruit pas l'information

---

## Étape 4 — Export vers 02_stg/

**Ce qu'on exporte**

Trois fichiers CSV dans `LIAR_DATA_SET/02_stg/` :
- `train_clean.csv`
- `test_clean.csv`
- `valid_clean.csv`

**Contenu de chaque fichier**

Toutes les **14 colonnes originales** + les **5 colonnes ajoutées** :

| Colonne ajoutée | Description |
|---|---|
| `clean_statement` | Texte nettoyé et lemmatisé |
| `label_binary` | Label binaire (0 = fake, 1 = real) |
| `label_3class` | Label ternaire (0 = faux, 1 = mixte, 2 = vrai) |
| `party_group` | Parti simplifié (Republican / Democrat / Autre) |
| `party_encoded` | Parti encodé numériquement (0 / 1 / 2) |

### Pourquoi séparer raw et stg ?

- `01_raw/` : données originales, **jamais modifiées** (source de vérité)
- `02_stg/` : données nettoyées et enrichies, prêtes pour la modélisation
- `03_dwh/` : données finales transformées (features engineerées, encodages avancés) — pour les notebooks suivants

Cette séparation permet de rejouer n'importe quelle étape sans risquer de corrompre les données source.

---

## Récapitulatif des transformations

```
train.tsv (raw)
    │
    ▼ Chargement + nommage des colonnes
    │
    ▼ EDA (aucune modification des données)
    │
    ▼ clean_text() : lower → regex → stopwords → lemmatisation
    │           → colonne clean_statement
    │
    ▼ BINARY_MAP + TRICLASS_MAP
    │           → colonnes label_binary, label_3class
    │
    ▼ simplify_party() + PARTY_ENCODE
    │           → colonnes party_group, party_encoded
    │
    ▼ Vérification (shape, nulls, exemples)
    │
    ▼ Export CSV
    │
train_clean.csv (02_stg)
```
# PARTIE 3 — Modélisation classique (TF‑IDF, SVD, K‑Means, Similarité cosinus)

**Objectif**
Explorer des méthodes classiques de représentation vectorielle et de regroupement non supervisé avant d’utiliser des modèles transformer.
L’objectif est d’évaluer si le vocabulaire seul permet de séparer les déclarations vraies et fausses.

**Contenu du notebook**
- Vectorisation TF‑IDF des textes nettoyés
- Réduction de dimension via SVD (LSA)
- Clustering K‑Means (k = 3)
- Évaluation des clusters : ARI, silhouette score
- Analyse lexicale des clusters (mots les plus fréquents)
- Similarité cosinus avec un “centre” des vrais
- Variante TF‑IDF + méta‑features (one‑hot) + SVD + K‑Means
- Variante TF‑IDF binaire + SVD + K‑Means

---

## Etape 5.1 - TF‑IDF + SVD + K‑Means (3 clusters)

**Objectif**
Représenter les textes sous forme de vecteurs TF‑IDF, réduire la dimension, puis regrouper les déclarations en 3 clusters pour voir si une structure lexicale correspond aux labels vrai/faux.

**Résultats**
Résultats
- ARI (vs labels vrais/faux) : 0.009
- Silhouette score : 0.026
Interprétation
- ARI ≈ 0 → les clusters ne correspondent pas du tout aux labels vrai/faux.

**Interprétation**
- ARI ≈ 0 → les clusters ne correspondent pas du tout aux labels vrai/faux.
- Silhouette ≈ 0 → les clusters sont très proches, mal séparés, presque équidistants des centres.
- Conclusion : aucune structure lexicale exploitable pour distinguer vrai/faux.

**Distribution clusters vs labels(3 classes)**
col_0         0         1         2
row_0                              
0      0.690062  0.060383  0.249554
1      0.570063  0.039812  0.390125


Les deux classes se répartissent presque pareil dans les trois clusters.

**Analyse lexicale des clusters**
Cluster 0 :
to, the, says, and, for, on, in, of, that, is, has, was, obama, he, from
Cluster 1 :
health, care, health care, the, insurance, the health, health insurance, of, care law, to, law, for, reform, in, care reform
Cluster 2 :
the, in, of, in the, percent, of the, than, more, percent of, is, are, have, and, states, has

**Conclusion**
Les clusters reflètent des thèmes lexicaux (santé, chiffres, politique générale), mais pas la véracité.
Le vocabulaire seul ne permet pas de séparer vrai/faux.

---

## Etape 5.2 - Similarité cosinus (LSA)

**Objectif**
Tester une approche simple : mesurer la similarité cosinus entre chaque déclaration et le “centre” des déclarations vraies.

**Résultats**
- ARI : 0.012
- Score médian vrais : 0.252
- Score médian fakes : 0.234

**Interprétation**
- Les distributions se chevauchent presque totalement.
- La similarité cosinus ne permet pas de distinguer vrai/faux.
- Confirme que la véracité n’est pas corrélée au vocabulaire.

---

## Etape 5.3 - TF-IDF + méta features + SVD + K-MEANS

**Objectif**
Ajouter des informations non textuelles (parti, sujet, état, etc.) pour voir si elles révèlent une structure plus nette.

**Résultats**
- ARI : 0.012
- Silhouette score : 0.243

**Interprétation**
- Le silhouette score augmente fortement → les clusters sont cohérents entre eux.
- L’ajout de méta‑features révèle des groupes réels dans les données (thèmes, profils d’orateurs).
- Mais l’ARI reste très faible → ces clusters ne correspondent pas aux labels vrai/faux.
- Ce résultat est attendu : la véracité n’est pas corrélée au parti, au sujet ou au vocabulaire.

**Distribution clusters vs labels**
col_0         0         1         2
row_0                              
0      0.499109  0.249777  0.251114
1      0.392385  0.223401  0.384214

Les deux classes se répartissent presque pareil dans les trois clusters.

---

## Etape 5.4 - TF IDF Binaire + SVD + K-Means

**Objectif**
Tester si la classification binaire (fake vs real) améliore la séparation.

**Résultats**
- ARI : 0.009
- Silhouette score : 0.024

**Distribution clusters vs labels**
col_0         0         1
row_0                     
0      0.263146  0.736854
1      0.408554  0.591446

**Interprétation**
- Les clusters ne correspondent toujours pas aux labels.
- Le vocabulaire ne suffit pas à distinguer vrai/faux, même en binaire.

---

## Etape 5.5 - Similarité cosinus en binaire

**Résultat**
- ARI : 0.012

**Conclusion**
Même résultat que précédemment : aucune séparation lexicale exploitable.

---

# PARTIE 3.5 — Modélisation supervisée classique (Logistic Regression, Random Forest, XGBoost)

**Notebook** : `03.5_Modelisation_pipeline.ipynb`

**Objectif**
Entraîner trois modèles supervisés de classification binaire (fake vs real) en combinant un espace TF-IDF et des features méta numériques, puis comparer leurs performances.

---

## Étape 5.5.1 — Préparation des features

### Features utilisées

| Type | Colonnes | Dimensions |
|---|---|---|
| Texte nettoyé | `clean_statement` via TF-IDF | 14 777 features (bigrammes, `min_df=2`, `max_df=0.90`, `sublinear_tf=True`) |
| Méta numériques | `barely_true_counts`, `false_counts`, `half_true_counts`, `mostly_true_counts`, `pants_fire_counts`, `party_encoded` | 6 features (StandardScaler) |

**Espace final** : 14 783 dimensions (sparse matrix TF-IDF + dense normalisé → `scipy.sparse.hstack`).

### Déséquilibre de classes

| Split | Fake (0) | Real (1) | Total |
|---|---|---|---|
| Train | 4 488 (43.8%) | 5 752 (56.2%) | 10 240 |
| Valid | 616 (48.0%) | 668 (52.0%) | 1 284 |
| Test  | 553 (43.6%) | 714 (56.4%) | 1 267 |

Déséquilibre modéré (~12 pts). Corrigé par `class_weight="balanced"` (LR, RF) et `scale_pos_weight=0.780` (XGBoost).

---

## Étape 5.5.2 — Logistic Regression

**Hyperparamètres** : `C=1.0`, `solver=lbfgs`, `max_iter=1000`, `class_weight=balanced`

**Résultats (Test)**

| Métrique | Valeur |
|---|---|
| Accuracy | 0.6204 |
| F1 (weighted) | 0.6205 |
| ROC-AUC | 0.6628 |

**Par classe (Test)**
- Fake (0) : precision=0.56, recall=0.57, F1=0.57
- Real (1) : precision=0.66, recall=0.66, F1=0.66

**Interprétation**
La régression logistique établit une baseline à 62%. Le modèle prédit mieux les vraies déclarations (F1=0.66) que les fausses (F1=0.57), suggérant que le vocabulaire associé à la vérité est légèrement plus discriminant. La combinaison linéaire des features TF-IDF et méta ne capture pas les interactions complexes entre historique du locuteur et contenu textuel.

---

## Étape 5.5.3 — Random Forest

**Hyperparamètres** : `n_estimators=300`, `min_samples_leaf=2`, `class_weight=balanced`, `n_jobs=-1`

**Résultats (Test)**

| Métrique | Valeur |
|---|---|
| Accuracy | 0.7316 |
| F1 (weighted) | 0.7314 |
| ROC-AUC | 0.7923 |

**Par classe (Test)**
- Fake (0) : precision=0.70, recall=0.68, F1=0.69
- Real (1) : precision=0.76, recall=0.77, F1=0.76

**Interprétation**
Bond de +11 pts d'accuracy par rapport à la LR (62% → 73%). Ce gain vient de deux facteurs :
1. **Interactions non-linéaires** : RF capture les combinaisons entre historique du locuteur et contenu
2. **Dominance des compteurs** : dans les feature importances, `mostly_true_counts`, `false_counts`, `barely_true_counts` et `pants_fire_counts` arrivent systématiquement en tête — l'historique de fiabilité du locuteur est le signal prédictif le plus fort
La performance est stable entre valid (72.7%) et test (73.2%), confirmant une bonne généralisation.

---

## Étape 5.5.4 — XGBoost

**Hyperparamètres** : `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=0.780`, `tree_method=hist`

**Convergence (logloss validation)** : 0.683 (epoch 0) → 0.494 (epoch 499) — descente régulière sans plateau net.

**Résultats (Test)**

| Métrique | Valeur |
|---|---|
| Accuracy | 0.7301 |
| F1 (weighted) | 0.7309 |
| ROC-AUC | **0.8175** |

**Par classe (Test)**
- Fake (0) : precision=0.68, recall=0.73, F1=0.70
- Real (1) : precision=0.78, recall=0.73, F1=0.75

**Interprétation**
Accuracy quasi-identique à Random Forest (73.0% vs 73.2%) mais **meilleur AUC : 0.818 vs 0.792**. XGBoost est mieux calibré dans ses probabilités — il discrimine plus finement les cas ambigus. La logloss converge encore à 500 arbres (pas de plateau) : augmenter `n_estimators` avec early stopping pourrait améliorer l'AUC.

---

## Étape 5.5.5 — Comparaison finale

| Modèle | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.6204 | 0.6205 | 0.6628 |
| Random Forest       | **0.7316** | **0.7314** | 0.7923 |
| XGBoost             | 0.7301 | 0.7309 | **0.8175** |

### Rôle des features méta vs texte

Les feature importances de RF et XGBoost révèlent que les **compteurs historiques dominent largement le signal TF-IDF** :
- `mostly_true_counts`, `false_counts`, `barely_true_counts`, `pants_fire_counts` → top des features
- Ces compteurs encodent la **réputation historique** du locuteur chez PolitiFact
- `party_encoded` apporte un signal secondaire
- Les tokens TF-IDF (texte) contribuent mais restent minoritaires

Ce résultat explique pourquoi la LR (combinaison linéaire) est moins performante que RF/XGBoost (interactions non-linéaires entre historique et contenu).

### Conclusion et limites

**Plafond atteint à ~73% d'accuracy / 0.82 d'AUC** avec cette approche.

| Limite | Piste |
|---|---|
| TF-IDF ignore le contexte sémantique | → BERT (notebook 03) |
| Compteurs à 0 pour blog posts / organisations | Imputation par groupe |
| XGBoost converge encore à 500 arbres | Early stopping + plus d'estimateurs |
| Pas de tuning des hyperparamètres | GridSearchCV / Optuna |

Le texte seul (TF-IDF) ne suffit pas à dépasser 62% — la sémantique contextuelle reste hors de portée des approches bag-of-words, ce qui justifie le passage à BERT.

---

# PARTIE 4 — Modelisation avancée (transformers)

## Étape 6 — Approches BERT successives
Avant d’obtenir un modèle performant, plusieurs variantes de BERT ont été testées.
Chaque approche est documentée ci‑dessous, avec ses objectifs, ses limites et ses résultats.

### Étape 6.1 — BERT en classification 3 classes

**Objectif**
Entraîner CamemBERT pour prédire les 3 classes dérivées du dataset :
- 0 = faux
- 1 = mixte / ambigu
- 2 = vrai
Pourquoi tester cette approche
- Le dataset LIAR contient 6 labels originaux, mais les regrouper en 3 classes permet de conserver une granularité intermédiaire.
- La classe “mixte” (1) reflète la réalité du fact‑checking : beaucoup de déclarations ne sont ni totalement vraies ni totalement fausses.

**Résultats obtenus**
- Accuracy : 0.44
- F1‑macro : 0.43
Analyse des erreurs
Classe 1 (mixte / half‑true)
- 262 prédictions correctes
- 208 confusions vers la classe 2
- Classe la plus facile pour le modèle (effet de majorité)
Classe 2 (vrai)
- 289 prédictions correctes
- 196 confusions vers la classe 1
- Forte proximité sémantique entre “mostly‑true” et “half‑true”
Classe 0 (faux)
- 158 prédictions correctes
- 166 confusions vers la classe 1
- Le modèle hésite fortement entre “faux” et “ambigu”

**Limites observées**
- Déséquilibre des classes : la classe 1 domine largement.
- Frontières floues : les labels originaux sont subjectifs et proches.
- Dataset très bruité : les textes sont courts, peu informatifs.
- Pas encore de méta‑features (speaker, party, subject…).

**Conclusion**
Le modèle 3 classes ne parvient pas à capturer des distinctions fines.
Cette approche est abandonnée au profit d’un modèle binaire.

---

### Étape 6.2 — BERT + méta‑features (3 classes)
**Objectif**
Améliorer la classification 3 classes en ajoutant des informations non textuelles :
- parti politique (party_encoded)
- groupe de parti (party_group)
- compteurs historiques
- sujet (subject)
- etc.

**Méthode**
- Extraction du vecteur [CLS] de CamemBERT
- Concaténation avec les méta‑features normalisées
- Passage dans un MLP final

**Résultats**
- Accuracy : 0.453
- F1‑macro : 0.444
- Confusion matrix :
[[119 132 105]
 [ 90 215 168]
 [ 59 146 246]]

**Analyse**
- L’ajout de features n’améliore pas significativement les performances.
- Les méta‑features sont elles‑mêmes bruitées et peu discriminantes.
- Le modèle continue de confondre massivement les classes 1 et 2.

**Conclusion**
L’ajout de méta‑features ne résout pas les limites structurelles du problème.
Cette approche est également abandonnée.

---

### Étape 6.3 — BERT + méta‑features + class weights (3 classes)
**Objectif**
Corriger le déséquilibre des classes en pondérant la loss :
- poids plus élevés pour les classes minoritaires

**Résultats (3 epochs)**
Epoch 1
- Train loss : 0.5409
- Val loss : 1.5366
- Val accuracy : 0.4402
- Val F1‑macro : 0.4306
Epoch 2
- Train loss : 0.3107
- Val loss : 1.7895
- Val accuracy : 0.4332
- Val F1‑macro : 0.4276
Epoch 3
- Train loss : 0.2018
- Val loss : 1.9220
- Val accuracy : 0.4332
- Val F1‑macro : 0.4302

**Analyse**
- Le modèle sur‑apprend rapidement (train loss en baisse, val loss en augmentation).
- Les class weights ne stabilisent pas l’apprentissage.
- Les performances restent équivalentes à l’approche précédente.

**Conclusion**
Même avec class weights et méta‑features, la classification 3 classes reste trop instable.
Cette approche est abandonnée.

---


### Étape 6.4 — Passage à BERT binaire
**Objectif**
Simplifier la tâche en regroupant les labels en deux classes :
- 0 = fake
- 1 = real

**Pourquoi cette approche fonctionne mieux**
- Les frontières entre “fake” et “real” sont plus nettes.
- Le dataset est moins déséquilibré en binaire.
- Les modèles transformers sont plus efficaces sur des tâches simples.
- Les erreurs de PolitiFact sont moins ambiguës en binaire.


**Pourquoi ne pas supprimer les labels intermédiaires**
Une idée initiale aurait été de supprimer les labels ambigus (barely-true, half-true, mostly-true) pour ne garder que :
- pants-fire, false : FAKE
- true : REAL
Cependant :
- cela réduit drastiquement la taille du dataset
- BERT nécessite beaucoup de données pour converger correctement
- les classes extrêmes sont trop rares pour entraîner un modèle stable
- on perd une grande partie de l’information annotée par PolitiFact
Conclusion : cette stratégie n’est pas viable.

**Nouveau mapping binaire retenu**
On adopte un regroupement plus équilibré :
FAKE (0) : pants-fire, false, barely-true
REAL (1) : half-true, mostly-true, true


**Pourquoi ce mapping fonctionne mieux**
- Les classes sont plus équilibrées que dans les versions précédentes.
- Les frontières entre FAKE et REAL sont plus nettes que dans la version 3 classes.
- On conserve 100 % du dataset, ce qui est essentiel pour BERT.
- Les labels reflètent une séparation réaliste :
- en dessous de “half‑true” : plutôt faux
- à partir de “half‑true” : plutôt vrai

**Méthode**
- Tokenisation CamemBERT
- Dataset PyTorch
- Grid Search sur :
- learning rate
- batch size
- dropout
- Sélection du meilleur modèle
- Entraînement final (2–3 epochs)
- Sauvegarde dans 03_models/

**Meilleurs hyperparamètres**
lr = 2e-5
batch_size = 8
dropout = 0.1

**Conclusion**
Le modèle binaire :
- est plus stable
- généralise mieux
- exploite pleinement le dataset
- évite les ambiguïtés des labels intermédiaires
- surpasse largement les approches 3 classes
C’est l’approche retenue pour la suite du projet.





