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

### Ce qu'on exporte

Trois fichiers CSV dans `LIAR_DATA_SET/02_stg/` :
- `train_clean.csv`
- `test_clean.csv`
- `valid_clean.csv`

### Contenu de chaque fichier

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
