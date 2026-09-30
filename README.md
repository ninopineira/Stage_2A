# Stage_2A — Analyse et prédiction de mobilité à partir de données cellulaires

Stage de 2ᵉ année (ENSTA) au **CVUT de Prague**, commencé le **5 mai 2026**. Le projet exploite des
**traces de téléphonie mobile** anonymisées (Tchéquie, région de Karlovy Vary, zone `cd_142`,
**15 jours du 2014-03-12 au 2014-03-26**, 369 cellules). Pour chaque utilisateur et chaque
journée, on connaît la suite des **antennes (cellules)** auxquelles son téléphone s'est
connecté, avec l'horodatage de chaque connexion.

Ce README est **le document de référence** du dépôt : il décrit ce qu'est le projet, le format
des données, le rôle de chaque dossier et de chaque script, l'ordre dans lequel ils
s'enchaînent, les résultats obtenus, puis les pièges et les chantiers ouverts.

### En bref

| | |
|---|---|
| Enregistrements consécutifs dans la **même cellule** (`no_duplicate`) | **72,0 %** |
| Baseline « il reste où il est » (ACC@1 moyenne par utilisateur) | **0,652** |
| Markov d'ordre 1 / VOMM (mêmes 20 000 utilisateurs de test) | 0,647 / **0,674** |
| Prévisibilité maximale P^max (réplication de Song et al.) | **0,58** (0,93 chez Song) |
| Classifieur « bouge / ne bouge pas » | accuracy 0,804 (vs 0,718), ROC-AUC 0,849 |

Le fil rouge : répondre *« il restera là où il est »* fait déjà **mieux qu'un Markov d'ordre 1**.
Tout l'enjeu est donc de **détecter les instants où l'utilisateur bouge réellement**.

---

## Sommaire

1. [Objectif du projet](#1-objectif-du-projet)
2. [Démarrage rapide](#2-démarrage-rapide)
3. [Historique du dépôt](#3-historique-du-dépôt)
4. [Format des données](#4-format-des-données--à-lire-absolument)
5. [Arborescence](#5-arborescence)
6. [Détail des dossiers de `Python/`](#6-détail-des-dossiers-de-python)
7. [Enchaînement des traitements](#7-enchaînement-des-traitements)
8. [Résultats acquis](#8-résultats-acquis--à-ne-pas-refaire)
9. [Bugs trouvés et corrigés](#9-bugs-trouvés-et-corrigés)
10. [Pièges et dette technique](#10-pièges-et-dette-technique)
11. [Chantiers ouverts](#11-chantiers-ouverts)
12. [Documentation complémentaire et rapports](#12-documentation-complémentaire-et-rapports)

---

## 1. Objectif du projet

Deux grands axes de travail, dans l'ordre chronologique :

1. **Analyse et caractérisation du dataset** (mai → juillet 2026)
   Comprendre la donnée : à quelles heures les gens se connectent, quelles cellules sont les
   plus fréquentées, qui entre et qui sort de la zone, quelle est l'**entropie** de chaque
   utilisateur, où sont ses cellules **domicile** et **activité**, puis **classifier les
   utilisateurs** selon leur profil de présence sur la journée et les **antennes** selon leur
   profil horaire (K-Means).

2. **Prédiction de mobilité** (juillet → septembre 2026)
   Prédire la **prochaine cellule** que va visiter un utilisateur à partir de son historique de
   la journée, et surtout **mesurer jusqu'où c'est théoriquement possible** :
   - Markov d'ordre 1, en évaluation **causale** (online) et en split aléatoire ;
   - modèles markoviens d'**ordre variable** (VOMM avec back-off) — la piste la plus aboutie ;
   - **bornes de prévisibilité** : inégalité de Fano sur S_unc (Song et al. 2010), entropie
     conditionnelle, entropie *held-out*, Lempel-Ziv ;
   - un **Transformer** (PyTorch, attention TUPE) — code écrit, pas de résultat consolidé ;
   - une reformulation en **classification binaire « bouge / ne bouge pas »** (XGBoost,
     RandomForest, LSTM sur des descripteurs construits à la main).

---

## 2. Démarrage rapide

**Prérequis.** Python 3.12. Il n'y a pas de `requirements.txt` ; les dépendances relevées dans
les imports sont :

```bash
pip install numpy pandas matplotlib seaborn scipy tqdm geopy folium plotly scikit-learn xgboost shap joblib torch
```

**Données.** `Database/` et `results/` ne sont **pas versionnés** (≈ 7 Go et ≈ 26 Go). Cloner
le dépôt ne suffit donc pas : il faut récupérer les données à part, et vérifier que
`Database/no_duplicate/` contient bien ses **15 fichiers** (un par jour). Les variantes de
`Database/` ne sont **pas régénérables** par un script du dépôt (voir §3).

**Lancer un script.** Les imports sont plats (`from utils import get_day`,
`from models import VOMM`) : lancer chaque script **depuis son propre dossier**, ou via
`python Python/<dossier>/<script>.py` (Python ajoute le dossier du script au `sys.path`). Il n'y
a pas d'`argparse` (sauf `deep_learning/train.py`) : les paramètres sont des **constantes en tête
de fichier**, et les variantes d'expérience sont des **blocs commentés** à décommenter. Chaque
script crée lui-même ses dossiers de sortie dans `results/`. Les scripts `PREDICTABILITY_*` et
`COMPARE_*` indiquent leurs prérequis dans leur en-tête : les lire avant de lancer.

**Par où commencer la lecture :**

1. Le §4 (format des données) : sans lui, rien du code n'est lisible ; l'idiome
   `line[8::2]` / `line[9::2]` est partout.
2. [`Python/Machin_learning/NOTES_prediction_prochaine_station.md`](Python/Machin_learning/NOTES_prediction_prochaine_station.md) :
   le raisonnement complet sur la prévisibilité, les deux protocoles d'évaluation et leurs limites.
3. `important_cells_work/generalised_classification_users.py` : il définit le vocabulaire de
   classes d'utilisateurs employé ailleurs.
4. `simple_predictor/models.py`, en particulier `VOMM._recursive_prob()` : le cœur mathématique
   de la partie prédiction.
5. Avant de lancer un pipeline de prédiction, vérifier `results/predictions/` : `train_test/`,
   les matrices de n-grammes et les métriques doivent exister, sinon relancer le pipeline 4 (§7).

---

## 3. Historique du dépôt

Ce dépôt réunit **deux sources de travail** :

- le travail du stage actuel, **depuis le 5 mai 2026** : analyse, entropies, classification,
  baseline Markov, bornes de prévisibilité, rapports ;
- le travail d'un **prédécesseur** (Arthur, stage commencé début mars 2026), importé le
  **31/07/2026** (commit `f4c0d82`) : les dossiers `dataset_creation/`, `simple_predictor/`,
  `deep_learning/` et une partie de `important_cells_work/`, repris et prolongés ensuite.

Le dépôt d'origine du prédécesseur couvrait **plusieurs zones géographiques** (`cd_010`,
`cd_142`, `cd_170`) avec une arborescence par zone (`python/cd_142/analysis/`,
`Database/cd_142_dataset/`, `final_python/`…). En reprenant le travail, **une seule zone a été
conservée** (les autres datasets ont été supprimés faute de place) et **l'arborescence a été
aplatie** : `Database/<variante>/` et `Python/<thème>/` directement.

**Conséquences pratiques :**

- D'anciens chemins (`results/cd_142/…`, `Database/cd_142_dataset/…`) subsistent parfois dans
  des commentaires. Le code **actif** est câblé sur les chemins actuels : en cas de doute, c'est
  l'arborescence du §5 qui fait foi.
- Les générateurs de variantes du dataset (`GENERATION_*`) et les scripts de la zone `cd_010`
  **ne sont pas dans ce dépôt** : les variantes de `Database/` sont des **données déjà
  générées**. Si une variante doit être reconstruite, le script est à réécrire.
- `Database/cells/` et `Database/map/` contiennent encore des fichiers relatifs à `cd_010` et
  `cd_170` (listes de cellules, shapefiles QGIS), conservés comme référentiel. **Aucun script
  actif ne les lit** : tout pointe sur `cd_142`.

---

## 4. Format des données — à lire absolument

Tout le code manipule des CSV **sans en-tête**, séparés par des **points-virgules**, où
**une ligne = un utilisateur pour une journée**. Un fichier par jour (`AAAA-MM-JJ_<variante>.csv`).

```
user_id ; age ; gender ; unknown ; letters ; BS1 ; BS2 ; n_records ; cell₁ ; ts₁ ; cell₂ ; ts₂ ; … ; cellₙ ; tsₙ
   0       1       2        3         4       5     6       7          8      9     10     11
```

| Index | Champ | Description |
|---|---|---|
| 0 | `user_id` | Identifiant anonymisé — **non stable d'un jour à l'autre** |
| 1 | `age` | Âge supposé — renseigné pour ~40 % des lignes seulement |
| 2 | `gender` | `F` / `M` — même taux de remplissage |
| 3 | `unknown_numbers` | Champ numérique fourni par l'opérateur, **signification jamais élucidée** |
| 4 | `letters` | Code opérateur à 4 lettres (`CEBU`, `BUNE`, `CETR`, `NUDM`, `CENU`, `CEZB`, `NUDU`, `NUNU`, ou vide) — **sémantique inconnue**, utilisé comme proxy de « comportement » dans les premières analyses |
| 5 | `BS1` | Cellule « domicile » supposée, fournie par l'opérateur |
| 6 | `BS2` | Cellule « activité » supposée, fournie par l'opérateur (souvent vide) |
| 7 | `n_records` | Nombre d'enregistrements de la ligne |
| 8, 10, 12… | `cellᵢ` | Identifiant de cellule (ex. `BSOBRE1`, `UKVKOL102`) |
| 9, 11, 13… | `tsᵢ` | Timestamp **en secondes depuis minuit** (0 → 86400), donc **relatif à la journée** |

Extrait réel :

```
43;;;35731;BUNE;BSOHSL2;;20;BSOHSL2;89;BSOHSL2;14492;BSOHSL2;28894;…;BSOHSL2;76790
```

L'idiome utilisé **partout** dans le code :

```python
cells      = line[8::2]
timestamps = [int(ts) for ts in line[9::2]]
```

### Contraintes structurantes

- ⚠️ **Les `user_id` ne sont pas stables d'un jour à l'autre** (vérifié : intersection vide entre
  les identifiants du 12/03 et du 13/03 — l'anonymisation est refaite chaque jour). Aucune
  séquence multi-jours n'est possible : **chaque « utilisateur » n'existe que sur une journée**
  (typiquement 10 à 200 enregistrements), et toute la modélisation personnelle est confinée à
  cette journée. C'est la contrainte la plus structurante du projet.

- **Technologie.** Le premier caractère du `cellid` code la technologie : **`B` et `D` → 2G**,
  **`U` et `V` → 3G**. Une même antenne physique apparaît donc sous plusieurs `cellid`.

- **Fusion des cellules en « stations de base ».** Un utilisateur qui bascule entre `BKVPER1` et
  `BKVPER3` **ne bouge pas réellement** : il change de secteur ou de technologie. Fusionner est
  indispensable pour la prédiction de mouvement. Deux familles de fonctions coexistent :

  ```python
  # Machin_learning/, STATS_*, plot_from_csv/  → dictionnaire MERGE à 3 modes
  MERGE = {"no_merge": lambda x: x,      # cellid brut
           "simple":   get_cell_code,    # préfixe alphabétique   (BSOBRE1 → BSOBRE)
           "2g3g":     get_cell_code2}   # préfixe sans le 1er car. (BSOBRE1 → SOBRE) : fusionne 2G+3G

  # dataset_creation/, simple_predictor/    → merge_cell_id
  def merge_cell_id(cell: str):
      if cell.startswith(('B', 'D')):  return cell[1:-1]   # 2G
      else:                            return cell[1:-3]   # 3G
  ```

- **Seuil de coupure temporel.** Au-delà d'environ 4 h sans enregistrement, on considère que
  l'utilisateur est **sorti de la zone** (état `outside` inséré de part et d'autre) et on ne
  construit pas de contexte à cheval sur le trou. ⚠️ Trois marges cohabitent : `4h+30s`
  (`MAX_DELTA`, `dataset_creation/`), `4h+60s` (`sample_for_training.py`) et `4h+10min`
  (`GAP_LIMIT`, `Machin_learning/`). Elles ne sont **pas** interchangeables quand on compare des
  populations d'utilisateurs ; garder la cohérence avec le dossier concerné en ajoutant du code.

### Périmètre

- **Zone `cd_142`** — la seule étudiée. **15 jours**, du **2014-03-12 au 2014-03-26**,
  ~90 000 lignes par jour, **369 cellules**.
- **Anomalie connue** : le **vendredi 2014-03-21** a une plage horaire quasiment vide (défaut de
  collecte côté opérateur). C'est la raison invoquée pour **ne pas encoder finement les jours**
  dans les modèles (cf. l'en-tête de `deep_learning/create_cellid_map.py`).

---

## 5. Arborescence

```
Stage_2A/
├── Database/                 ← données brutes et dérivées (~7 Go, NON versionné)
│   ├── cells/                    référentiels cellid;lat;lon;x;y (cells.csv = 11 030 antennes
│   │                             du pays ; cd_142_cells.csv = les 369 de la zone)
│   ├── raw_dataset/              15 × AAAA-MM-JJ_vektory.csv — la source
│   ├── no_duplicate/             ⭐ variante de référence, utilisée par défaut partout
│   ├── no_duplicate_max_512_records/  tronqué à 512 records (deep learning)
│   ├── no_duplicate_merge_2g3g/  cellules fusionnées en stations de base
│   ├── with_distance/            ⚠️ la colonne 4 n'est plus `letters` mais la distance parcourue
│   ├── without_records/          les 8 colonnes de métadonnées seulement
│   ├── sample_for_training/      utilisateurs à plus forte entropie (pire cas, voir §8)
│   ├── class_merge/              dataset filtré sur une classe d'utilisateurs
│   └── map/                      shapefiles QGIS — non lus par le code actif
│
├── Python/                   ← tout le code (détail au §6)
│   ├── Machin_learning/          entropies, matrices de transition, baseline Markov, P^max
│   ├── important_cells_work/     cellules domicile/activité, classification users et antennes
│   ├── plot_from_csv/            figures à partir des CSV/NPY déjà produits
│   ├── dataset_creation/         pipeline 0→6 de préparation à la prédiction
│   ├── simple_predictor/         VOMM, bornes de prévisibilité, prédiction de mouvement
│   └── deep_learning/            Transformer TUPE
│
├── results/                  ← sorties de tous les scripts (~26 Go, NON versionné)
│   ├── numpy/                    .npy : entropies, matrices de transition, accuracies
│   ├── intermediate_result/      ⭐ CSV pivots relus par d'autres scripts
│   ├── plots/ maps/ json/ classification/
│   ├── 2014-03-XX/               un dossier par jour (classification des antennes)
│   └── predictions/              splits train/test, n-grammes, métriques, checkpoints
│
├── rapport/                  ← rapports LaTeX FR + EN et leurs figures (NON versionné, voir §12)
├── img_merge/                ← illustrations de la fusion 2G/3G d'une antenne (secteurs)
├── cell_locations.html       ← carte folium des cellules
├── satellite_ville.html      ← carte folium sur fond satellite
└── README.md                 ← ce document
```

Seul le **code** est versionné (≈ 110 fichiers) : `Database/`, `results/` et `rapport/` sont
dans `.gitignore`.

---

## 6. Détail des dossiers de `Python/`

### 6.1 Scripts à la racine

| Script | Rôle |
|---|---|
| `utils.py` | Boîte à outils commune : ouverture de CSV avec/sans en-tête (`open_csv_as_dataframe`, `has_header`), `get_day()`, `is_weekend()`, distances géodésiques (`geopy`), `compute_distance_travelled_by_user()`, séparation par code `letters`, helpers QGIS/GeoJSON (trajectoires). |
| `STATS_use_cell_by_hour.py` | Fréquence d'utilisation de chaque cellule **heure par heure** (nombre d'utilisateurs présents et nombre de connexions), pour les 3 modes de fusion → `results/intermediate_result/stats_(use_cell\|connections_cell)_by_hour*.csv`. **Base de la classification des antennes.** |
| `STATS_get_start_sequece.py` | Proportion des antennes qui servent d'**entrée** ou de **sortie** de la zone d'étude. |
| `sample_for_training.py` | Construit `Database/sample_for_training` : les N utilisateurs de **plus forte entropie** par jour (10–200 records, sans trou > 4 h). ⚠️ Population volontairement difficile, voir §8. |
| `find_cell.py` | Liste les stations de base disposant à la fois de cellules 2G et 3G. |
| `test_home_act_cell_dataset.py` | Compte les lignes où BS1/BS2 sont présents, égaux ou absents. |
| `Exemples.py` | Figures d'illustration pour un utilisateur : profil horaire des connexions, profil 10 min empilé par cellule. |

### 6.2 `Machin_learning/` — entropies, transitions, baseline Markov

Le cœur de la mesure de **prévisibilité**. Sorties dans `results/numpy/`. Ordre logique :
`transition_matrix` → `transition_emtropy` → `maximal_previsibility` → `markov_*` → `plot_*`.

| Script | Rôle |
|---|---|
| `transition_matrix.py` | Matrice de transition cellule A → cellule B sur les 15 jours, pour les 3 modes de fusion → `transition_matrix_{merge}.npy` et sa version normalisée. |
| `transition_emtropy.py` | Entropie **S_unc** par utilisateur → `user_entropies_{merge}.npy`, un tuple `(day, entropy, rel_entropy, n_records, n_states)` par utilisateur. |
| `transition_entropy_by_period.py` | Même chose découpée en **Matin / Jour / Soir**, 9 combinaisons de bornes (matin finissant à 4/5/6 h, soir démarrant à 18/19/20 h) → 55 valeurs par utilisateur. ⚠️ Découpe par indices (le 1ᵉʳ enregistrement tombe dans « Matin » même s'il est plus tard) et `rel` = S/N : **remplacé pour le rapport par `song_replication.py`**. |
| **`song_replication.py`** | **Réplication de Song et al.** : journée entière + 3 périodes (chaque enregistrement dans la période de son horodatage), utilisateurs d'entropie non nulle, N = états distincts, S_rel = S/log₂N → `results/plots/song_replication_no_merge.png`, `entropy_by_period_no_merge.png`, `results/intermediate_result/song_replication_*.csv(.gz)`. |
| **`maximal_previsibility.py`** | **P^max** par l'inégalité de Fano (`compute_pmax`, `load_pmax_dataframe`). Importé par la plupart des scripts de prédiction. Deux bugs importants y ont été corrigés (§9). |
| `markov_baseline.py` | Markov personnel d'ordre 1, split **aléatoire 70/30** des transitions → `markov_accuracy_no_merge.npy`. Répond à *« est-ce que je capture mon P^max sur un échantillon quelconque de mes trajets ? »*. |
| **`markov_sequential_prediction.py`** | Markov **causal / online** : à la transition *i*, on prédit avec les seules transitions `0..i-1`, puis on met à jour le modèle. **C'est le protocole honnête**, celui qui correspond au vrai objectif → `markov_sequential_accuracy_no_merge.npy`. |
| `plot_markov_vs_pmax.py` | Accuracy empirique vs P^max : scatter + droite y = x, histogrammes, découpage par nombre de records et par jour. |
| `plot_markov_sequential_by_day.py` | Vue agrégée jour par jour (accuracy moyenne vs P^max moyen, week-ends en rouge). |
| `plot_markov_methods_comparison.py` | Les deux protocoles côte à côte, **sans** référence à P^max, sur exactement la même population. |
| `NOTES_prediction_prochaine_station.md` | **Journal de bord détaillé** de cette phase : bugs trouvés, chiffres obtenus, limites méthodologiques. À lire avant de toucher à ces scripts. |

### 6.3 `important_cells_work/` — cellules importantes et classifications

Deuxième bloc de la phase d'analyse, et conceptuellement le socle de la suite.

| Script | Rôle |
|---|---|
| `2_get_user_important_cells_handmade_continue.py` | **Redétermine** la cellule domicile de chaque utilisateur en raisonnant sur la **position physique** plutôt que sur le `cellid` (`AAAAAA101` et `AAAAAA102` = même endroit), pour 9 découpages horaires → `results/intermediate_result/classified_dataset_merge_{merge}.csv`. |
| `2_get_user_act_cell_continue.py` | Même travail pour la **cellule d'activité**, avec 4 définitions concurrentes (`cont_no_gap`, `cont_gap`, `no_cont_no_gap`, `no_cont_gap`) selon qu'on exige une présence continue et qu'on tolère les trous. |
| `2_get_user_important_cells_handmade_statistics.py` | Comptages agrégés sur ces résultats (par cellule, par période, par raison de non-détection). |
| `2_activity_cells_results_against_dataset.py` | Confronte les cellules d'activité **redétectées** à la colonne `BS2` du dataset. |
| `3_home_cells_results_against_dataset.py` | Idem pour la cellule domicile contre `BS1`. |
| **`generalised_classification_users.py`** | **Le script clé.** Classification **6 bits** : la journée est découpée en 6 fenêtres de 4 h, un bit à 1 = présent dans au moins une antenne pendant cette tranche → 64 classes, regroupées en **5 clusters** : `0` toujours présent, `1` domicile mais sort la journée, `2` résident de nuit partant la journée, `3` arrivée de jour restant la nuit, `4` de passage → `user_generalised_classification.csv` et `…_by_user.csv`. |
| `classification_cells.py` | Classification des **antennes** : K-Means sur le profil horaire 24 h, après soustraction adaptative de la ligne de base (proportionnelle au coefficient de variation) et normalisation L1. Courbe du coude + centroïdes, **un dossier de résultats par jour** → `results/<jour>/classification_base_stations.csv` + PNG. |
| `utils.py` | Copie de `Python/utils.py` (maintenue identique). |

### 6.4 `plot_from_csv/` — figures

Ces scripts ne relisent (presque) jamais le dataset brut (seule exception :
`plot_sum_by_day_user_presence.py`) : ils travaillent sur les CSV et NPY déjà produits, ce qui
les rend rapides à itérer.

- Histogrammes d'entropie (`plot_hist_entropy.py`, `plot_hist_entropy_by_period.py`), entropies
  par période et par nombre de records. ⚠️ `plot_hist_entropy.py` date d'avant la correction du N
  de Fano : pour S_rand / P^max, utiliser `Machin_learning/song_replication.py`.
- Occupation horaire agrégée (`plot_sum_by_day_cell_connections.py`,
  `plot_sum_by_day_user_presence.py`), dashboard HTML Chart.js de l'usage des cellules
  (`plot_html_cell_use_by_hour.py`).
- Carte folium des entrées/sorties (`plot_entree_exit_on_map.py`), répartition des classes
  d'utilisateurs (`plot_classification.py`), distribution travail/activité
  (`plot_work_activity_distribution.py`).
- **`make_report_figures.py`** rassemble toutes les figures des rapports dans `rapport/figures/`
  (copies, deux composites d'activité, capture Chrome sans interface de la carte folium). **À
  relancer après toute régénération de figure.**
- `clean_old_plots.py` supprime les PNG produits sous d'anciennes conventions de nommage.

### 6.5 `dataset_creation/` — préparation pour la prédiction

Pipeline numéroté, pont entre l'analyse et la modélisation.

| Script | Rôle |
|---|---|
| `0_make_full_dataset_from_class_or_weekend.py` | Reconstruit **un seul fichier** à partir des 15 jours, en filtrant sur une classe d'utilisateurs et/ou en séparant semaine / week-end. |
| **`1_train_test_split.py`** | Split **80 % train / 20 % test** (mélange aléatoire, `SEED = 67`) → `results/predictions/train_test/{train,test}_random.csv`. Contient aussi, en commentaire, les variantes de split (fusionné 2G/3G, classe 1 uniquement, avec validation). |
| `2a_create_full_ngrams_matrix.py` | **V1** — n-grammes calculés sur **tout** le dataset. |
| **`2b_create_train_ngrams_matrix.py`** | **V2** — n-grammes calculés **uniquement sur le split train**. C'est la version correcte, sans fuite de données. |
| `2c_create_train_ngrams_matrix_by_class.py` | Idem, mais **une matrice par classe** d'utilisateur. |
| `3_deduplicate_csv.py` | Supprime les **répétitions consécutives de cellule** : prédire `A → A` n'a pas d'intérêt pour la prédiction de destination et les répétitions faussent les métriques. |
| `4_create_train_ngrams_with_time.py` | **V3** — ajoute une matrice des **délais** de transition. ⚠️ ~1 h de calcul et beaucoup de RAM. |
| `5_correlation_time.py` | Corrélation entre le délai moyen du contexte et le délai vers la cellule suivante. **Conclusion négative assumée : aucune corrélation exploitable.** Conservé pour ne pas refaire l'essai. |
| `6a_convert_context_to_idx.py` / `6b_convert_csv_for_gpu.py` | Conversion des contextes en indices puis en tenseurs PyTorch (`contexts`, `hours`, `targets`), **un fichier par longueur de contexte**. |
| `stats_ngrams.py` | Statistiques sur les matrices produites (couverture, nombre de suffixes par contexte). |
| `utils.py` | `find_ngrams()` et `find_ngrams_optimized()` — extraction de n-grammes avec **coupure quand l'écart temporel dépasse `max_gap`**, pour ne pas créer de faux contextes à cheval sur une nuit. |

**Format des matrices de n-grammes** (JSON), contextes de longueur **1 à 5** :

```json
{ "longueur_de_contexte": { "cellA-cellB-cellC": { "cellule_suivante": nb_occurrences } } }
```

### 6.6 `simple_predictor/` — VOMM, prévisibilité, mouvement

Le dossier le plus dense. Trois sous-chantiers partagent `models.py` et `utils.py`.

#### A. Prédiction de la prochaine cellule

- **`models.py`** — `NaiveMarkovChain` (ordre fixe), `NaiveMarkovChainWithTimeWeight`,
  **`VOMM`** (ordre variable, `max_order = 5`, back-off par *discounting* absolu — le principe de
  base de Kneser-Ney, **sans** ses comptes de continuation —, `discount = 0.75`, cache),
  `VOMM_V4` (back-off jusqu'à l'ordre 0, scoring en log-space + softmax), **`VOMM_V5`**
  (+ `temporal_boost` domicile-la-nuit / activité-le-jour, + `build_user_profile` à
  décroissance exponentielle, `discount = 0.90` — **le modèle du pipeline actuel**),
  `MovementPredictor` / `MovementPredictorV2` (extracteurs de descripteurs move/stay).

  Le cœur mathématique est `_recursive_prob` :

  ```
  P(cible | contexte) = max(count(contexte, cible) − d, 0) / count(contexte)
                        + (d · nb_suffixes_uniques / count(contexte)) · P(cible | contexte_raccourci)
  ```

  La récursion réduit le contexte d'un cran à chaque échec, **jusqu'à l'ordre 0 (l'unigramme)**.

- **`utils.py`** — `HelperVOMM` (préparation des n-grammes, comptages de contextes, unigrammes),
  `HelperData` (mapping utilisateur → jour/ligne, sauvegardes), **`Metrics`**
  (`top_k_accuracy`, `map_k`, `log_likelihood`, `negative_log_likelihood`, `perplexity`,
  `mean_reciprocal_rank`), et un `MobilityDataset` PyTorch.
- `NAIVE_prediction_pipeline.py` / `VOMM_prediction_pipeline.py` — pipelines d'évaluation. **Une
  dizaine de configurations sont préparées puis commentées** en fin de fichier (avec/sans
  répétitions, fusionné 2G/3G, par classe, semaine/week-end) : un catalogue à décommenter.
- **`COMPARE_markov_vs_vomm.py`** — Markov d'ordre 1 vs VOMM sur **exactement** le même test set
  dédupliqué, les mêmes points de prédiction et les mêmes métriques →
  `markov_vs_vomm_dedup.{json,png}`. Le seul endroit où les deux familles sont réellement
  comparables.
- `DISCOUNT_study_no_duplicate.py` — sensibilité du VOMM au discount d ∈ [0,1 ; 0,95]
  (5 000 utilisateurs, `no_duplicate`) : accuracy presque plate, log-loss minimale à **d = 0,9**.
- `PLOT_VOMM_metrics_by_discount.py` — métriques en fonction du paramètre `discount`.
- `VOMM_single_prediction.py` — prédiction sur une séquence unique, pour inspecter le modèle.
- `user_mapping.py` — index `user_id → (jour, ligne)`.
- `train_time_weight.py` — apprentissage par descente de gradient d'une matrice de poids
  `θ[heure, cellule]` (24 × 369). **N'a pas donné de résultat exploitable.**

**Métriques suivies** : `ACC@k` et `MAP@k` pour k ∈ {1, 3, 5, 10}, log-vraisemblance,
perplexité, MRR, plus un indicateur « méta » très parlant : la proportion de prédictions où le
modèle **répond simplement la dernière cellule du contexte**, en distinguant les cas justes et
faux. C'est la mesure directe du biais d'immobilité.

#### B. Bornes de prévisibilité (`PREDICTABILITY_*`)

| Script | Ce qu'il mesure |
|---|---|
| `PREDICTABILITY_vs_accuracy.py` | Dans **une seule passe** sur le test `no_duplicate` (20 000 utilisateurs, graine 67) : ACC@1/3/5/10 du VOMM, du Markov **et de la baseline « rester »**, **et** P^max issu de deux entropies (S_unc = plafond d'un prédicteur sans mémoire, S_cond = plafond d'ordre 1). Tout est aligné par utilisateur. Figures P^max, quintiles, `markov_vs_vomm_no_duplicate.png`. `REPLOT_FROM_CSV = True` retrace les figures sans refaire les ~25 min de calcul. |
| `PREDICTABILITY_by_order_calendar.py` | P^max conditionnel et ACC@1 **par ordre k = 1..5**, en calendriers de scatter (un panneau par jour). Utilise `NaiveMarkovChain(order=k)` (Markov pur, sans lissage), qui conditionne sur exactement k cellules. Montre **délibérément** l'artefact de sur-apprentissage : quand k grandit, S_cond → 0 et P^max → 1. |
| `PREDICTABILITY_heldout_bound.py` | Corrige cet artefact par une **entropie held-out** (validation croisée, lissage de Laplace `ALPHA = 1` + back-off vers l'unigramme). La cross-entropie surestime l'entropie vraie, donc la borne est **conservatrice** — c'est le bon comportement. |
| `PREDICTABILITY_lempelziv_bound.py` | Borne **Lempel-Ziv** (Song et al. 2010), déclinée par ordre maximal en plafonnant la longueur des correspondances. ⚠️ L'estimateur converge lentement : lire les moyennes par jour, pas les valeurs par utilisateur. |

#### C. Prédiction de mouvement (`MOVEMENT_PREDICTION_*`)

Reformulation binaire : au lieu de prédire *quelle* cellule, prédire **si** l'utilisateur va
changer de cellule au prochain enregistrement. Travaille sur les cellules **fusionnées** pour
ne capturer que les vrais déplacements.

| Fichier | Rôle |
|---|---|
| `MOVEMENT_PREDICTION_analyze_user_movements.py` | Étude préalable : distribution de l'heure du **premier** et du **dernier** déplacement de la journée, par jour, week-ends en rouge. |
| `MOVEMENT_PREDICTION_extract_features.py` / `…_V2.py` | Extraction des descripteurs + label. La V1 génère **tous** les points de chaque utilisateur, la V2 tire **un seul** point aléatoire par utilisateur. ⚠️ 20–40 min de calcul. |
| `MOVEMENT_PREDICTION_fit_features.py` / `…_V2.py` | **XGBoost** et **RandomForest** (déséquilibre géré par `scale_pos_weight`, early stopping) ; sorties : modèles `.joblib`, métriques JSON, ROC/PR, importances, calibration, métriques vs seuil. |
| `MOVEMENT_PREDICTION_train_lstm.py` | Alternative séquentielle : la **séquence** des vecteurs de descripteurs donnée à un **LSTM bidirectionnel** (le signal serait dans la *dérivée* des descripteurs), avec un MLP résiduel et XGBoost en comparaison. |
| `MOVEMENT_PREDICTION_analyze_features_usage.py` | Analyse **SHAP** : importance globale, beeswarm, dependence plots, et surtout **quels descripteurs trompent le modèle sur les faux positifs / faux négatifs**. |
| `MOVEMENT_PREDICTION_mobility_explorer.py` | Outil interactif : saisir un `user_id` et tracer (Plotly) sa distance à la première cellule de la journée au fil du temps. |

Les **descripteurs** tournent autour de la notion de **cellule d'ancrage** (*anchor cell*, la
cellule de référence de la journée) :

- *Position vs ancre* : `is_at_anchor_cell`, `trip_phase`, `anchor_dominance_ratio`
- *Déclenchement* : `has_departed_today`, `first_departure_elapsed_h`, `time_in_anchor_before_departure_h`
- *Retour* : `returned_to_anchor`, `time_since_return_h`
- *Dynamique* : `local_acceleration`, `n_complete_trips`, `current_trip_duration_h`, `is_oscillating`, `phase_transition_signal`
- *Profil* : `out_of_anchor_entropy`, `last_n_distinct_cells`, `recent_record_density`, et en V2 des
  descripteurs d'entropie glissante (`running_entropy`, `running_cond_entropy`, `running_pmax`)

Le **label** : on coupe l'historique de l'utilisateur à un point donné, et `label = 1` si la
cellule suivante diffère de la dernière du contexte.

### 6.7 `deep_learning/` — Transformer TUPE

| Fichier | Rôle |
|---|---|
| `create_cellid_map.py` | Mappings `cellid` → entier : `cell_map.json` (0-indexé, pour le modèle de time-weight) et `cell_map_start_1.json` (1-indexé, 0 réservé au padding), plus `day_map.json`. L'en-tête explique **pourquoi les jours ne sont pas encodés finement**. |
| `encode_and_convert_csv_to_pytorch.py` | Encode les séquences en tenseurs `(cells, times, mask)` de longueur fixe **512**, en ignorant les utilisateurs à moins de **8** enregistrements ; timestamps normalisés par 86400 ; split 60/20/20 → `encoded_{train,val,test}.pt`. |
| `dataset.py` | `MobilityDataset(Dataset)` — charge ces tenseurs. |
| `models.py` | `MobilityTransformer` avec attention **TUPE** (*Untied Positional Encoding*, Ke et al.) : les scores d'attention séparent strictement le flux *contenu* (embedding de cellule) du flux *position* (projection MLP du timestamp), au lieu de les additionner. Défauts : `d_model = 128`, `n_heads = 4`, `n_layers = 2`. |
| `train.py` | Teacher forcing décalé d'un pas, `CrossEntropyLoss` sur les positions **non paddées** et **≥ `MIN_CONTEXT` = 6**, `OneCycleLR`, AdamW, clip de gradient, reprise depuis un checkpoint. `N_CELLS` lu depuis `cell_map_start_1.json`. **~27 min par epoch.** |

Convention d'indices : **0 = PAD**, **1…N = cellules**, **N+1 = EOS** (`vocab_size = N+2`).

> **État :** infrastructure complète et cohérente, mais **aucun résultat consolidé** et **pas de
> script d'évaluation finale** (voir §11).

---

## 7. Enchaînement des traitements

### Pipeline 1 — Analyse et classification

```
Database/no_duplicate/
    ├─► STATS_use_cell_by_hour.py ──► results/intermediate_result/stats_*_by_hour_*.csv
    │        ├─► important_cells_work/classification_cells.py ──► results/<jour>/classification_base_stations.csv
    │        └─► plot_from_csv/plot_html_cell_use_by_hour.py, plot_sum_by_day_*.py
    ├─► important_cells_work/generalised_classification_users.py ──► user_generalised_classification*.csv
    │        └─► plot_from_csv/plot_classification.py
    ├─► important_cells_work/2_get_user_*_continue.py ──► classified_dataset_merge_*.csv
    │        └─► 2_/3_*_results_against_dataset.py   (confrontation à BS1/BS2)
    └─► STATS_get_start_sequece.py ──► plot_from_csv/plot_entree_exit_on_map.py
```

### Pipeline 2 — Entropies et prévisibilité théorique

```
Database/no_duplicate/
    ├─► Machin_learning/transition_matrix.py      ──► results/numpy/transition_matrix_*.npy
    │       └─► transition_emtropy.py             ──► user_entropies_{merge}.npy
    │               ├─► transition_entropy_by_period.py
    │               ├─► maximal_previsibility.py  (P^max, importé partout ailleurs)
    │               └─► plot_from_csv/plot_hist_entropy*.py, plot_entropies_by_*.py
    └─► Machin_learning/song_replication.py       ──► réplication de Song (journée + 3 périodes)
```

### Pipeline 3 — Baseline Markov

```
Database/no_duplicate/
    ├─► markov_baseline.py              ──► markov_accuracy_no_merge.npy
    └─► markov_sequential_prediction.py ──► markov_sequential_accuracy_no_merge.npy
            └─► plot_markov_vs_pmax.py / plot_markov_sequential_by_day.py / plot_markov_methods_comparison.py
```

### Pipeline 4 — VOMM et bornes de prévisibilité

```
Database/no_duplicate/
    └─► 1_train_test_split.py ──► train_random.csv / test_random.csv
            ├─► 3_deduplicate_csv.py       ──► removed_repeat/*.csv   (requis par les pipelines VOMM)
            └─► 2b_create_train_ngrams_matrix.py ──► ngrams_matrix_train_random.json
                    ├─► VOMM_prediction_pipeline.py / NAIVE_prediction_pipeline.py ──► metrics/*.json
                    │       └─► PLOT_VOMM_metrics_by_discount.py
                    ├─► COMPARE_markov_vs_vomm.py
                    ├─► DISCOUNT_study_no_duplicate.py
                    └─► PREDICTABILITY_vs_accuracy.py / _by_order_calendar.py
                        / _heldout_bound.py / _lempelziv_bound.py
```

### Pipeline 5 — Prédiction de mouvement

```
Database/no_duplicate_merge_2g3g/  (cellules fusionnées)
    └─► MOVEMENT_PREDICTION_extract_features[_V2].py ──► CSV de descripteurs
            ├─► MOVEMENT_PREDICTION_fit_features[_V2].py ──► XGBoost / RandomForest + figures
            │       └─► MOVEMENT_PREDICTION_analyze_features_usage.py  (SHAP)
            └─► MOVEMENT_PREDICTION_train_lstm.py
```

### Pipeline 6 — Deep learning

```
Database/no_duplicate_max_512_records/
    └─► create_cellid_map.py                 ──► cell_map*.json
            └─► encode_and_convert_csv_to_pytorch.py ──► encoded_{train,val,test}.pt
                    └─► train.py ──► results/predictions/deep_learning/checkpoints/
```

### Figures des rapports

```
results/plots/ … ──► plot_from_csv/make_report_figures.py ──► rapport/figures/
```

---

## 8. Résultats acquis — à ne pas refaire

### Volumes

- **1 377 475 utilisateurs-jours** ; **13 enregistrements** par utilisateur-jour en médiane, 44 en
  moyenne.
- **20,5 %** des utilisateurs-jours ont une entropie nulle (un seul état).
- `BS1` (domicile) est renseigné pour **38,6 %** des utilisateurs-jours, les deux étiquettes
  (`BS1` et `BS2`) pour **9,2 %** seulement.
- **72,0 %** des enregistrements consécutifs de `no_duplicate` sont dans la **même cellule** (le
  nettoyage « no_duplicate » ne supprime pas les répétitions consécutives).

### Réplication de Song et al.

`Machin_learning/song_replication.py`, **1 094 958 utilisateurs-jours d'entropie non nulle**,
N = nombre d'états distincts :

| Mesure | Ce projet | Song et al. |
|---|---|---|
| S_rand | 2,33 | |
| S_unc | 1,81 | |
| S_rel = S_unc / log₂N | **0,77** | |
| P^max | **0,58** | 0,93 |

Par période (chaque enregistrement dans la période de son horodatage) : matin **0,633**, journée
**0,553**, soir **0,614**.

> ⚠️ L'ancien **0,78** venait de N = nombre d'enregistrements (bug, §9), et l'ancienne
> « S_rel = 0,24 » de S_unc/N : ne plus les citer.

**Trois P^max, trois populations** : 0,58 (réplication ci-dessus), 0,575 (les 740 029
utilisateurs-jours du Markov ci-dessous), 0,598 (les 20 000 utilisateurs de test). Toujours
préciser de laquelle on parle.

### Markov d'ordre 1 sur le dataset complet

Configuration : seuil de gap 4 h 10, **≥ 10 enregistrements**, utilisateurs à **entropie nulle
exclus** (ils n'ont jamais quitté une seule cellule ; Fano leur donne P^max = 1 par convention
et ils gonflent les moyennes). 803 530 utilisateurs prédits → **740 029 conservés**.

| Mesure | Valeur |
|---|---|
| Accuracy séquentielle (causale) | **0,555** (médiane 0,571) |
| Accuracy split aléatoire 70/30 | 0,592 |
| P^max moyen (via S_unc) | 0,575 |
| Corrélation accuracy / P^max | **0,844** |
| Utilisateurs avec accuracy > P^max | 38,8 % |

Deux enseignements : **ne voir que le passé ne coûte que ~3,6 points d'accuracy** (le prix
honnête de la contrainte causale est modeste), et la corrélation de 0,844 montre que le modèle
réussit là où la théorie prédit qu'il peut réussir.

**Effet week-end net** : accuracy 0,593 le week-end contre 0,544 en semaine (P^max 0,609 vs
0,564). Les gens sont plus prévisibles le week-end — cohérent avec les profils week-end
distincts vus en classification.

#### Limite méthodologique importante

38,8 % des utilisateurs dépassent leur P^max, et ce **n'est pas une erreur** : ce P^max dérive de
**S_unc**, l'entropie *non corrélée*, qui ne connaît que la distribution des fréquences de
visite. Fano appliqué à S_unc donne le plafond d'un prédicteur **sans mémoire**. Or Markov
exploite l'**ordre** — exactement l'information que S_unc jette. Le dépasser est donc attendu.

À présenter comme *« comparaison au plafond d'un prédicteur sans mémoire »*, jamais comme une
borne absolue. C'est ce trou méthodologique que les scripts `PREDICTABILITY_*` comblent
(entropie conditionnelle, held-out, Lempel-Ziv).

### Baseline « rester », Markov et VOMM sur le même test

20 000 utilisateurs de test (graine 67), ~1,6 M points de prédiction, sur `no_duplicate`
(`PREDICTABILITY_vs_accuracy.py`, mêmes utilisateurs que `COMPARE_markov_vs_vomm.py`) :

| Modèle | ACC@1 (moyenne par utilisateur) |
|---|---|
| Baseline « reste où il est » | **0,652** (0,724 poolé) |
| Markov d'ordre 1 | 0,647 |
| VOMM | **0,674** (ACC@5 : 0,882) |

Sur le test **dédupliqué** (répétitions consécutives supprimées, donc sans le bonus « rester ») :
VOMM ACC@1 / ACC@5 / ACC@10 = **0,194 / 0,713 / 0,820**. La correction du back-off (§9) change
très peu ces chiffres.

Sensibilité au discount (`DISCOUNT_study_no_duplicate.py`) : accuracy presque plate sur
d ∈ [0,1 ; 0,95], log-loss minimale à **d = 0,9**.

### Bornes de prévisibilité par ordre

À l'ordre 5 (≈ 130 000 utilisateurs-jours), trois estimateurs de P^max :

| Estimateur | P^max |
|---|---|
| Plug-in (entropie conditionnelle empirique) | 0,98 — **artefact** de sur-apprentissage |
| Held-out (validation croisée) | **0,57** |
| Lempel-Ziv | 0,63 (0,688 sans plafond d'ordre) |

### Cellules importantes

Concordance de la cellule d'activité redétectée avec celle de l'opérateur (`BS2`) : **90–98 %**
(algo 2), 65–83 % (algo 1), 57–76 % (algo 4), 41–54 % (algo 3). Accord entre deux définitions,
parmi les utilisateurs-jours où l'une au moins trouve une cellule : **48–75 %**. (L'ancien
« 16–28 % » divisait par tous les utilisateurs, y compris ceux sans cellule d'activité.)

### Prédiction de mouvement (move/stay)

Sur 1 186 points de test : accuracy **0,804** contre **0,718** pour « ne bouge jamais »,
ROC-AUC **0,849**.

Apport des descripteurs d'entropie (même XGBoost, 400 arbres, sans / avec) : ROC-AUC
0,834 → 0,843, AP 0,687 → 0,695, F1 0,626 → 0,647 — gain **dans la marge d'incertitude**
(±0,02 pour 1 186 points). Figure `movement_entropy_before_after.png`.

### L'échantillon `sample_for_training` : le pire cas

`Database/sample_for_training` est un échantillon **à forte entropie** (99,6ᵉ percentile) : ses
chiffres (accuracy ~0,258) sont le **pire cas**, jamais représentatifs. C'est sur lui qu'avait
été mesuré le chiffre de **43,6 %** d'enregistrements consécutifs identiques qui traîne dans
d'anciennes notes (le vrai chiffre sur `no_duplicate` est 72,0 %). Sur un jour de cet
échantillon (42 133 transitions) :

| Stratégie | Accuracy |
|---|---|
| Implémentation d'alors (fallback = cellule la plus fréquente) | 0,313 |
| « L'utilisateur ne bouge pas » (baseline triviale) | **0,434** |
| Markov + fallback « ne bouge pas » | 0,425 |
| Markov + fallback + départage des ex æquo vers « rester » | 0,437 |

**32,9 %** des prédictions y sont en cold-start (cellule de départ jamais vue). Corriger le
fallback cold-start (prédire la cellule actuelle plutôt que le mode global) valait **+0,11**
d'accuracy sur ce jour.

### Ce qui a été tenté sans succès (documenté, ne pas refaire)

| Piste | Où c'est documenté |
|---|---|
| Corréler le délai du contexte au délai de la prochaine transition | En-tête de `dataset_creation/5_correlation_time.py` |
| Apprendre un poids par (heure, cellule) par descente de gradient | `simple_predictor/train_time_weight.py` |
| Estimer P^max par ordre avec l'entropie plug-in sur une seule journée | `PREDICTABILITY_by_order_calendar.py` : l'entropie s'effondre et P^max → 1 dès k ≥ 2, d'où la version held-out |
| Un score heuristique move/stay combinant les descripteurs à la main | Supprimé de `models.py` : remplacé par l'apprentissage XGBoost |

---

## 9. Bugs trouvés et corrigés

Ces corrections **invalident des chiffres produits avant elles**. Face à d'anciens résultats,
vérifier de quel côté de ces corrections ils tombent.

1. **Le N de l'inégalité de Fano.** P^max était calculé avec N = *nombre de records*, alors que
   Fano attend N = *nombre d'états distincts* (cellules distinctes visitées, + `outside` le cas
   échéant). Sur 2 731 utilisateurs du 12/03 : N moyen 145,6 → **10,6**, P^max moyen 0,767 →
   **0,580**, part des utilisateurs > 0,8 : 42,6 % → **8,1 %**. `transition_emtropy.py` stocke
   désormais `n_states` comme 5ᵉ champ du `.npy`, et `maximal_previsibility.py` lève une erreur
   explicite sur l'ancien format.

2. **L'intervalle de recherche de `brentq`.** Révélé par la correction précédente. La racine était
   cherchée sur `[1e-10, 1-1e-10]`, où la fonction de Fano a le **même signe aux deux bornes** dès
   que `S > log2(N-1)` — donc **toujours pour N = 2**. `brentq` levait `ValueError` et la fonction
   renvoyait silencieusement `NaN`, sur **37,8 %** des utilisateurs une fois le N corrigé. Corrigé
   en bornant sur `[1/N, 1]`, où le changement de signe est garanti et où la racine a un sens
   (prévisibilité au moins égale au hasard). Après correction : **0 NaN**.

3. **Le back-off du VOMM sautait l'ordre 1** (corrigé en août 2026). `_recursive_prob` s'arrêtait
   à `order == 1` en renvoyant directement l'unigramme, si bien que `train_data[1]` (les vrais
   comptages « une cellule → cellule suivante ») n'était **jamais consulté** : la récursion
   passait de l'ordre 2 à l'unigramme sans contexte. `VOMM_V4` avait déjà la bonne borne
   (`order == 0`) ; `VOMM` et `VOMM_V5` ont été alignés dessus. Les chiffres VOMM du §8 sont
   postérieurs à la correction ; **toute métrique VOMM produite avant est à relancer**.

Autres corrections déjà faites (ne pas re-signaler) :

- `Metrics.top_k_accuracy` et `map_k` étaient **définies deux fois** (la 2ᵉ écrasant la 1ʳᵉ) →
  une seule définition.
- `convert_lat_lon_distance_to_meter` était défini **deux fois avec des signatures différentes**
  dans `simple_predictor/utils.py` → une seule, alignée sur les autres copies.
- `MovementPredictor.predict_movement` appelait `extract_features` avec une mauvaise signature et
  lisait des descripteurs disparus → **méthode supprimée** (décision move/stay apprise par
  XGBoost / LSTM).
- `VOMM_single_prediction.py` utilisait une API obsolète → **réécrit** contre l'API actuelle.
- `N_CELLS = 369` était **codé en dur** dans `deep_learning/train.py` → lu depuis
  `cell_map_start_1.json`.
- `deep_learning/loss.py` et `test.py` étaient **vides** → supprimés.
- `separate_data_by_letter_code` (`utils.py`) levait un `IndexError` sans valeur manquante et
  ignorait son paramètre `column_name` → corrigé dans les deux copies.
- Import `from typing import Counter` qui masquait `collections.Counter` dans
  `STATS_get_start_sequece.py` → supprimé.

---

## 10. Pièges et dette technique

- ⚠️ **`NaiveMarkovChain.predict_next` renvoie la dernière cellule quand le contexte est
  inconnu.** Aux ordres élevés, la plupart des contextes sont inédits et le modèle retombe sur
  « reste où il est » : c'est ce qui fait monter l'ACC@1 de 0,649 (ordre 1) à 0,741 (ordre 5)
  dans `PREDICTABILITY_*`. Ce n'est **pas** un gain dû au contexte.
- **Convention d'ordre** : pour une comparaison *par ordre*, utiliser `NaiveMarkovChain(order=k)`,
  qui conditionne sur exactement k cellules. (Le commentaire d'en-tête de
  `PREDICTABILITY_by_order_calendar.py` qui dit que l'ordre du VOMM est « décalé d'un cran » date
  d'avant la correction du back-off.)
- La borne **held-out** lisse par **Laplace**, pas par *discounting* absolu comme le VOMM : même
  principe (lisser + se replier), pas le même lissage.
- **Fichiers de résultats périmés** : `predictability_by_order_per_day.csv` (une journée, modèle
  unigramme d'avant la correction). Les fichiers `predictability_*_per_user.csv` (15 jours,
  129 942 utilisateurs-jours) sont à jour. La colonne `acc` de `predictability_lz_per_user.csv`
  (moyenne 0,654) ne correspond à aucun Markov d'ordre 1–5 : provenance inconnue, ne pas la citer.
- **Populations comparées** : `Machin_learning/` évalue par utilisateur sur le dataset complet
  **avec** répétitions consécutives ; les pipelines VOMM évaluent sur un split de test
  **dédupliqué**. Ne pas comparer leurs accuracies directement — utiliser
  `COMPARE_markov_vs_vomm.py`, qui garantit les mêmes points de prédiction.
- `Database/with_distance/` : la colonne 4 n'est **plus** `letters` mais une distance — un script
  qui suppose `letters` en 4 donnera n'importe quoi.
- **Trois marges de gap** (4h+30s / 4h+60s / 4h+10min) selon les dossiers (§4). Conservées telles
  quelles pour ne pas invalider les résultats existants ; à unifier un jour, en connaissance de
  cause.
- **Code volontairement dupliqué** : `utils.py` existe en 3 copies (racine,
  `important_cells_work/`, `simple_predictor/`), et `get_cell_code` /
  `MERGE` / `get_day` sont recopiés dans presque chaque script. **Modifier l'un impose de vérifier
  les autres.**
- **Pas de `requirements.txt`** (voir §2 pour la liste des dépendances).
- **Pilotage par commentaires** : les configurations d'expérience sont activées en décommentant
  des blocs. L'historique des runs n'est pas rejouable en un clic, et deux résultats ne sont pas
  garantis d'avoir été produits avec les mêmes réglages.
- **Pas d'évaluation du Transformer** : `train.py` a une fonction `evaluate()` interne, mais rien
  ne produit de métriques de test comparables à celles du VOMM.
- Des `.pyc` (Python 3.12 et 3.14) sont versionnés par erreur dans des `__pycache__/`.

---

## 11. Chantiers ouverts

1. **Relancer les anciennes métriques VOMM** du catalogue de `VOMM_prediction_pipeline.py` après
   la correction du back-off (§9.3) — préalable à toute comparaison de configurations.
2. **Un tableau de comparaison unifié.** Tous les modèles (Markov d'ordre fixe, VOMM,
   VOMM + boosts, Transformer, classifieur move/stay) sur **le même split de test** et les mêmes
   métriques (ACC@1/3/5, MAP@k, log-loss). `COMPARE_markov_vs_vomm.py` en donne le patron.
3. **Aller au bout du Transformer** : il n'est configuré que pour 3 epochs et n'a pas de
   procédure d'évaluation. Son ACC@1 dira si le coût du deep learning se justifie sur seulement
   deux semaines de données.
4. **Prédicteur en deux étages.** Étant donné les 72 % de self-transitions, un étage 1 « bouge /
   bouge pas » (le classifieur move/stay existe déjà) suivi d'un étage 2 « où ? » (VOMM,
   seulement quand un mouvement est prédit) est la piste la plus prometteuse pour l'ACC@1.
5. **Exploiter les bornes held-out et Lempel-Ziv** : comparer l'accuracy des modèles à ces
   plafonds-là (et non au seul P^max issu de S_unc) donne une lecture méthodologiquement correcte
   de « à quelle distance de l'optimum on est ».
6. **Évaluer les poids temporels** (`train_time_weight.py`) face au VOMM nu, ou acter leur abandon.
7. **Fixer un seul dataset de référence et un seul split partagé** (et, au passage, une seule
   marge de gap).

---

## 12. Documentation complémentaire et rapports

**Notes versionnées dans `Python/` :**

- [`Machin_learning/NOTES_prediction_prochaine_station.md`](Python/Machin_learning/NOTES_prediction_prochaine_station.md)
  — journal de bord de la phase Markov / P^max : bugs, chiffres, limites.
- [`NOTES_previsibilite_et_entropie_RAPPORT.md`](Python/NOTES_previsibilite_et_entropie_RAPPORT.md)
  — thèses, formules et raisonnement derrière la prédiction et la prévisibilité.
- [`EXPLICATION_deep_learning_et_simple_predictor.md`](Python/EXPLICATION_deep_learning_et_simple_predictor.md)
  — description fichier par fichier des dossiers `deep_learning/` et `simple_predictor/`.

**Rapports (`rapport/`, non versionné) :**

- `rapport_stage_2A.tex` (français, ~31 p.) et `report_detailed_en.tex` (anglais, détaillé,
  ~67 p.). Compilation : `pdflatex` **deux fois** (références). Les figures absentes s'affichent
  en cadre gris. Les figures sont rassemblées par `plot_from_csv/make_report_figures.py`.
- Les deux rapports doivent rester **cohérents entre eux** (mêmes chiffres, même bibliographie).
- **Bibliographie** : uniquement des travaux lus en entier — Song et al. 2010, Gambs et al. 2012,
  González et al. 2008, Lu et al. 2013, Ikanovic & Mollgaard 2017, Begleiter et al. 2004,
  Chen & Goodman 1999, Ke et al. 2021 (TUPE), Cover & Thomas 2006, Kontoyiannis 1998,
  Antos & Kontoyiannis 2001, Brown et al. 1992.
- Chez Ikanovic & Mollgaard, « stationarity » désigne l'**immobilité** (les gens restent au même
  endroit), pas la stationnarité statistique du processus.
- Le stage a donné lieu à **23 présentations de suivi**, environ tous les trois jours à partir du
  5 mai 2026 ; les diapositives datées de mars–avril 2026 sont celles du prédécesseur.
