# Résumé du projet — Analyse et prédiction de mobilité à partir de données cellulaires

> Document de reprise. Il décrit **ce qu'est le projet**, **le format des données**, **le rôle de chaque dossier et de chaque script**, **l'ordre dans lequel les choses s'enchaînent**, et enfin **l'état d'avancement et les pièges** à connaître avant de reprendre le travail.
>
> Rédigé à partir de la lecture du code, des notes internes ([Python/Machin_learning/NOTES_prediction_prochaine_station.md](Python/Machin_learning/NOTES_prediction_prochaine_station.md), [Python/EXPLICATION_deep_learning_et_simple_predictor.md](Python/EXPLICATION_deep_learning_et_simple_predictor.md)) et de l'historique git (mai → août 2026).
>
> **Document compagnon :** [CLAUDE.md](CLAUDE.md) — même contenu factuel, mais organisé comme une carte de référence (conventions, chemins, sorties de chaque script). Ce fichier-ci raconte le *pourquoi* et l'*état d'avancement* ; CLAUDE.md sert à se repérer dans le code.

---

## 1. Objectif du projet

Le projet exploite un **jeu de données de traces de téléphonie mobile** (Tchéquie, mars 2014) : pour chaque utilisateur et pour chaque journée, on dispose de la suite des **antennes (cellules) auxquelles son téléphone s'est connecté**, avec l'horodatage de chaque connexion.

Deux grands axes de travail, dans cet ordre chronologique :

1. **Analyse et caractérisation du dataset** (mai → juillet 2026)
   Comprendre la donnée : à quelles heures les gens se connectent, quelles cellules sont les plus fréquentées, qui entre et qui sort de la zone, quelle est l'**entropie** de chaque utilisateur, et **classifier les utilisateurs** selon leur profil de présence sur la journée ainsi que les **antennes** selon leur profil horaire.

2. **Prédiction de mobilité** (juillet → août 2026)
   Prédire la **prochaine cellule** que va visiter un utilisateur à partir de son historique de la journée, et surtout **mesurer jusqu'où c'est théoriquement possible** :
   - Markov d'ordre 1, en évaluation **causale** (online) et en split aléatoire ;
   - modèles markoviens d'**ordre variable** (VOMM avec back-off) — la piste la plus aboutie ;
   - **bornes de prévisibilité** (inégalité de Fano sur S_unc, entropie conditionnelle, entropie held-out, Lempel-Ziv) ;
   - un **Transformer** (PyTorch, attention TUPE) — code écrit, pas de résultat consolidé ;
   - une reformulation en **classification binaire « bouge / ne bouge pas »** (XGBoost / RandomForest / LSTM sur features construites à la main).

Le fil rouge implicite : la prédiction « naïve » qui consiste à répondre *« il restera là où il est »* est déjà très bonne (**72,0 %** des enregistrements consécutifs sont dans la même cellule sur `no_duplicate` ; elle obtient 0,652 d'ACC@1, au-dessus du Markov d'ordre 1), donc tout l'enjeu est de **détecter les instants où l'utilisateur bouge réellement**.

---

## 2. Historique du dépôt — à savoir avant de lire le code

Ce dépôt réunit **deux sources de travail** :

- le travail mené ici depuis le **11/05/2026** (analyse, entropies, classification, baseline Markov) ;
- le travail d'un **ancien collaborateur (Arthur)**, importé le **31/07/2026** (commit `f4c0d82`, *« Markov and adding of the work of arthur »*) : les dossiers `dataset_creation/`, `simple_predictor/`, `deep_learning/`, et une partie de `important_cells_work/`.

Le dépôt d'origine d'Arthur travaillait sur **plusieurs zones géographiques** (`cd_010`, `cd_142`, `cd_170`) et son arborescence reflétait ce découpage (`python/cd_142/analysis/`, `Database/cd_142_dataset/`, `final_python/`…). En reprenant le travail, **une seule zone a été conservée** — les autres datasets ont été supprimés pour libérer de la place (les fichiers sont lourds : `Database/` pèse encore ~7 Go et `results/` ~26 Go) — et **l'arborescence a été aplatie** : plus de niveau par zone, `Database/<variante>/` directement, `Python/<thème>/` directement.

**Conséquences pratiques :**

- Certains scripts importés portent encore des traces de l'ancienne organisation dans leurs commentaires ou dans des chemins commentés (`results/cd_142/…`, `Database/cd_142_dataset/…`). Le code **actif** a été recâblé sur les chemins actuels, mais en cas de doute, c'est toujours l'arborescence décrite au §4 qui fait foi.
- Les scripts d'analyse de la zone `cd_010` et les générateurs de variantes du dataset (`GENERATION_*`) d'Arthur **ne sont pas dans ce dépôt** : les variantes de `Database/` sont présentes en tant que **données déjà générées**, pas en tant que code régénérable. Si une variante doit être reconstruite, le script est à réécrire.
- Il reste dans `Database/cells/` et `Database/map/` des fichiers relatifs à `cd_010` et `cd_170` (listes de cellules, shapefiles QGIS). Ils sont conservés comme référentiel, mais **aucun script actif ne les lit** : tout pointe sur `cd_142`.

---

## 3. Le format des données — à lire absolument

Tout le code manipule des CSV **sans en-tête**, séparés par des **points-virgules**, où **une ligne = un utilisateur pour une journée**. Un fichier = un jour.

```
user_id ; age ; gender ; unknown ; letters ; BS1 ; BS2 ; n_records ; cell₁ ; ts₁ ; cell₂ ; ts₂ ; … ; cellₙ ; tsₙ
   0       1       2        3         4       5     6       7          8      9     10     11
```

| Index | Champ | Description |
|---|---|---|
| 0 | `user_id` | Identifiant anonymisé de l'utilisateur |
| 1 | `age` | Âge supposé — renseigné pour une minorité d'utilisateurs seulement |
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

**Points d'attention :**

- `cells = row[8::2]` et `timestamps = row[9::2]` est **l'idiome utilisé partout** dans le code.
- ⚠️ **Les `user_id` ne sont PAS stables d'un jour à l'autre** (vérifié : intersection vide entre les ids du 12/03 et du 13/03 — l'anonymisation est refaite chaque jour). Impossible donc de construire une séquence multi-jours : **chaque « utilisateur » n'existe que sur une journée** (typiquement 10–200 records), et toute la modélisation personnelle est confinée à cette journée. C'est la contrainte la plus structurante du projet.
- Le premier caractère du `cellid` code la technologie : **`B` et `D` → 2G**, **`U` et `V` → 3G**. Une même antenne physique apparaît donc sous plusieurs `cellid`.
- **Fusion des cellules en « stations de base »** — récurrente dans tout le projet. Deux familles de fonctions coexistent :
  ```python
  # Machin_learning/, STATS_*, plot_from_csv/  → dictionnaire MERGE à 3 modes
  MERGE = {"no_merge": lambda x: x,      # cellid brut
           "simple":   get_cell_code,    # préfixe alphabétique (BSOBRE1 → BSOBRE)
           "2g3g":     get_cell_code2}   # préfixe sans le 1er caractère (BSOBRE1 → SOBRE)

  # dataset_creation/, simple_predictor/    → merge_cell_id
  def merge_cell_id(cell: str):
      if cell.startswith(('B','D')):  return cell[1:-1]   # 2G
      else:                           return cell[1:-3]   # 3G
  ```
  Objectif identique : un utilisateur qui bascule entre `BKVPER1` et `BKVPER3` **ne bouge pas réellement**, il change juste de secteur ou de technologie. C'est indispensable pour la prédiction de mouvement.
- **Seuil de coupure temporel** : au-delà d'environ 4 h sans enregistrement, on considère que l'utilisateur est **sorti de la zone** (état `outside`) et on ne construit pas de contexte à cheval sur le trou. ⚠️ Trois marges cohabitent dans le code : `4h+30s` (`MAX_DELTA`, dataset_creation), `4h+60s` (`sample_for_training.py`) et `4h+10min` (`GAP_LIMIT`, Machin_learning). Elles ne sont **pas** interchangeables quand on compare des populations d'utilisateurs.

### Périmètre

- **Zone `cd_142`** — la seule zone étudiée. **15 jours**, du **2014-03-12 au 2014-03-26**, ~90 000 lignes/jour, **369 cellules**.
- **Anomalie connue** : le **vendredi 2014-03-21** a une plage horaire quasiment vide (problème de collecte côté opérateur). C'est la raison invoquée pour **ne pas encoder finement les jours** dans les modèles (cf. l'en-tête de `deep_learning/create_cellid_map.py`).

---

## 4. Arborescence réelle

```
Stage_2A/
├── Database/                 ← Données brutes et dérivées (~7 Go, NON versionné)
│   ├── cells/                    Référentiels cellid;lat;lon;x;y (cells.csv = 11 030 antennes
│   │                             du pays ; cd_142_cells.csv = les 369 de la zone)
│   ├── raw_dataset/              15 × AAAA-MM-JJ_vektory.csv — la source
│   ├── no_duplicate/             ⭐ variante de référence, utilisée par défaut partout
│   ├── no_duplicate_max_512_records/  tronqué à 512 records (deep learning)
│   ├── no_duplicate_merge_2g3g/  cellules fusionnées en stations de base
│   ├── with_distance/            ⚠️ la colonne 4 n'est plus `letters` mais la distance parcourue
│   ├── without_records/          les 8 colonnes de métadonnées seulement
│   ├── sample_for_training/      échantillon des utilisateurs à plus forte entropie
│   ├── class_merge/              dataset filtré sur une classe d'utilisateurs
│   └── map/                      shapefiles QGIS (cd_010, cd_142) — non lus par le code actif
│
├── Python/                   ← Tout le code (détail au §5)
│   ├── Machin_learning/          Entropies, matrices de transition, baseline Markov, P^max
│   ├── important_cells_work/     Cellules domicile/activité, classification users et antennes
│   ├── plot_from_csv/            Figures à partir des CSV/NPY déjà produits
│   ├── dataset_creation/         Pipeline 0→6 de préparation à la prédiction
│   ├── simple_predictor/         VOMM, prévisibilité, prédiction de mouvement
│   └── deep_learning/            Transformer TUPE
│
├── results/                  ← Sorties de tous les scripts (~26 Go, NON versionné)
│   ├── numpy/                    .npy : entropies, matrices de transition, accuracies
│   ├── intermediate_result/      ⭐ CSV pivots relus par d'autres scripts
│   ├── plots/ maps/ json/ classification/
│   ├── 2014-03-XX/               un dossier par jour (classification des antennes)
│   └── predictions/              splits train/test, n-grammes, métriques, checkpoints
│
├── CLAUDE.md                 ← Carte de référence du dépôt
├── RESUME_PROJET.md          ← Ce document
└── README.md
```

Seuls **100 fichiers** sont versionnés : le code, `Database/cells/`, et les deux documents de synthèse. `Database/` et `results/` sont dans `.gitignore`.

---

## 5. Détail des dossiers de `Python/`

### 5.1 Scripts à la racine

| Script | Rôle |
|---|---|
| `utils.py` | Boîte à outils commune : ouverture de CSV avec/sans en-tête (`has_header`), `get_day()`, `is_weekend()`, distance géodésique (`geopy`), `compute_distance_travelled_by_user()`, séparation par code `letters`. Contient aussi du code QGIS commenté (trajectoires en GeoJSON). |
| `STATS_use_cell_by_hour.py` | Fréquence d'utilisation de chaque cellule **heure par heure** (nombre d'utilisateurs présents et nombre de connexions), pour les 3 modes de fusion. C'est la **base de la classification des antennes**. → `results/intermediate_result/stats_*_by_hour_*.csv` |
| `STATS_get_start_sequece.py` | Proportion des antennes qui servent d'**entrée** ou de **sortie** de la zone d'étude. |
| `sample_for_training.py` | Construit `Database/sample_for_training` : les N utilisateurs de **plus forte entropie** par jour (10–200 records, sans trou > 4 h). ⚠️ Population volontairement difficile — voir §7. |
| `find_cell.py` | Liste les stations de base disposant à la fois de cellules 2G et 3G. |
| `test_home_act_cell_dataset.py` | Compte, dans le dataset brut, les lignes où BS1/BS2 sont présents, égaux ou absents. |
| `Exemples.py` | Figures d'illustration pour un utilisateur : profil horaire des connexions, profil 10 min empilé par cellule. |

### 5.2 `Machin_learning/` — entropies, transitions, baseline Markov

Le cœur de la mesure de **prévisibilité**. Toutes les sorties vont dans `results/numpy/`.

| Script | Rôle |
|---|---|
| `transition_matrix.py` | Matrice de transition cellule A → cellule B sur les 15 jours, pour les 3 modes de fusion → `transition_matrix_{merge}.npy` et sa version normalisée. |
| `transition_emtropy.py` | Entropie **S_unc** par utilisateur → `user_entropies_{merge}.npy`, un tuple `(day, entropy, rel_entropy, n_records, n_states)` par utilisateur. |
| `transition_entropy_by_period.py` | Même chose découpée en **Matin / Jour / Soir**, avec 9 combinaisons de bornes (matin finissant à 4/5/6 h, soir démarrant à 18/19/20 h) → 55 valeurs par utilisateur. |
| **`maximal_previsibility.py`** | **P^max** par l'inégalité de Fano (`compute_pmax`, `load_pmax_dataframe`). Importé par la plupart des scripts de prédiction. Deux bugs importants y ont été corrigés — voir §8. |
| `markov_baseline.py` | Markov personnel d'ordre 1, split **aléatoire 70/30** des transitions → `markov_accuracy_no_merge.npy`. Répond à *« est-ce que je capture mon P^max sur un échantillon quelconque de mes trajets ? »*. |
| **`markov_sequential_prediction.py`** | Markov **causal / online** : à la transition *i*, on prédit avec les seules transitions `0..i-1`, puis on met à jour le modèle. **C'est le protocole honnête**, celui qui correspond au vrai objectif → `markov_sequential_accuracy_no_merge.npy`. |
| `plot_markov_vs_pmax.py` | Accuracy empirique vs P^max : scatter + droite y=x, histogrammes, découpage par nombre de records et par jour. |
| `plot_markov_sequential_by_day.py` | Vue agrégée jour par jour (accuracy moyenne vs P^max moyen, week-ends en rouge). |
| `plot_markov_methods_comparison.py` | Les deux protocoles côte à côte, **sans** référence à P^max, sur exactement la même population. |
| `NOTES_prediction_prochaine_station.md` | **Journal de bord détaillé** de cette phase : bugs trouvés, chiffres obtenus, limites méthodologiques. À lire avant de toucher à ces scripts. |

### 5.3 `important_cells_work/` — cellules importantes et classifications

Deuxième bloc de la phase d'analyse, et **conceptuellement le socle de la suite**.

| Script | Rôle |
|---|---|
| `2_get_user_important_cells_handmade_continue.py` | **Redétermine soi-même** la cellule domicile de chaque utilisateur, en raisonnant sur la **position physique** plutôt que sur le `cellid` (`AAAAAA101` et `AAAAAA102` = même endroit), pour 9 découpages horaires → `results/intermediate_result/classified_dataset_merge_{merge}.csv`. |
| `2_get_user_act_cell_continue.py` | Même travail pour la **cellule d'activité**, avec 4 définitions concurrentes (`cont_no_gap`, `cont_gap`, `no_cont_no_gap`, `no_cont_gap`) selon qu'on exige une présence continue et qu'on tolère les trous. |
| `2_get_user_important_cells_handmade_statistics.py` | Comptages agrégés sur ces résultats (par cellule, par période, par raison de non-détection). |
| `2_activity_cells_results_against_dataset.py` | Confronte les cellules d'activité **redétectées** à la colonne `BS2` du dataset — combien coïncident, selon quelle définition. |
| `3_home_cells_results_against_dataset.py` | Idem pour la cellule domicile contre `BS1`. |
| **`generalised_classification_users.py`** | **Le script clé.** Classification **6 bits** : la journée est découpée en 6 fenêtres de 4 h, un bit à 1 = présent dans au moins une antenne pendant cette tranche → 64 classes possibles, regroupées ensuite en **5 clusters** interprétables : `0` toujours présent, `1` domicile mais sort la journée, `2` résident de nuit partant la journée, `3` arrivée de jour restant la nuit, `4` de passage. → `user_generalised_classification.csv` et `…_by_user.csv`. |
| `classification_cells.py` | Classification des **antennes** (et non des utilisateurs) : K-Means sur le profil horaire 24 h, après soustraction adaptative de la ligne de base (proportionnelle au coefficient de variation) et normalisation L1. Courbe du coude + centroïdes, **un dossier de résultats par jour**. |
| `utils.py` | Copie de `Python/utils.py` (maintenue identique). |

### 5.4 `plot_from_csv/` — figures

Ces scripts ne relisent (presque) jamais le dataset brut : ils travaillent sur les CSV et NPY déjà produits, ce qui les rend rapides à itérer.

Histogrammes d'entropie (`plot_hist_entropy.py`, `plot_hist_entropy_by_period.py`), entropies par période et par nombre de records, occupation horaire agrégée (`plot_sum_by_day_cell_connections.py`, `plot_sum_by_day_user_presence.py`), dashboard HTML Chart.js de l'usage des cellules (`plot_html_cell_use_by_hour.py`), carte folium des entrées/sorties (`plot_entree_exit_on_map.py`), répartition des classes d'utilisateurs (`plot_classification.py`), distribution travail/activité. `clean_old_plots.py` supprime les PNG produits sous d'anciennes conventions de nommage.

### 5.5 `dataset_creation/` — préparation pour la prédiction

Pipeline numéroté, pont entre l'analyse et la modélisation.

| Script | Rôle |
|---|---|
| `0_make_full_dataset_from_class_or_weekend.py` | Reconstruit **un seul fichier** à partir des 15 jours, en filtrant sur une classe d'utilisateurs et/ou en séparant semaine / week-end. |
| **`1_train_test_split.py`** | Split **80 % train / 20 % test** (mélange aléatoire, `SEED = 67`) → `results/predictions/train_test/{train,test}_random.csv`. Le fichier contient aussi, en commentaire, les variantes de split (fusionné 2G/3G, classe 1 uniquement, avec validation). |
| `2a_create_full_ngrams_matrix.py` | **V1** — n-grammes calculés sur **tout** le dataset. |
| **`2b_create_train_ngrams_matrix.py`** | **V2** — n-grammes calculés **uniquement sur le split train**. C'est la version correcte, sans fuite de données. |
| `2c_create_train_ngrams_matrix_by_class.py` | Idem, mais **une matrice par classe** d'utilisateur. |
| `3_deduplicate_csv.py` | Supprime les **répétitions consécutives de cellule**. Indispensable : prédire `A → A` n'a aucun intérêt et les répétitions faussent complètement les métriques. |
| `4_create_train_ngrams_with_time.py` | **V3** — ajoute une matrice des **délais** de transition. ⚠️ ~1 h de calcul et beaucoup de RAM. |
| `5_correlation_time.py` | Étude de corrélation entre le délai moyen du contexte et le délai vers la cellule suivante. **Conclusion négative écrite en gros dans le fichier : aucune corrélation exploitable.** Conservé pour ne pas refaire l'essai. |
| `6a_convert_context_to_idx.py` / `6b_convert_csv_for_gpu.py` | Conversion des contextes en indices puis en tenseurs PyTorch (`contexts`, `hours`, `targets`), **un fichier par longueur de contexte**. |
| `stats_ngrams.py` | Statistiques sur les matrices produites (couverture, nombre de suffixes par contexte). |
| `utils.py` | `find_ngrams()` et `find_ngrams_optimized()` — extraction de n-grammes avec **coupure quand l'écart temporel dépasse `max_gap`**, pour ne pas créer de faux contextes à cheval sur une nuit. |

**Format des matrices de n-grammes** (JSON), contextes de longueur **1 à 5** :

```json
{ "longueur_de_contexte": { "cellA-cellB-cellC": { "cellule_suivante": nb_occurrences } } }
```

### 5.6 `simple_predictor/` — VOMM, prévisibilité, mouvement

Le dossier le plus dense. Trois sous-chantiers partagent `models.py` et `utils.py`.

**A. Prédiction de la prochaine cellule**

- **`models.py`** — `NaiveMarkovChain` (ordre fixe), `NaiveMarkovChainWithTimeWeight`, **`VOMM`** (ordre variable + *absolute discounting* et back-off récursif dans l'esprit de Kneser-Ney), `VOMM_V4` (back-off jusqu'à l'ordre 0, scoring en log-space + softmax), **`VOMM_V5`** (+ `temporal_boost` domicile-la-nuit / activité-le-jour, + profil utilisateur à décroissance exponentielle ; `discount = 0.90` — **le modèle du pipeline actuel**).

  Le cœur mathématique est `_recursive_prob` :
  ```
  P(cible | contexte) = max(count(contexte,cible) − d, 0) / count(contexte)
                        + (d · nb_suffixes_uniques / count(contexte)) · P(cible | contexte_raccourci)
  ```
  La récursion réduit le contexte d'un cran à chaque échec, **jusqu'à l'ordre 0 (l'unigramme)**.

- **`utils.py`** — `HelperVOMM` (préparation des n-grammes, comptages de contextes, unigrammes), `HelperData` (mapping utilisateur → jour/ligne, sauvegardes), **`Metrics`** (`top_k_accuracy`, `map_k`, `log_likelihood`, `negative_log_likelihood`, `perplexity`, `mean_reciprocal_rank`), et un `MobilityDataset` PyTorch.
- `NAIVE_prediction_pipeline.py` / `VOMM_prediction_pipeline.py` — pipelines d'évaluation. **Une dizaine de configurations d'expérience sont préparées puis commentées** en fin de fichier (avec/sans répétitions, fusionné 2G/3G, par classe, semaine/week-end) : c'est un catalogue à décommenter une par une.
- `COMPARE_markov_vs_vomm.py` — Markov d'ordre 1 vs VOMM sur **exactement** le même test set, les mêmes points de prédiction et les mêmes métriques. C'est le seul endroit où les deux familles sont réellement comparables.
- `VOMM_single_prediction.py` — prédiction sur une séquence unique, pour inspecter/déboguer le comportement du modèle.
- `PLOT_VOMM_metrics_by_discount.py` — métriques en fonction du paramètre `discount`.
- `user_mapping.py` — index `user_id → (jour, ligne)`.
- `train_time_weight.py` — apprentissage par descente de gradient d'une matrice de poids `θ[heure, cellule]` (24 × 369). **N'a pas donné de résultat exploitable.**

**Métriques suivies** : `ACC@k` et `MAP@k` pour k ∈ {1, 3, 5, 10}, log-vraisemblance moyenne, plus un indicateur « méta » très parlant : la proportion de prédictions où le modèle **répond simplement la dernière cellule du contexte**, en distinguant les cas où il a raison de ceux où il a tort. C'est la mesure directe du biais d'immobilité évoqué en introduction.

**B. Bornes de prévisibilité (`PREDICTABILITY_*`)** — l'apport le plus récent (août 2026)

| Script | Ce qu'il mesure |
|---|---|
| `PREDICTABILITY_vs_accuracy.py` | Dans **une seule passe** : accuracy empirique des modèles **et** P^max issu de deux entropies (S_unc = plafond d'un prédicteur sans mémoire, S_cond = plafond d'ordre 1). Tout est aligné par utilisateur, sans jointure fragile. |
| `PREDICTABILITY_by_order_calendar.py` | P^max conditionnel et ACC@1 **par ordre k = 1..5**, en calendriers de scatter (15 panneaux). Utilise volontairement `NaiveMarkovChain(order=k)`, seul modèle qui conditionne sur exactement k cellules. Montre **délibérément** l'artefact de sur-apprentissage : à mesure que k grandit, S_cond → 0 et P^max → 1. |
| `PREDICTABILITY_heldout_bound.py` | Corrige cet artefact par une **entropie held-out** (validation croisée, lissage de Laplace + back-off vers l'unigramme). La cross-entropie surestime l'entropie vraie, donc la borne obtenue est **conservatrice** — c'est le bon comportement. |
| `PREDICTABILITY_lempelziv_bound.py` | Borne **Lempel-Ziv** (Song et al. 2010), déclinée par ordre maximal en plafonnant la longueur des correspondances. ⚠️ L'estimateur converge lentement : lire les moyennes par jour, pas les valeurs par utilisateur. |

**C. Prédiction de mouvement (`MOVEMENT_PREDICTION_*`)**

Reformulation binaire : au lieu de prédire *quelle* cellule, prédire **si** l'utilisateur va changer de cellule au prochain enregistrement. Travaille sur les cellules **fusionnées** pour ne capturer que les vrais déplacements.

| Fichier | Rôle |
|---|---|
| `MOVEMENT_PREDICTION_analyze_user_movements.py` | Étude préalable : distribution de l'heure du **premier** et du **dernier** déplacement de la journée, par jour, week-ends en rouge. |
| `MOVEMENT_PREDICTION_extract_features.py` / `…_V2.py` | Extraction des features + label. La V1 génère **tous** les points de chaque utilisateur, la V2 tire **un seul** point aléatoire par utilisateur. ⚠️ 20–40 min de calcul. |
| `MOVEMENT_PREDICTION_fit_features.py` / `…_V2.py` | **XGBoost** et **RandomForest** (gestion du déséquilibre par `scale_pos_weight`, early stopping) ; sorties : modèles `.joblib`, métriques JSON, ROC/PR, importances, calibration, métriques vs seuil. Référence indiquée dans le code : **F1 ≈ 0.60, ROC-AUC ≈ 0.84**. |
| `MOVEMENT_PREDICTION_train_lstm.py` | Alternative séquentielle : donner **la séquence** des vecteurs de features à un **LSTM bidirectionnel** (l'idée étant que le signal est dans la *dérivée* des features), avec un MLP résiduel et XGBoost en comparaison. |
| `MOVEMENT_PREDICTION_analyze_features_usage.py` | Analyse **SHAP** : importance globale, beeswarm, dependence plots, et surtout **quelles features trompent le modèle sur les faux positifs / faux négatifs**. |
| `MOVEMENT_PREDICTION_mobility_explorer.py` | Outil interactif : saisir un `user_id` et tracer (Plotly) sa distance à la première cellule de la journée au fil du temps. |

**Les features** tournent autour de la notion de **cellule d'ancrage** (`anchor cell`, la cellule de référence de la journée) :

- *Position vs ancre* : `is_at_anchor_cell`, `trip_phase`, `anchor_dominance_ratio`
- *Déclenchement* : `has_departed_today`, `first_departure_elapsed_h`, `time_in_anchor_before_departure_h`
- *Retour* : `returned_to_anchor`, `time_since_return_h`
- *Dynamique* : `local_acceleration`, `n_complete_trips`, `current_trip_duration_h`, `is_oscillating`, `phase_transition_signal`
- *Profil* : `out_of_anchor_entropy`, `last_n_distinct_cells`, `recent_record_density`, et en V2 des features d'entropie glissante (`running_entropy`, `running_cond_entropy`, `running_pmax`)

Le **label** : on coupe l'historique de l'utilisateur à un point donné, et `label = 1` si la cellule suivante diffère de la dernière du contexte.

### 5.7 `deep_learning/` — Transformer TUPE

| Fichier | Rôle |
|---|---|
| `create_cellid_map.py` | Mappings `cellid` → entier : `cell_map.json` (0-indexé, pour le modèle de time-weight) et `cell_map_start_1.json` (1-indexé, l'indice 0 étant réservé au padding), plus `day_map.json`. L'en-tête explique **pourquoi les jours ne sont pas encodés finement**. |
| `encode_and_convert_csv_to_pytorch.py` | Encode les séquences en tenseurs `(cells, times, mask)` de longueur fixe **512**, en ignorant les utilisateurs à moins de **8** enregistrements ; timestamps normalisés par 86400 ; split 60/20/20 → `encoded_{train,val,test}.pt`. |
| `dataset.py` | `MobilityDataset(Dataset)` — charge ces tenseurs. |
| `models.py` | `MobilityTransformer` avec attention **TUPE** (*Untied Positional Encoding*, Ke et al. 2020) : les scores d'attention séparent strictement le flux *contenu* (embedding de cellule) du flux *position* (projection MLP du timestamp), au lieu de les additionner. Défauts : `d_model=128`, `n_heads=4`, `n_layers=2`. |
| `train.py` | Boucle complète : teacher forcing décalé d'un pas, loss `CrossEntropyLoss` sur les positions **non paddées** et **≥ `MIN_CONTEXT` = 6**, `OneCycleLR`, AdamW, clip de gradient, reprise depuis un checkpoint. **~27 min par epoch.** |

Convention d'indices : **0 = PAD**, **1…N = cellules**, **N+1 = EOS** (`vocab_size = N+2`).

> **État :** infrastructure complète et cohérente, mais **aucun résultat consolidé** dans le dépôt, et **pas de script d'évaluation finale** (voir §9).

---

## 6. Enchaînement des traitements

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
    └─► Machin_learning/transition_matrix.py      ──► results/numpy/transition_matrix_*.npy
            └─► transition_emtropy.py             ──► user_entropies_{merge}.npy
                    ├─► transition_entropy_by_period.py
                    ├─► maximal_previsibility.py  (P^max, importé partout ailleurs)
                    └─► plot_from_csv/plot_hist_entropy*.py, plot_entropies_by_*.py
```

### Pipeline 3 — Baseline Markov

```
Database/no_duplicate/
    ├─► markov_baseline.py             ──► markov_accuracy_no_merge.npy
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
                    └─► PREDICTABILITY_vs_accuracy.py / _by_order_calendar.py
                        / _heldout_bound.py / _lempelziv_bound.py
```

### Pipeline 5 — Prédiction de mouvement

```
Database/no_duplicate_merge_2g3g/  (cellules fusionnées)
    └─► MOVEMENT_PREDICTION_extract_features[_V2].py ──► features CSV
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

---

## 7. Résultats acquis — à ne pas refaire

### Markov d'ordre 1 sur le dataset complet

Configuration : seuil de gap 4 h 10, utilisateurs à **entropie nulle exclus** (ils n'ont jamais quitté une seule cellule ; Fano leur donne P^max = 1 par convention et ils gonflent les moyennes). 803 530 utilisateurs prédits → **740 029 conservés**.

| Mesure | Valeur |
|---|---|
| Accuracy séquentielle (causale) | **0.555** (médiane 0.571) |
| Accuracy split aléatoire 70/30 | 0.592 |
| P^max moyen (via S_unc) | 0.575 |
| Corrélation accuracy / P^max | **0.844** |
| Utilisateurs avec accuracy > P^max | 38,8 % |

Deux enseignements : **voir seulement le passé ne coûte que ~3,6 points d'accuracy** (le prix honnête de la contrainte causale, et il est modeste), et la corrélation de 0.844 montre que le modèle réussit là où la théorie prédit qu'il peut réussir.

**Effet week-end net** : accuracy 0.593 le week-end contre 0.544 en semaine (P^max 0.609 vs 0.564). Les gens sont plus prévisibles le week-end — cohérent avec les profils week-end distincts déjà vus en classification.

### Limite méthodologique importante

38,8 % des utilisateurs dépassent leur P^max, et ce **n'est pas une erreur** : ce P^max dérive de **S_unc**, l'entropie *non corrélée*, qui ne connaît que la distribution des fréquences de visite. Fano appliqué à S_unc donne le plafond d'un prédicteur **sans mémoire**. Or Markov exploite l'**ordre** — exactement l'information que S_unc jette. Le dépasser est donc attendu ; c'est même le point central de Song et al. (Π^unc < Π^max).

À présenter comme *« comparaison au plafond d'un prédicteur sans mémoire »*, jamais comme une borne absolue. C'est précisément ce trou méthodologique que les scripts `PREDICTABILITY_*` sont venus combler ensuite (entropie conditionnelle, held-out, Lempel-Ziv).

### Mesure de contexte

Sur l'ensemble de `no_duplicate` (15 jours), **72,0 % des enregistrements consécutifs sont dans la même cellule** — le nettoyage « no_duplicate » ne supprime pas les répétitions consécutives. *(Le chiffre de 43,6 % longtemps cité ici avait été mesuré sur `sample_for_training`, échantillon à forte entropie : il n'est pas représentatif.)* Sur un jour de cet échantillon difficile (42 133 transitions) :

| Stratégie | Accuracy |
|---|---|
| Implémentation d'alors (fallback = cellule la plus fréquente) | 0.313 |
| « l'utilisateur ne bouge pas » (baseline triviale) | **0.434** |
| Markov + fallback « ne bouge pas » | 0.425 |
| Markov + fallback + départage des ex æquo vers « rester » | 0.437 |

Et **32,9 % des prédictions sont en cold-start** (cellule de départ jamais vue). Corriger le fallback cold-start (prédire la cellule actuelle plutôt que le mode global) valait **+0.11 d'accuracy** sur le jour testé.

### Ce qui a été tenté sans succès (documenté, ne pas refaire)

| Piste | Où c'est documenté |
|---|---|
| Corréler le délai du contexte au délai de la prochaine transition | En-tête de `5_correlation_time.py`, en gros caractères |
| Apprendre un poids par (heure, cellule) par descente de gradient | `train_time_weight.py` |
| Estimer P^max par ordre avec l'entropie plug-in sur une seule journée | `PREDICTABILITY_by_order_calendar.py` : l'entropie s'effondre et P^max → 1 dès k ≥ 2. C'est ce qui a motivé la version held-out |
| Une méthode heuristique de score move/stay combinant les features à la main | Supprimée de `models.py` (voir §9) : remplacée par l'apprentissage XGBoost |

---

## 8. Bugs trouvés et corrigés (historique à connaître)

Ces corrections **invalident des chiffres produits avant elles**. Si vous retrouvez d'anciens résultats, vérifiez de quel côté de ces corrections ils tombent.

1. **Le N de l'inégalité de Fano.** `P^max` était calculé avec `N` = *nombre de records* de l'utilisateur, alors que Fano attend `N` = *nombre d'états distincts* (cellules distinctes visitées, + `outside` le cas échéant). Sur 2 731 utilisateurs du 12/03 : N moyen 145.6 → **10.6**, P^max moyen 0.767 → **0.580**, part des utilisateurs > 0.8 : 42,6 % → **8,1 %**. `transition_emtropy.py` stocke désormais `n_states` comme 5ᵉ champ du `.npy`, et `maximal_previsibility.py` lève une erreur explicite sur l'ancien format.

2. **L'intervalle de recherche de `brentq`.** Révélé par la correction précédente. La racine était cherchée sur `[1e-10, 1-1e-10]`, où la fonction de Fano a le **même signe aux deux bornes** dès que `S > log2(N-1)` — donc **toujours pour N = 2**. `brentq` levait `ValueError` et la fonction renvoyait silencieusement `NaN`, sur **37,8 %** des utilisateurs une fois le N corrigé. Corrigé en bornant sur `[1/N, 1]`, où le changement de signe est garanti et où la racine a un sens (prévisibilité au moins égale au hasard). Après correction : **0 NaN**.

3. **Le back-off du VOMM sautait l'ordre 1.** `_recursive_prob` s'arrêtait à `order == 1` en renvoyant directement l'unigramme, si bien que `train_data[1]` (les vrais comptages « une cellule → cellule suivante ») n'était **jamais consulté** pour le calcul de probabilité : la récursion passait de l'ordre 2 directement à l'unigramme sans contexte. `VOMM_V4` avait déjà la bonne borne (`order == 0`) ; `VOMM` et `VOMM_V5` ont été alignés dessus. **Toutes les métriques VOMM produites avant cette correction sont à relancer.**

---

## 9. Dette technique — état après nettoyage

**Corrigé :**

- `Metrics.top_k_accuracy` et `map_k` étaient **définies deux fois** chacune (la 2ᵉ écrasant silencieusement la 1ʳᵉ) → une seule définition, documentée.
- `convert_lat_lon_distance_to_meter` était défini **deux fois avec des signatures différentes** dans `simple_predictor/utils.py` → une seule, alignée sur les autres copies du dépôt.
- `MovementPredictor.predict_movement` appelait `extract_features` avec 4 arguments (elle en prend 2) et lisait des features d'une ancienne version qui n'existent plus → **méthode supprimée**, la décision move/stay étant apprise par XGBoost / LSTM.
- `VOMM_single_prediction.py` utilisait une API obsolète (`VOMM(counts=…, context_totals=…)`, `prepare_ngrams(computed_ngrams_filepath=…)`) et ne tournait plus → **réécrit** contre l'API actuelle, avec vérification des prérequis.
- `N_CELLS = 369` était **codé en dur** dans `deep_learning/train.py` → lu depuis `cell_map_start_1.json`, le mapping qui a servi à encoder les `.pt` (les deux ne peuvent plus diverger).
- `deep_learning/loss.py` et `test.py` étaient **vides** → supprimés (la loss est définie dans `train.py`).
- `separate_data_by_letter_code` (`utils.py`) levait un `IndexError` quand la colonne `letters` ne contenait aucune valeur manquante, et filtrait en dur sur `"letters"` en ignorant son paramètre `column_name` → corrigé dans les deux copies.
- Import mort `from typing import Counter` qui masquait `collections.Counter` dans `STATS_get_start_sequece.py` → supprimé.

**Restant :**

- **Trois marges de gap différentes** (4h+30s / 4h+60s / 4h+10min) selon les dossiers. Valeurs conservées telles quelles pour ne pas invalider les résultats existants, mais les commentaires trompeurs qui prétendaient qu'elles étaient identiques ont été corrigés. À unifier un jour, en connaissance de cause.
- **Code volontairement dupliqué** : `utils.py` existe en 3 copies (racine, `important_cells_work/`, `simple_predictor/`), et `get_cell_code` / `MERGE` / `get_day` sont recopiés dans presque chaque script. Modifier l'un impose de vérifier les autres.
- **Pas de `requirements.txt`.** Dépendances relevées dans les imports : `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `tqdm`, `geopy`, `folium`, `plotly`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `torch`.
- **Pilotage par commentaires.** Les configurations d'expérience sont activées en décommentant des blocs : l'historique des runs n'est pas rejouable en un clic, et deux résultats ne sont pas garantis d'avoir été produits avec les mêmes réglages.
- **Pas d'évaluation du Transformer.** `train.py` a une fonction `evaluate()` interne, mais rien ne produit de métriques de test comparables à celles du VOMM.

---

## 10. Chantiers ouverts

1. **Relancer les métriques VOMM** après la correction du back-off (§8.3) — c'est le préalable à toute comparaison.
2. **Un tableau de comparaison unifié.** Le livrable qui manque : tous les modèles (Markov d'ordre fixe, VOMM, VOMM+boosts, Transformer, classifieur move/stay) sur **le même split de test** et les mêmes métriques (ACC@1/3/5, MAP@k, log-loss). `COMPARE_markov_vs_vomm.py` en donne le patron : c'est le seul script qui garantit déjà des points de prédiction identiques entre deux modèles.
3. **Aller au bout du Transformer** : il n'est configuré que pour 3 epochs, et il n'a pas de procédure d'évaluation. Récupérer son ACC@1 dira si le coût du deep learning se justifie sur seulement 2 semaines de données.
4. **Prédicteur en deux étages.** Étant donné les 72 % de self-transitions, un étage 1 « bouge / bouge pas » (le classifieur move/stay existe déjà) suivi d'un étage 2 « où ? » (VOMM, seulement quand un mouvement est prédit) est la piste la plus prometteuse pour améliorer l'ACC@1.
5. **Exploiter les bornes held-out et Lempel-Ziv** : maintenant qu'elles existent, comparer l'accuracy des modèles à ces plafonds-là (et non plus au seul P^max issu de S_unc) donne enfin une lecture méthodologiquement correcte de « à quelle distance de l'optimum on est ».
6. **Évaluer les poids temporels** (`train_time_weight.py`) face au VOMM nu, ou acter leur abandon.

---

## 11. Par où commencer concrètement

1. **Lire le §3 (format des données)** — sans ça, rien du code n'est lisible ; l'idiome `line[8::2]` / `line[9::2]` est partout.
2. **Lire [NOTES_prediction_prochaine_station.md](Python/Machin_learning/NOTES_prediction_prochaine_station.md)** — le raisonnement complet sur la prévisibilité, les deux protocoles d'évaluation et leurs limites.
3. **Lire `generalised_classification_users.py`** — il définit le vocabulaire de classes d'utilisateurs employé ailleurs.
4. **Lire `models.py` de `simple_predictor/`**, en particulier `VOMM._recursive_prob()` — c'est le cœur mathématique du projet.
5. **Vérifier la présence des données** : `Database/no_duplicate/` doit contenir 15 fichiers. Sinon tout le reste est bloqué (les variantes du dataset ne sont plus régénérables par un script du dépôt).
6. **Vérifier `results/predictions/`** avant de lancer un pipeline de prédiction : `train_test/`, `simple_predictor/transition_matrix/` et les métriques doivent exister, sinon relancer le pipeline 4 (§6).
