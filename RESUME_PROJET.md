# Résumé du projet — Analyse et prédiction de mobilité à partir de données cellulaires

> Document de reprise. Il décrit **ce qu'est le projet**, **le format des données**, **le rôle de chaque dossier et de chaque script**, **l'ordre dans lequel les choses s'enchaînent**, et enfin **l'état d'avancement et les pièges** à connaître avant de reprendre le travail.
>
> Rédigé à partir de la lecture du code, des deux `README.md` existants ([python/README.md](python/README.md), [results/README.md](results/README.md)) et de l'historique git (mars → juin 2026).

---

## 1. Objectif du projet

Le projet exploite un **jeu de données de traces de téléphonie mobile** (Tchéquie, mars/avril 2014) : pour chaque utilisateur et pour chaque journée, on dispose de la suite des **antennes (cellules) auxquelles son téléphone s'est connecté**, avec l'horodatage de chaque connexion.

Deux grands axes de travail ont été menés, dans cet ordre chronologique :

1. **Analyse et caractérisation du dataset** (mars → avril 2026)
   Comprendre la donnée : combien d'enregistrements par personne, à quelles heures, quelles cellules sont les plus fréquentées, combien de temps y reste-t-on, quelles anomalies existent (le fameux « vendredi 21/03 »), et **classifier les utilisateurs** selon leur profil de présence sur la journée.

2. **Prédiction de mobilité** (mai → juin 2026)
   Prédire la **prochaine cellule** que va visiter un utilisateur à partir de son historique de la journée. Trois familles d'approches ont été tentées :
   - modèles markoviens d'ordre variable (**VOMM**) — c'est la piste la plus aboutie ;
   - un **Transformer** (PyTorch, encodage positionnel TUPE) — code écrit, entraînement lancé, pas de résultat consolidé ;
   - une reformulation en **classification binaire « bouge / ne bouge pas »** (XGBoost / RandomForest / LSTM sur features construites à la main) — c'est le chantier en cours au moment de l'arrêt.

Le fil rouge implicite : la prédiction « naïve » qui consiste à répondre *« il restera là où il est »* est déjà très bonne (les gens sont majoritairement immobiles), donc tout l'enjeu est de **détecter les instants où l'utilisateur bouge réellement**.

---

## 2. Le format des données — à lire absolument

Tout le code manipule des CSV **sans en-tête**, séparés par des **points-virgules**, où **une ligne = un utilisateur pour une journée**. Un fichier = un jour.

```
user_id ; age ; gender ; unknown ; letters ; BS1 ; BS2 ; n_records ; cell₁ ; ts₁ ; cell₂ ; ts₂ ; … ; cellₙ ; tsₙ ;
   0       1       2        3         4       5     6       7          8      9     10     11
```

| Index | Champ | Description |
|---|---|---|
| 0 | `user_id` | Identifiant anonymisé de l'utilisateur |
| 1 | `age` | Âge supposé — **renseigné pour ~40 % des utilisateurs seulement** |
| 2 | `gender` | `F` / `M` — même taux de remplissage |
| 3 | `unknown_numbers` | Champ numérique fourni par l'opérateur, **signification jamais élucidée** (ressemble à un code postal) |
| 4 | `letters` | Code opérateur à 4 lettres (`CEBU`, `BUNE`, `CETR`, `NUDM`, `CENU`, `CEZB`, `NUDU`, `NUNU`, ou vide) — **sémantique inconnue**, utilisé comme proxy de « comportement » dans les premières analyses |
| 5 | `BS1` | Cellule « domicile » supposée, fournie par l'opérateur |
| 6 | `BS2` | Cellule « activité » supposée, fournie par l'opérateur (souvent vide) |
| 7 | `n_records` | Nombre d'enregistrements de la ligne |
| 8, 10, 12… | `cellᵢ` | Identifiant de cellule (ex. `BSOBRE1`) |
| 9, 11, 13… | `tsᵢ` | Timestamp **en secondes depuis minuit** (0 → 86400), donc **relatif à la journée** |

Extrait réel :

```
10634423;51;F;;CEBU;BSOBRE1;;19;BSOBRE1;1110;BSOBRE1;6555;BSOBRE1;10822;…;BSOBRE1;86025;
```

**Points d'attention :**

- Les lignes du `raw_dataset` se terminent par un `;` → un dernier champ vide parasite. Plusieurs scripts font `row[:-1]` pour s'en débarrasser.
- `cells = row[8::2]` et `timestamps = row[9::2]` est **l'idiome utilisé partout** dans le code.
- Le premier caractère du `cellid` code la technologie : **`B` et `D` → 2G**, **`U` et `V` → 3G**. Une même antenne physique apparaît donc sous plusieurs `cellid`.
- **Fusion des cellules en « stations de base »** — récurrente dans tout le projet, via la fonction `merge_cell_id` (dupliquée dans plusieurs fichiers) :
  ```python
  def merge_cell_id(cell: str):
      if cell.startswith(('B','D')):  return cell[1:-1]   # 2G
      else:                           return cell[1:-3]   # 3G
  ```
  Objectif : un utilisateur qui bascule entre `BKVPER1` et `BKVPER3` **ne bouge pas réellement**, il change juste de secteur/technologie. C'est indispensable pour la prédiction de mouvement.

### Périmètre géographique

- **`cd_142`** — la zone étudiée dans quasiment tout le projet. **15 jours**, du **2014-03-12 au 2014-03-26**, ~**92 000 lignes/jour**, **369 cellules**.
- **`cd_010`** — une première zone, explorée en début de stage puis abandonnée. **7 jours**, du **2014-04-07 au 2014-04-13**.
- **`cd_170`** — une troisième zone, dont seule la liste de cellules est présente.

---

## 3. Arborescence générale

```
Stage3A/
├── Database/                 ← Données brutes et dérivées (4,4 Go, NON versionné)
│   ├── cells/                    Coordonnées lat/lon de toutes les antennes
│   └── cd_142_dataset/           Le dataset + ses variantes générées
│
├── python/                   ← Tout le code
│   ├── cd_010/                   Analyses de la 1ʳᵉ zone — OBSOLÈTE
│   ├── cd_142/
│   │   ├── analysis/             Analyse statistique + graphes de la zone 142
│   │   ├── important_cells_work/ Détection cellule domicile/activité + classification des users
│   │   └── dataset_creation/     Préparation des données pour la prédiction (pipeline 0→6)
│   ├── simple_predictor/         Modèles markoviens (VOMM) + prédiction de mouvement
│   ├── deep_learning/            Transformer PyTorch
│   └── other/                    Scripts jetables
│
├── results/                  ← Sorties de tous les scripts (NON versionné, sauf README)
│   ├── cd_010/ , cd_142/         json / html / plots / intermediate_result
│   └── predictions/              ⚠️ ABSENT sur le disque — à régénérer
│
├── final_python/             ← Scripts « propres » retenus pour le rapport/soutenance
└── final_results/            ← Figures finales correspondantes (versionnées)
```

---

## 4. Détail des dossiers

### 4.1 `Database/` — les données

**Non versionné** (`.gitignore`), ~4,4 Go. Seul `Database/cells/` est dans git.

#### `Database/cells/`
| Fichier | Contenu |
|---|---|
| `cells.csv` | **Référentiel complet** : `cellid;lat;lon;x;y` pour ~11 030 antennes de tout le pays. C'est le fichier chargé par la plupart des scripts pour convertir un `cellid` en coordonnées. |
| `cd_010_cells.csv`, `cd_142_cells.csv`, `cd_170_cells.csv` | Sous-ensembles par zone (369 cellules pour cd_142). |
| `cells_of_dataset_cd_142.csv` | Cellules réellement observées dans le dataset cd_142. |

#### `Database/cd_142_dataset/`
Chaque sous-dossier contient **15 fichiers CSV** (un par jour), et représente **une variante du même dataset**. Toutes sont produites par le script [GENERATION_generate_all_csv.py](python/cd_142/analysis/GENERATION_generate_all_csv.py) à partir de `raw_dataset/`.

| Dossier | Ce qu'il contient | À quoi ça sert |
|---|---|---|
| `raw_dataset/` | Les données brutes livrées (`AAAA-MM-JJ_vektory.csv`) | Source de tout |
| `no_duplicate/` | Les `(cellule, timestamp)` strictement identiques consécutifs sont supprimés | **La variante la plus utilisée** dans le projet |
| `no_duplicate_max_512_records/` | Idem + lignes tronquées à ≤ 512 enregistrements | Analyses statistiques : 512 = 2⁹, et **99,4 % des utilisateurs sont en dessous** ; au-delà ce sont des artefacts (M2M, systèmes automatiques) |
| `no_duplicate_merge_2g3g/` | Idem `no_duplicate` mais les cellules sont **fusionnées en stations de base** | Prédiction de **mouvement réel** |
| `without_records/` | Seulement les 8 colonnes de métadonnées, sans les enregistrements | Analyses rapides (distributions, âge, genre) sans charger 4 Go |
| `with_distance/` | La colonne `letters` (index 4) est **remplacée** par la distance totale parcourue dans la journée, en mètres | Statistiques de mobilité |
| `deep_learning/` | **Vide** — dossier prévu pour les tenseurs encodés, jamais rempli à cet emplacement |

> ⚠️ Attention à `with_distance/` : la colonne 4 change de sens. Un script qui suppose `letters` en 4 donnera n'importe quoi sur cette variante.

---

### 4.2 `python/cd_010/` — première zone (OBSOLÈTE)

11 scripts écrits en tout début de stage (commits de fin mars 2026) sur la zone `cd_010` (7 jours d'avril 2014). Ils ont ensuite été **réécrits et améliorés dans `python/cd_142/analysis/`**.

**Ils ne tournent plus** : leurs chemins pointent vers `../../Database/dataset/…`, une arborescence qui n'existe plus (ni `dataset/`, ni `added_n_records/`). Ils sont conservés à titre historique.

Contenu (chaque script a son équivalent modernisé côté `cd_142`) :
`insert_number_of_records.py` (ajoute la colonne `n_records`, qui n'était pas dans le brut de cd_010), `generate_all_csv.py`, `create_transition_matrix.py`, `get_cell_dataset_origin.py`, `get_global_data_analysis.py`, `get_most_recorded_cells.py`, `get_most_stayed_cells.py`, `plot_comparison_2g_and_3g_records.py`, `plot_records_distribution.py`, `plot_timestamp_distribution.py`, `utils.py`.

> **Recommandation :** ne pas y toucher, et se référer uniquement à `cd_142/analysis/`.

---

### 4.3 `python/cd_142/analysis/` — analyse statistique de la zone 142

C'est le cœur de la première phase du stage. Les noms sont préfixés selon [la convention du README](python/README.md) :

- `GENERATION_*` → produit des variantes du dataset
- `STATS_*` → produit des `.json` / `.csv` de statistiques
- `GRAPH_*` → produit des graphes (networkx / gephi)
- `plot_*` → produit des `.png` (matplotlib) ou `.html` (plotly)

| Script | Rôle |
|---|---|
| **`GENERATION_generate_all_csv.py`** | **Le générateur central.** À partir de `raw_dataset/`, produit toutes les variantes de `Database/cd_142_dataset/`. On active/désactive chaque variante via le dictionnaire booléen `CSV_TO_GENERATE` en tête de fichier. C'est ici qu'on ajoute une nouvelle variante. |
| `GENERATION_base_station_list.py` | Produit `cd_142_base_stations.csv` : la version « fusionnée » du référentiel de cellules (une ligne par station physique). |
| `GENERATION_create_transition_matrix.py` | Matrice de transition cellule A → cellule B sur les 15 jours. Génère aussi une version **à diagonale nulle** (P(A→A)=0) — car sinon la transition vers soi-même écrase tout — ainsi que des variantes par jour, par heure, par classe, et des versions qui **ignorent les transitions séparées de plus de 4 h**. |
| `STATS_get_global_data_analysis.py` | Le « rapport statistique » du dataset : par jour, nombre d'enregistrements (max/moyenne/médiane), timestamps moyens du premier et du dernier enregistrement, temps moyen entre deux enregistrements, distance moyenne entre cellules consécutives, distance totale parcourue. → `results/cd_142/json/global_data_analysis_*.json`. |
| `STATS_get_global_data_analysis_plots.py` | Même chose mais en figures Plotly (demandées par l'encadrant). |
| `STATS_get_most_recorded_cells.py` | Les cellules les plus **visitées** (une visite comptée **une seule fois par utilisateur**, pour éviter que les utilisateurs immobiles ne créent des « supercellules »). |
| `STATS_get_most_stayed_cells.py` | Les cellules où l'on **reste le plus longtemps**, avec plusieurs découpages horaires (4h–20h, 5h–19h, 6h–18h et leurs compléments nuit/matin) pour distinguer domicile et lieu d'activité. |
| `STATS_get_cell_dataset_origin.py` | Répartit les cellules du dataset entre les 3 zones (cd_010 / cd_142 / cd_170). |
| `STATS_get_user_by_main_cell.py` | Compte les utilisateurs par « comportement » (code `letters`), par jour et au total. |
| `STATS_get_proportion_of_user_with_X_records.py` | Proportion d'utilisateurs ayant 1, 2, … 8 enregistrements. **Résultat structurant :** une part énorme des utilisateurs a très peu d'enregistrements et est donc inexploitable pour la prédiction. |
| `STATS_special_friday_analysis.py` | Analyse ciblée du **vendredi 21/03/2014**, où une plage horaire est quasiment vide : problème de collecte côté opérateur. Cette anomalie est la raison pour laquelle les jours ne sont pas encodés finement dans les modèles. |
| `GRAPH_get_graph_popularity_cell.py` / `…_merge.py` | Construisent le graphe des cellules (nœuds = cellules, arêtes = transitions) au format `.gexf`, en version normale et fusionnée 2G/3G. |
| `GRAPH_apply_community_alg_gephi.py` / `…_merge.py` | Appliquent des algorithmes de détection de communautés (bibliothèque `cdlib` : Leiden, etc.) sur ces graphes et évaluent modularité, densité interne, conductance. |
| `GRAPH_community_analysis.py` | Métriques complémentaires pour comparer graphe fusionné vs non fusionné (ratio de voisinage, ratio de voisins intra-communauté). |
| `plot_records_distribution.py` | Distribution du nombre d'enregistrements par utilisateur, par jour + vue d'ensemble. *(x = nombre d'enregistrements, y = nombre d'utilisateurs ayant exactement ce nombre.)* |
| `plot_timestamp_distribution.py` | Heatmap des timestamps (résolution 1 min × 15 jours) — **c'est la figure qui révèle l'anomalie du vendredi**. |
| `plot_comparison_2g_and_3g_records.py` | Comptage 2G vs 3G par jour. |
| `plot_records_distribution_by_behaviour.py` | Distribution du nombre d'enregistrements par code `letters`, en dashboards HTML interactifs. |
| `plot_age_and_gender_dist_by_behaviour.py` | Distributions d'âge et de genre par comportement (dashboards HTML). |
| `utils.py` | **Boîte à outils commune** : ouverture de CSV avec/sans en-tête (`has_header`), `get_day()`, `is_weekend()`, calcul de distance géodésique (`geopy`), `compute_distance_travelled_by_user()`, séparation par code `letters`. Contient aussi du code QGIS commenté (visualisation de trajectoires en GeoJSON). |

---

### 4.4 `python/cd_142/important_cells_work/` — cellules importantes et classification des utilisateurs

Deuxième bloc de la phase d'analyse, et **conceptuellement le plus important pour la suite**. Objectif : déterminer, pour chaque utilisateur, sa **cellule domicile** et sa **cellule d'activité**, puis en déduire une **classe de comportement journalier**.

Les scripts sont numérotés dans leur ordre logique d'exécution.

| Script | Rôle |
|---|---|
| `1_get_user_important_cells_from_dataset.py` | Version « naïve » : compte les utilisateurs ayant un `BS1`/`BS2` renseigné **directement d'après les colonnes du dataset**. Sert de point de comparaison. |
| `2_get_user_important_cells_handmade.py` | **Redétermine soi-même** domicile et activité, avec des plages horaires choisies et surtout en raisonnant sur la **position physique** plutôt que sur le `cellid` (`AAAAAA101` et `AAAAAA102` = même endroit). Produit `results/cd_142/intermediate_result/classified_dataset_merge_{simple,2g3g}.csv`. |
| `2_get_user_important_cells_handmade_statistics.py` | Statistiques sur ces résultats (comptages par cellule domicile / activité). |
| `2_barplot_user_important_cells_handmade.py` | Barplots comparant les catégories : `both` / `home cell only` / `activity cell only` / aucune. |
| `2_user_categories_plots.py` | Autres visualisations des mêmes catégories, avec distinction semaine / week-end. |
| `3_full_stats_for_1_record_users.py` | Étude spécifique des utilisateurs à **un seul enregistrement** dans la journée : combien, et à quelle heure. Contient `shortest_interval()`, qui calcule le plus petit intervalle temporel contenant 80 % des données. |
| **`4_user_classification.py`** | **Le script clé.** Découpe la journée en **6 fenêtres de 4 h** et classe chaque utilisateur selon le motif de présence dans ces fenêtres (voir tableau ci-dessous). Produit `results/cd_142/intermediate_result/cell_classification*.csv` avec, par utilisateur et par jour : la classe, le nombre de trous > 4 h, et la fréquence d'enregistrements dans chaque fenêtre. |
| `4_count_user_by_category.py` | Agrège les effectifs par jour et par classe. |
| `4_plot_timestamp_distribution_by_class.py` | Heatmap des timestamps **par classe** — permet de vérifier visuellement que les classes ont du sens. |
| `4_search_weird_results.py` | Densité d'utilisateurs par cellule pour la classe majoritaire vs les autres, pour investiguer des résultats surprenants. |
| `utils.py` | Version allégée : `get_day()` et `is_weekend()` seulement. |
| `deprecated/` | Anciennes définitions de « cellule d'activité » (`old_def.py` = continu, `my_def.py` = non continu) et leurs statistiques. Conservées pour retracer l'évolution du raisonnement. |

**Le schéma de classification** (`4_user_classification.py`, fenêtres 0-4h, 4-8h, 8-12h, 12-16h, 16-20h, 20-24h) :

| Classe | Signification |
|---|---|
| `1` | Présent dans **les 6 fenêtres** — utilisateur le mieux couvert (c'est la classe utilisée pour les tests de prédiction) |
| `2` – `5` | Présence **continue** sur une sous-partie de la journée (2 : 0h–20h, 3 : 4h–24h, 4 : 4h–20h, 5 : autre plage continue) |
| `6` | Présent **matin et soir uniquement**, rien entre les deux |
| `7`, `8`, `9` | Présent dans **une seule fenêtre** (respectivement 0-4h, 20-24h, autre) |
| `10` – `16` | Présence **discontinue**, sous-classée par la taille du plus grand trou (4-6h, 6-8h, … , ≥16h) |
| `21` – `25` | Utilisateurs avec **1 à 5 enregistrements** seulement (`20 + n_records`) — trop peu de données, écartés partout |

---

### 4.5 `python/cd_142/dataset_creation/` — préparation pour la prédiction

Pipeline numéroté qui transforme le dataset en **matériel d'entraînement**. C'est le pont entre la phase d'analyse et la phase de modélisation.

| Script | Rôle |
|---|---|
| `0_make_full_dataset_from_class_or_weekend.py` | Reconstruit **un seul fichier** à partir des 15 jours, en filtrant sur une classe d'utilisateurs (issue de la classification 4.4) et/ou en séparant semaine / week-end. |
| **`1_train_test_split.py`** | Split **80 % train / 20 % test** (mélange aléatoire, `SEED = 67`). Produit `results/predictions/train_test/train_random.csv` et `test_random.csv`. Le fichier contient aussi, **en commentaire**, les variantes de split : sur données fusionnées 2G/3G, sur classe 1 uniquement, avec split de validation. |
| `2a_create_full_ngrams_matrix.py` | **V1** — n-grammes calculés sur **tout** le dataset. |
| `2b_create_train_ngrams_matrix.py` | **V2** — n-grammes calculés **uniquement sur le split train** (c'est la version correcte, sans fuite de données). |
| `2c_create_train_ngrams_matrix_by_class.py` | Idem mais **une matrice par classe** d'utilisateur. |
| `3_deduplicate_csv.py` | Supprime les **répétitions consécutives de cellule** dans un CSV de split. Indispensable : prédire `A → A` n'a aucun intérêt, et les répétitions faussent complètement les métriques. |
| `4_create_train_ngrams_with_time.py` | **V3** — ajoute une matrice des **délais** de transition en parallèle des comptages. ⚠️ **~1 h de calcul et beaucoup de RAM.** |
| `5_correlation_time.py` | Étude de corrélation entre le délai moyen du contexte et le délai vers la cellule suivante. **Conclusion explicitement écrite dans le fichier : aucune corrélation exploitable.** Le temps ne peut donc pas servir de feature directe pour prédire *quand* aura lieu la prochaine transition. Le script est conservé pour ne pas refaire l'essai. |
| `6a_convert_context_to_idx.py` | Convertit les contextes (chaînes) en indices entiers pour PyTorch. |
| `6b_convert_csv_for_gpu.py` | Convertit les CSV en tenseurs (`contexts`, `hours`, `targets`), **un fichier par longueur de contexte**. Ignore les utilisateurs à moins de 6 enregistrements. |
| `stats_ngrams.py` | Statistiques sur les matrices de n-grammes produites (couverture, distribution des comptages). |
| `utils.py` | `find_ngrams()` et `find_ngrams_optimized()` — extraction de n-grammes, avec option de **coupure quand l'écart temporel dépasse un seuil** (`max_gap`), pour ne pas créer de faux contextes à cheval sur une nuit. |

**Format des matrices de n-grammes** (JSON) :

```json
{
  "longueur_de_contexte": {
    "séquence_de_cellules_du_contexte": {
      "cellule_suivante": nombre_d_occurrences
    }
  }
}
```

Les longueurs de contexte vont de **1 à 5** (parfois 6).

---

### 4.6 `python/simple_predictor/` — modèles de prédiction

Le dossier le plus actif de mai/juin 2026. Il contient **deux sous-chantiers distincts** qui partagent le même `models.py`.

#### A. Prédiction de la prochaine cellule (`VOMM_*`, `NAIVE_*`)

| Fichier | Rôle |
|---|---|
| **`models.py`** | Toutes les classes de modèles (voir détail ci-dessous). |
| **`utils.py`** | Classes utilitaires : `HelperVOMM` (préparation des n-grammes, comptages de contextes, unigrammes), `HelperData` (mapping utilisateur → jour/ligne, sauvegardes), **`Metrics`** (`top_k_accuracy`, `map_k`, `log_likelihood`, `negative_log_likelihood`, `perplexity`, `mean_reciprocal_rank`), et un `MobilityDataset` PyTorch. |
| `VOMM_prediction_pipeline.py` | **Pipeline principal d'évaluation.** Charge une matrice d'entraînement + un split de test, instancie le modèle, prédit et écrit les métriques en JSON. La fin du fichier contient **~10 configurations d'expérience préparées et commentées** (avec/sans répétitions, fusionné 2G/3G, par classe, semaine/week-end) : c'est un catalogue d'expériences à décommenter une par une. |
| `NAIVE_prediction_pipeline.py` | Même pipeline pour les baselines `NaiveMarkovChain`. |
| `VOMM_single_prediction.py` | Prédiction sur un seul utilisateur — utile pour déboguer / inspecter. |
| `PLOT_VOMM_metrics_by_discount.py` | Trace les métriques en fonction du paramètre `discount` du modèle. |
| `user_mapping.py` | Construit un index `user_id → (jour, numéro de ligne)` pour retrouver rapidement un utilisateur. |
| `train_time_weight.py` | Tentative d'apprentissage par descente de gradient d'un poids par (heure, cellule) — matrice 24 × 369. **Noté comme n'ayant pas fonctionné** dans le message de commit. |

**Les modèles de `models.py` :**

- **`NaiveMarkovChain`** — chaîne de Markov d'ordre fixe. Baseline.
- **`NaiveMarkovChainWithTimeWeight`** — idem, pondérée par l'heure.
- **`VOMM`** *(le modèle de référence)* — **Variable-Order Markov Model** avec *absolute discounting* et *backoff* récursif, dans l'esprit de Kneser-Ney :

  ```
  P(cible | contexte) = max(count(contexte,cible) − d, 0) / count(contexte)
                        + (d · nb_suffixes_uniques / count(contexte)) · P(cible | contexte_raccourci)
  ```

  La récursion réduit le contexte d'un cran à chaque échec, jusqu'à l'unigramme. Paramètres : `max_order=5`, `discount=0.75`. Un `cache` sur les contextes accélère fortement les prédictions.
  *(Un bug sur la borne de récursion — descente jusqu'à 1 au lieu de 0 — a été corrigé le 29/05, cf. commit `688f68e`.)*
- **`VOMM_V4`** — ajoute un **boost temporel** : les cellules domicile/activité voient leur score majoré selon l'heure.
- **`VOMM_V5`** — ajoute en plus un **profil utilisateur** construit sur son historique avec décroissance exponentielle (`decay=0.95`). `discount` par défaut porté à 0.90. **C'est le modèle utilisé par le pipeline actuel.**

**Métriques suivies** : `ACC@k` et `MAP@k` pour k ∈ {1, 3, 5, 10}, log-vraisemblance moyenne, plus un indicateur « meta » très parlant : la proportion de prédictions où le modèle **répond simplement la dernière cellule du contexte**, en distinguant les cas où il a raison de ceux où il a tort. C'est la mesure du biais d'immobilité évoqué en introduction.

#### B. Prédiction de mouvement (`MOVEMENT_PREDICTION_*`)

Reformulation du problème : au lieu de prédire *quelle* cellule, prédire **binairement si l'utilisateur va changer de cellule** au prochain enregistrement. Suggestion de l'encadrant ; travaille sur les **cellules fusionnées en stations de base** pour ne capturer que les vrais déplacements.

| Fichier | Rôle |
|---|---|
| `MOVEMENT_PREDICTION_analyze_user_movements.py` | **Étude préalable** : comment se comportent les utilisateurs quand ils bougent vs quand ils restent, distribution du premier et du dernier déplacement de la journée. L'en-tête du fichier liste explicitement les **cas non résolus** (déplacements trop courts pour être anticipés). |
| `MOVEMENT_PREDICTION_extract_features.py` | **V1 puis V3/V4** de l'extraction de features (~20–40 min de calcul). Produit un CSV de features + label. |
| `MOVEMENT_PREDICTION_extract_features_V2.py` | Version suivante, avec un jeu de features différent (et un sous-ensemble « late features »). |
| `MOVEMENT_PREDICTION_fit_features.py` / `_V2.py` | Entraînent **XGBoost** et **RandomForest** sur ces features, évaluent sur le test, et sauvegardent modèles `.joblib`, métriques JSON, courbes ROC/PR et importances de features. |
| `MOVEMENT_PREDICTION_analyze_features_usage.py` | Analyse **SHAP** : importance globale, par classe (`move` vs `stay`), interactions entre features, explication locale, et analyse des features dominantes sur les **faux positifs / faux négatifs**. |
| `MOVEMENT_PREDICTION_train_lstm.py` | Alternative : au lieu d'un vecteur de features à l'instant T, donner **la séquence** des vecteurs [t=1…T] à un LSTM, l'idée étant que « le signal est dans la dérivée, pas dans la valeur absolue ». |
| `MOVEMENT_PREDICTION_mobility_explorer.py` | Visualisation interactive Plotly des trajectoires utilisateur sur carte. |

**Les features** (classes `MovementPredictor` et `MovementPredictorV2` dans `models.py`) tournent autour de la notion de **cellule d'ancrage** (`anchor cell`, la cellule de référence de la journée) :

- *Position vs ancre* : `is_at_anchor_cell`, `trip_phase`, `anchor_dominance_ratio`
- *Déclenchement* : `has_departed_today`, `first_departure_elapsed_h`, `time_in_anchor_before_departure_h`
- *Retour* : `returned_to_anchor`, `time_since_return_h`
- *Dynamique* : `local_acceleration`, `n_complete_trips`, `current_trip_duration_h`, `is_oscillating`, `phase_transition_signal`
- *Profil* : `out_of_anchor_entropy`, `last_n_distinct_cells`, `recent_record_density`

Le **label** est construit ainsi : on coupe l'historique de l'utilisateur à un point aléatoire, et `label = 1` si la cellule suivante diffère de la dernière du contexte.

> **État :** le dernier commit (`57ad37e`, 08/06) indique explicitement *« V4 extract feature and fit features **not so good** »*. **C'est le point exact où le travail s'est arrêté.**

---

### 4.7 `python/deep_learning/` — Transformer PyTorch

Approche séquentielle profonde, développée début mai 2026.

| Fichier | Rôle |
|---|---|
| `create_cellid_map.py` | Crée le mapping `cellid` (chaîne) → entier, ainsi qu'un mapping jour → entier (0–14). **L'en-tête explique pourquoi les jours ne sont pas encodés plus finement** : deux semaines sont trop courtes pour apprendre une saisonnalité hebdomadaire, et l'anomalie du vendredi risquerait d'induire le modèle en erreur. |
| `encode_and_convert_csv_to_pytorch.py` | Encode les séquences en tenseurs `(cells, times, mask)` de longueur fixe **512**, en ignorant les utilisateurs à moins de **8** enregistrements. Écrit `encoded_train.pt`, `encoded_val.pt`, `encoded_test.pt`. Convention d'indices : **0 = PAD**, **1…N = cellules**, **N+1 = EOS**. |
| `dataset.py` | `MobilityDataset(Dataset)` — charge ces tenseurs. |
| `models.py` | `MobilityTransformer` avec attention **TUPE** (*Transformer with Untied Positional Encoding*, Ke et al. 2020) : les scores d'attention séparent strictement le flux *contenu* (embedding de cellule) du flux *position* (projection MLP du timestamp normalisé), au lieu de les additionner. Défauts : `d_model=128`, `n_heads=4`, `n_layers=2`, `d_ff=256`. |
| `train.py` | Boucle d'entraînement complète, avec `argparse` (`--save-path`) et sauvegarde de checkpoints. **~27 min par epoch** (indiqué en commentaire). Les `MIN_CONTEXT = 6` premiers tokens ne sont pas prédits. |
| `loss.py`, `test.py` | **Fichiers vides** — prévus, jamais écrits. |

> **État :** infrastructure complète et cohérente, mais **aucun résultat consolidé** n'est présent dans le dépôt. `test.py` vide signifie qu'il n'y a pas de procédure d'évaluation finale.

---

### 4.8 `python/other/` — scripts jetables

| Fichier | Rôle |
|---|---|
| `extract_users.py` | Extrait 100 utilisateurs « intéressants » d'une journée (lundi 24/03) : au moins 15 enregistrements, jamais sortis de la zone (aucun trou > ~4 h), dont 20 immobiles et 80 mobiles. Sert à constituer un petit échantillon d'inspection manuelle. |
| `loaddata.py` | Micro-script de bricolage — contient un `breakpoint()` pour explorer les données en interactif. |

---

### 4.9 `final_python/` et `final_results/` — le livrable

Contrairement au reste, **ces deux dossiers sont versionnés dans git**. Ce sont les scripts et figures **retenus pour la restitution** : versions nettoyées, chemins simplifiés (`MAIN_DIR = Path(__file__).parent.parent`), sorties directement dans `final_results/`.

| Script | Figure produite |
|---|---|
| `plot_heatmap.py` | `final_results/Timestamp_heatmap.png` — heatmap 1 min × 15 jours |
| `plot_timestamp_distribution.py` | `final_results/Timestamp_Distribution_All_Day.png` |
| `plot_records_distribution.py` | `final_results/plots/Record_Distribution_All_Days.png` + un dossier par jour avec les versions plafonnées à 128 et 512 |
| `plot_number_user_by_day.py` | `final_results/number_of_users_by_day.png` |
| `plot_classification_repartition.py` | `final_results/cell_classification_repartition.png` — top 5 des classes par jour + « Other » |
| `utils.py` | Copie de `cd_142/analysis/utils.py` |

> **C'est le meilleur point d'entrée pour comprendre visuellement le dataset avant de plonger dans le code.**

---

### 4.10 `results/` — les sorties

**Non versionné** (sauf `results/README.md`, qui documente en détail le contenu de chaque JSON, PNG et HTML — **à lire**).

```
results/
├── README.md                    ← Description fichier par fichier
├── cd_010/  json/ plots/            Résultats de la zone abandonnée
└── cd_142/
    ├── json/                        Statistiques globales, cellules les plus visitées,
    │   └── time_spent/              temps passé par cellule selon les découpages horaires
    ├── html/                        Dashboards Plotly interactifs (âge, genre, distributions,
    │                                réseau de cellules, exploration de mobilité)
    ├── plots/                       PNG : un dossier par jour + figures globales
    │   ├── behaviour/
    │   ├── important_cells_work/    Barplots des cellules domicile/activité
    │   │   └── CLASSIFICATION/      Barplots par classe d'utilisateur (préfixés 1_, 2_, …)
    │   └── gif/                     evolution_distribution.gif
    └── intermediate_result/         ⭐ Le plus important : CSV de classification par
                                     utilisateur/jour, réutilisés par les scripts de prédiction
```

> ⚠️ **`results/predictions/` est absent du disque.** C'est pourtant le dossier vers lequel pointent **tous** les scripts de `dataset_creation/`, `simple_predictor/` et `deep_learning/` (splits train/test, matrices de n-grammes, métriques, checkpoints). Il faudra le régénérer — voir §6.

---

## 5. Enchaînement des traitements

### Pipeline 1 — Analyse (déjà exécuté, résultats disponibles)

```
Database/cd_142_dataset/raw_dataset/
        │
        └─► GENERATION_generate_all_csv.py
                └─► no_duplicate/ , without_records/ , with_distance/ ,
                    no_duplicate_max_512_records/ , no_duplicate_merge_2g3g/
                        │
                        ├─► STATS_*.py  ──► results/cd_142/json/
                        ├─► plot_*.py   ──► results/cd_142/plots|html/
                        └─► GRAPH_*.py  ──► results/cd_142/gexf/
```

### Pipeline 2 — Classification des utilisateurs (déjà exécuté)

```
no_duplicate/
    └─► 4_user_classification.py
            └─► results/cd_142/intermediate_result/cell_classification*.csv
                    ├─► 4_count_user_by_category.py
                    ├─► 4_plot_timestamp_distribution_by_class.py
                    └─► (réutilisé par les modèles de prédiction)
```

### Pipeline 3 — Prédiction de la prochaine cellule (à régénérer)

```
no_duplicate/  (+ classification)
    └─► 0_make_full_dataset_from_class_or_weekend.py   (optionnel : filtrage classe / week-end)
    └─► 1_train_test_split.py          ──► train_random.csv / test_random.csv
            ├─► 3_deduplicate_csv.py   ──► versions sans répétitions consécutives
            └─► 2b_create_train_ngrams_matrix.py ──► ngrams_matrix_train_random.json
                    └─► VOMM_prediction_pipeline.py ──► metrics/*.json
                            └─► PLOT_VOMM_metrics_by_discount.py
```

### Pipeline 4 — Prédiction de mouvement (chantier en cours)

```
no_duplicate_merge_2g3g/  (cellules fusionnées, classe 1)
    └─► 1_train_test_split.py  ──► class1_train_random.csv / class1_test_random.csv
            └─► MOVEMENT_PREDICTION_extract_features[_V2].py ──► features/*.csv
                    └─► MOVEMENT_PREDICTION_fit_features[_V2].py ──► XGBoost + RandomForest
                            └─► MOVEMENT_PREDICTION_analyze_features_usage.py  (SHAP)
                    └─► MOVEMENT_PREDICTION_train_lstm.py   (variante séquentielle)
```

### Pipeline 5 — Deep learning (infrastructure prête, pas de résultat)

```
no_duplicate_max_512_records/
    └─► create_cellid_map.py                ──► cell_map.json
    └─► encode_and_convert_csv_to_pytorch.py ──► encoded_{train,val,test}.pt
            └─► train.py ──► checkpoints/
                    └─► test.py   ⚠️ VIDE — à écrire
```

---

## 6. Conventions, pièges et prérequis

### Conventions de nommage (issues de [python/README.md](python/README.md))

| Préfixe | Signification |
|---|---|
| `generate` / `GENERATION_` | Produit des variantes du dataset |
| `get` / `create` / `STATS_` | Produit des `.json` et `.csv` |
| `plot` / `PLOT_` | Produit des `.png` (matplotlib) ou `.html` (plotly) |
| `special` | Analyse ponctuelle d'une anomalie |
| `GRAPH_` | Manipulation de graphes |
| `MOVEMENT_PREDICTION_` / `VOMM_` / `NAIVE_` | Chantiers de prédiction |
| `0_`, `1_`, `2a_`… | Ordre d'exécution du pipeline |

**Règles à respecter** si vous ajoutez du code :
- La ou les premières lignes du script **doivent décrire ce qu'il fait**.
- Les chemins d'entrée/sortie **doivent se résoudre à partir de la position du script** :
  ```python
  MAIN_DIR = Path(__file__).parent.parent.parent   # ajuster le nombre de .parent
  ```

### ⚠️ Piège n°1 : deux styles d'import incompatibles

Le dépôt mélange deux conventions, et **elles n'exigent pas le même répertoire de lancement** :

| Style | Fichiers concernés | Comment lancer |
|---|---|---|
| `import python.cd_142.analysis.utils as utils` | 14 scripts de `cd_142/analysis/` (`GENERATION_*`, `STATS_*`, `GRAPH_*`, certains `plot_*`) | **depuis la racine du dépôt**, ex. `python -m python.cd_142.analysis.STATS_get_most_recorded_cells` |
| `import utils` | tout le reste (`cd_010/`, `dataset_creation/`, `important_cells_work/`, `simple_predictor/`, `final_python/`) | **depuis le dossier du script** |

Deux fichiers du **même dossier** peuvent différer : `plot_records_distribution.py` utilise `import utils` alors que ses voisins utilisent le chemin complet. **En cas de `ModuleNotFoundError`, c'est presque toujours ça.** Uniformiser les imports serait un bon premier chantier de nettoyage.

### ⚠️ Piège n°2 : le nombre de `.parent`

`MAIN_DIR` compte un nombre de `.parent` qui **dépend de la profondeur du script**. Un script déplacé d'un dossier casse silencieusement (il écrira ses sorties au mauvais endroit). Certains scripts déduisent aussi le nom du dataset du nom du dossier parent :

```python
dataset_name = Path(__file__).parent.parent.name   # → "cd_142"
```

Renommer un dossier casse donc les chemins.

### ⚠️ Piège n°3 : les données ne sont pas dans git

`.gitignore` exclut `Database/`, `results/`, `Docs/`, `QGIS/`, `cd_010_work/`, `cd_142_work/`. **331 fichiers seulement sont versionnés** : le code, `Database/cells/`, `final_python/` et `final_results/`.

Concrètement, en repartant d'un clone propre, il faut :
1. récupérer `Database/cd_142_dataset/raw_dataset/` (4,4 Go, hors dépôt) ;
2. relancer `GENERATION_generate_all_csv.py` pour reconstruire les variantes ;
3. relancer les pipelines 2 à 5 pour reconstituer `results/`.

**Et même sur la machine actuelle, `results/predictions/` n'existe pas** : tous les splits, matrices de n-grammes et métriques de prédiction doivent être régénérés (pipeline 3).

### ⚠️ Piège n°4 : temps de calcul et mémoire

| Traitement | Coût |
|---|---|
| `4_create_train_ngrams_with_time.py` | ~1 h + beaucoup de RAM |
| `5_correlation_time.py` | 30 min – 1 h + beaucoup de RAM |
| `cd_010/plot_timestamp_distribution.py` | ~45 min (16 Go de RAM) |
| `GENERATION_generate_all_csv.py` (variante `with_distance`) | ~20 min |
| `MOVEMENT_PREDICTION_extract_features.py` | 20–40 min selon la version |
| `deep_learning/train.py` | ~27 min / epoch |

### Dépendances

Il n'y a **pas de `requirements.txt`** (à créer). D'après les imports :

```
pandas · numpy · matplotlib · seaborn · plotly · tqdm
geopy                    (distances géodésiques)
networkx · cdlib         (graphes et détection de communautés)
scikit-learn · xgboost · joblib · shap   (prédiction de mouvement)
torch                    (deep learning)
```

Environnement configuré sous **conda** (cf. [.vscode/settings.json](.vscode/settings.json)). Les `.pyc` présents indiquent **Python 3.14**.

---

## 7. État d'avancement et pistes de reprise

### Ce qui est solide et réutilisable

- **La compréhension du dataset** : distributions, anomalies, temps passé par cellule, cellules populaires. Toutes les figures sont dans `final_results/` et les chiffres dans `results/cd_142/json/`.
- **La classification des utilisateurs en 20+ classes de présence** (`4_user_classification.py`) — c'est le socle de tout ce qui suit ; la classe 1 (présence sur les 6 fenêtres) sert de population de référence.
- **La chaîne de génération des variantes du dataset** — bien factorisée, un simple booléen pour ajouter une variante.
- **Le modèle VOMM** avec backoff par *absolute discounting*, sa chaîne d'évaluation et son jeu de métriques.

### Ce qui a été tenté sans succès (ne pas refaire)

| Piste | Où c'est documenté |
|---|---|
| Corréler le délai du contexte au délai de la prochaine transition | En-tête de `5_correlation_time.py`, en gros caractères |
| Apprendre un poids par (heure, cellule) par descente de gradient | `train_time_weight.py`, commit `eeda5b6` |
| Graphes construits par station de base plutôt que par cellule | Commit `9ae354d` : *« turned out the graphs are less meaningful this way »* |
| Features V4 pour la prédiction de mouvement | Commit `57ad37e` : *« not so good »* |

### Ce qui reste ouvert

1. **Prédiction de mouvement** — c'est là que le travail s'est arrêté. Les V1/V2 de features existent, la V4 a régressé. L'analyse SHAP (`MOVEMENT_PREDICTION_analyze_features_usage.py`) est l'outil à mobiliser pour comprendre **pourquoi** : elle isole les features dominantes sur les faux positifs et faux négatifs.
2. **Transformer** — infrastructure complète mais `test.py` et `loss.py` sont **vides**. Il manque toute l'évaluation. C'est la tâche la plus rapidement rentable : le modèle et l'entraînement existent déjà.
3. **`VOMM_V5`** — le boost domicile/activité et le profil utilisateur sont **implémentés mais désactivés** dans le pipeline (`home_cell_boost`, `decay`, `alpha_profile` sont commentés dans `VOMM_prediction_pipeline.py` avec la mention *« Need more test »*). Un balayage de ces hyperparamètres est un travail immédiat et peu coûteux.
4. **Le catalogue d'expériences commentées** en fin de `VOMM_prediction_pipeline.py` (10 configurations : par classe, semaine/week-end, fusionné, dédupliqué) n'a jamais été exécuté de bout en bout de façon comparable. Les lancer toutes et les mettre dans un même tableau donnerait la première vue d'ensemble des performances.
5. **Dette technique** : uniformiser les imports (piège n°1), écrire un `requirements.txt`, et remplacer le pilotage par commentaires/décommentaires par de vrais arguments en ligne de commande ou un fichier de configuration.

### Chronologie du travail (repères git)

| Période | Travail |
|---|---|
| 24/03 → 09/04/2026 | Mise en place, analyse statistique globale, distances, dashboards |
| 09/04 → 04/05/2026 | Cellules domicile/activité, itérations successives sur les définitions, classification des utilisateurs |
| 04/05 → 22/05/2026 | Matrices de transition et n-grammes, splits train/test, encodage PyTorch, Transformer |
| 19/05 → 08/06/2026 | Modèles VOMM (V1 → V5), métriques, puis bascule vers la prédiction de mouvement |
| 30/06/2026 | Dernier commit (`Updated`) — mise au propre de `final_python/` et `final_results/` |

---

## 8. Par où commencer concrètement

1. **Regarder les figures** de `final_results/` (heatmap des timestamps, distribution des enregistrements, répartition des classes) — 10 minutes pour se faire une idée du dataset.
2. **Lire [results/README.md](results/README.md)** — il documente chaque fichier de sortie et le script qui l'a produit.
3. **Lire `4_user_classification.py`** — la fonction `classify()` définit le vocabulaire (« classe 1 », « classe 6 »…) employé partout ailleurs.
4. **Lire `models.py` de `simple_predictor/`**, en particulier `VOMM._recursive_prob()` — c'est le cœur mathématique du projet.
5. **Vérifier la présence des données** : `Database/cd_142_dataset/raw_dataset/` doit contenir 15 fichiers. Sinon, tout le reste est bloqué.
6. **Régénérer `results/predictions/`** en exécutant le pipeline 3 (§5), pour retrouver un état où les modèles tournent.
