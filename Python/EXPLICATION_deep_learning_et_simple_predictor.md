# Explication des dossiers `deep_learning/` et `simple_predictor/`

Ce document décrit le contenu des deux dossiers ajoutés, fichier par fichier, puis
propose des pistes pour la suite.

Les deux dossiers attaquent le **même objectif que le travail Markov** de
`Machin_learning/` (prédire la prochaine station), mais avec des modèles beaucoup
plus aboutis, et sur les variantes du dataset : `Database/no_duplicate`,
`Database/no_duplicate_merge_2g3g`, `Database/no_duplicate_max_512_records`,
`Database/class_merge`. Format de ligne identique au reste
du projet : `line[8::2]` = cellules, `line[9::2]` = timestamps, `line[7]` = nombre
de records, `line[5]`/`line[6]` = home/activity cell.

> Note : ces dossiers proviennent du dépôt d'un ancien collaborateur, organisé par zone
> géographique (`Database/cd_142_dataset/…`). Une seule zone ayant été conservée,
> l'arborescence a été aplatie ; d'anciens chemins peuvent subsister dans des
> commentaires. Voir [RESUME_PROJET.md](../RESUME_PROJET.md) §2 et [CLAUDE.md](../CLAUDE.md).

---

# 1. Dossier `deep_learning/` — Transformer (TUPE) pour la prédiction de cellule

Pipeline complet d'un **Transformer** qui prédit la prochaine cellule, séquence par
séquence, en auto-supervisé (décalage d'un pas / teacher forcing).

### `create_cellid_map.py`
Crée les tables de correspondance à partir de `Database/cells/cd_142_cells.csv` :
- `cell_map.json` : `cellid` (str) → entier `0..N-1` (pour le modèle de time-weight).
- `cell_map_start_1.json` : `cellid` → entier `1..N` (pour le Transformer, l'indice 0
  étant réservé au padding).
- `day_map.json` : les 15 jours → `0..14`.
Le commentaire d'en-tête justifie de **ne pas** encoder finement les jours : 2 semaines
sont trop courtes pour apprendre un motif hebdomadaire, et le 21/03/2014 a des données
défectueuses qui pourraient tromper le modèle.

### `encode_and_convert_csv_to_pytorch.py`
Transforme les CSV (`no_duplicate_max_512_records`) en tenseurs PyTorch prêts à
l'emploi. Pour chaque utilisateur (≥ 8 records) :
- cellules → entiers via `cell_map_start_1.json`, ajout d'un token **EOS** (= N+1),
  troncature/padding à **512** (99,4 % des utilisateurs ont < 512 records, et 512 = 2⁹).
- timestamps normalisés dans `[0,1]` (division par 86400).
- masque booléen des positions paddées.
Sauvegarde un split **train / val / test** (`encoded_train.pt`, `encoded_val.pt`,
`encoded_test.pt`) — mélange aléatoire puis découpe 60/20/20 (via `TEST_PART`/`VAL_PART`).

### `dataset.py`
Classe `MobilityDataset` (torch `Dataset`) : charge un `.pt` et renvoie les triplets
`(cells, times, mask)`. La formation des paires (entrée, cible) se fait dans la boucle
d'entraînement.

### `models.py`
Le modèle `MobilityTransformer` et ses briques :
- **`TUPEMultiHeadAttention`** : attention avec *Untied Positional Encoding*
  (Ke et al., 2020). Le score d'attention sépare strictement le flux **contenu**
  (embedding de cellule) et le flux **position** (projection du timestamp), au lieu de
  les additionner. Deux jeux de projections indépendants (`Wq/Wk` pour le contenu,
  `Uq/Uk` pour la position).
- **`TUPETransformerLayer`** : couche pré-norm (attention + feed-forward GELU).
- **`MobilityTransformer`** : embedding de cellule (`padding_idx=0`), projection
  temporelle par petit MLP, N couches TUPE, tête linéaire vers le vocabulaire
  (`vocab_size = N+2` : PAD, cellules, EOS). Sort des logits `(B, L, vocab)`.

### `train.py`
Boucle d'entraînement (`~27 min/epoch`) :
- teacher forcing décalé d'un pas : entrée = tokens `0..L-2`, cible = `1..L-1` ;
- loss `CrossEntropyLoss` calculée seulement sur les positions **non paddées** et
  **≥ MIN_CONTEXT=6** (on ne prédit pas les 6 premières, contexte trop court) ;
- `evaluate()` renvoie loss + **accuracy top-1** ;
- reprise possible depuis un checkpoint, `OneCycleLR`, AdamW, clip de gradient.
- Hyperparamètres par défaut : `N_CELLS=369`, `d_model=128`, 4 têtes, 2 couches,
  `N_EPOCHS=3`. Sauvegarde dans `results/predictions/deep_learning/checkpoints/`.

### `loss.py`, `test.py`
**Supprimés.** C'étaient deux fichiers vides : la loss est définie directement dans
`train.py` (`nn.CrossEntropyLoss`), et l'évaluation finale du Transformer reste à écrire.

---

# 2. Dossier `simple_predictor/` — VOMM, poids temporels, et prédiction de mouvement

Trois sous-projets cohabitent : (A) la prédiction de la prochaine cellule par modèles
de Markov d'ordre variable (VOMM), (B) l'apprentissage de poids temporels, (C) un
classifieur binaire « l'utilisateur bouge-t-il ? » (move/stay).

## 2.A — Prédiction de cellule : Markov naïf & VOMM

### `utils.py`
Boîte à outils partagée :
- `merge_cell_id`, distances géodésiques, `get_day`, `is_weekend`.
- **`HelperVOMM`** : prépare les n-grammes (`prepare_ngrams` → totaux par contexte +
  nombre de suffixes uniques) et les comptes d'unigrammes (`prepare_unigram_counts`).
- **`HelperData`** : mapping utilisateur → (jour, ligne), sauvegardes JSON/CSV.
- **`Metrics`** : `top_k_accuracy` (ACC@k), `map_k` (MAP@k), `log_likelihood`,
  `perplexity`, `mean_reciprocal_rank`. *(Ces deux premières étaient définies en double,
  la 2ᵉ écrasant la 1ʳᵉ ; les doublons ont été supprimés.)*
- `MobilityDataset` (torch) pour le modèle de time-weight.

### `models.py`
Le cœur des modèles de prédiction :
- **`NaiveMarkovChain`** : Markov d'ordre fixe. Pour un contexte donné, renvoie les
  suffixes les plus fréquents (probabilités pré-triées). Repli : prédire la dernière
  cellule (`last`).
- **`NaiveMarkovChainWithTimeWeight`** : idem, mais pondère par un poids horaire.
- **`VOMM`** : *Variable Order Markov Model* avec **back-off par *discounting* absolu** (le
  principe de base de Kneser-Ney, sans ses comptes de continuation).
  `_recursive_prob` combine la probabilité au plus long contexte disponible avec un
  repli récursif vers des contextes plus courts, **jusqu'à l'ordre 0** (l'unigramme).
  `predict_next` agrège les candidats de tous les ordres et les classe (avec cache).
  ⚠️ La récursion s'arrêtait auparavant à l'ordre 1, ce qui court-circuitait l'étage
  Markov d'ordre 1 (`train_data[1]` n'était jamais lu pour le scoring). Corrigé : les
  métriques VOMM antérieures à cette correction sont à relancer.
- **`VOMM_V4`** : variante (scoring en log-space + softmax) ; elle avait déjà la bonne
  borne de récursion, qui a servi de référence pour corriger `VOMM` et `VOMM_V5`.
- **`VOMM_V5`** : la version la plus riche. Ajoute :
  - `temporal_boost` : bonus multiplicatif si la cellule candidate est le **home**
    la nuit ou l'**activity** en journée (transitions douces aux bords de plage) ;
  - `build_user_profile` : profil de mobilité par decay positionnel, modulé par la
    **concentration** (entropie) et le **momentum** récent (nouveauté des dernières
    cellules).
- **`MovementPredictor` / `MovementPredictorV2`** : extracteurs de features pour le
  sous-projet (C) — voir plus bas. *(Une méthode heuristique `predict_movement`
  référençait des features `current_streak`, `is_home_now`… qui ne sont plus produites
  par `extract_features`, et l'appelait avec 4 arguments au lieu de 2 : elle levait un
  `TypeError` et a été supprimée. La décision move/stay est apprise par XGBoost / LSTM.)*

### `user_mapping.py`
Génère `user_mapping.json` : `user_id` → `[jour, numéro_de_ligne]`, pour retrouver
rapidement un utilisateur dans les CSV.

### `NAIVE_prediction_pipeline.py`
Évalue le **Markov naïf** (ordres 1→5) et le **VOMM** sur les utilisateurs de test,
calcule ACC@{1,3,5,10}, MAP@k, log-likelihood, et des méta-stats (part des
prédictions où « prédire = rester sur place » est juste/faux, nombre d'utilisateurs
non calculables). Plusieurs configurations de split (aléatoire, par âge, avec/sans
répétitions) sont présentes mais **la plupart commentées** ; seule tourne la variante
« split aléatoire, test sans répétitions, limité à 10 000 utilisateurs ».

### `VOMM_prediction_pipeline.py`
Évalue **VOMM_V5** (avec les boosts) sur le test. `VOMM_boosted` prédit pour chaque
position, applique les métriques, et gère home/activity cell. La fonction `predict`
balaie une liste de `discount`. De nombreux blocs (random, merge 2g3g, par classe,
semaine/week-end) sont prêts mais commentés.

### `VOMM_single_prediction.py`
Démo : prédire la suite d'une séquence codée en dur, et afficher les 10 meilleurs
candidats avec leur probabilité (en signalant celui qui est identique à la dernière
cellule du contexte). *(Le script utilisait auparavant une API obsolète —
`VOMM(counts=…, context_totals=…)`, `prepare_ngrams(computed_ngrams_filepath=…)` — et
ne tournait plus ; il a été réécrit contre l'API actuelle, avec contrôle des prérequis.)*

### `PLOT_VOMM_metrics_by_discount.py`
Trace, à partir d'un JSON d'étude, l'évolution de ACC@{1,3,5,10} et du log-loss moyen
en fonction du **discount** du VOMM. Sauvegarde `vomm_combined_metrics_plot.png`.

## 2.B — Apprentissage de poids temporels

### `train_time_weight.py`
Apprend, par **descente de gradient** (PyTorch, GPU requis), une matrice de poids
`θ[heure, cellule]` (24 × 369) qui module la matrice de transition Markov selon
l'heure. Entraîné pour les ordres 2→5, loss = log-vraisemblance négative de la
vraie cellule suivante, avec courbe train/val sauvegardée. C'est une manière
« apprise » (plutôt qu'heuristique comme le `temporal_boost` du VOMM_V5) d'injecter
l'information horaire.

## 2.C — Prédiction binaire de mouvement (move / stay)

Sous-problème distinct : à chaque instant, l'utilisateur **va-t-il changer de
cellule** au prochain record (1) ou rester (0) ? Utile car 72 % des records
consécutifs sont dans la même cellule (`no_duplicate`, 15 jours).

### `MOVEMENT_PREDICTION_extract_features.py` et `…_extract_features_V2.py`
Parcourent le CSV et produisent, pour chaque point de prédiction, un vecteur de
features + le label move/stay, via `MovementPredictor.extract_features` (position vs
cellule d'ancrage, phase du trajet, départs/retours, accélération locale, entropie
hors-ancre, etc.). Différence entre les deux : la version « … » génère **tous** les
points de chaque utilisateur, la V2 tire **un seul** point aléatoire par utilisateur.
Sortie : `..._features_V6.csv`.

### `MOVEMENT_PREDICTION_fit_features.py` et `…_fit_features_V2.py`
Entraînent un **XGBoost** sur ces features (gère le déséquilibre via
`scale_pos_weight`, early stopping), évaluent sur le test (accuracy, F1, ROC-AUC,
average precision, matrice de confusion) et produisent les graphes : ROC/PR,
importance des features, matrice de confusion, calibration, métriques vs seuil de
décision. La V2 utilise le jeu de features de `MovementPredictorV2` (signaux de
rupture locaux), la V1 celui de `MovementPredictor`. Une référence dans le code
indique un XGBoost autour de **F1 ≈ 0.60, ROC-AUC ≈ 0.84**.

### `MOVEMENT_PREDICTION_train_lstm.py`
Alternative deep learning au XGBoost : reconstruit des **séquences** de vecteurs de
features (l'idée étant que le signal est dans la *dérivée* des features, pas leur
valeur absolue) et entraîne un **LSTM bidirectionnel** + un **MLP résiduel** de
comparaison. Compare LSTM / MLP / XGBoost sur precision, recall, F1, ROC-AUC.

### `MOVEMENT_PREDICTION_analyze_features_usage.py`
Analyse **SHAP** du classifieur XGBoost : importance globale, beeswarm (direction +
magnitude), dependence plots, analyse par issue (TP/TN/FP/FN — quelles features
trompent le modèle), explications locales (waterfall), et insights texte.

### `MOVEMENT_PREDICTION_analyze_user_movements.py`
Analyse **exploratoire** (pas un modèle) : distribution des heures du **premier** et
du **dernier** mouvement de la journée (le dernier n'étant calculé que pour ceux qui
finissent là où ils ont commencé). Histogrammes globaux + un par jour (week-ends en
rouge). Sert à concevoir des features pertinentes.

### `MOVEMENT_PREDICTION_mobility_explorer.py`
Outil interactif : demande un `user_id` en console et trace (Plotly) la **distance à
la première cellule de la journée** au fil du temps. Pour inspecter visuellement des
trajectoires individuelles.

---

# 3. Vue d'ensemble et articulation avec le travail Markov précédent

- Le **VOMM** de `simple_predictor` est une **généralisation** du Markov séquentiel de
  `Machin_learning/` : ordre variable + back-off par *discounting* absolu + boosts temporels. Le
  travail Markov d'ordre 1 en est, de fait, un cas particulier.
- Le **Transformer** de `deep_learning` est l'approche la plus lourde (contexte long,
  attention, position apprise).
- La **prédiction de mouvement** (move/stay) est un problème binaire séparé, qui peut
  servir de **première étape** avant de prédire *où*.
- ⚠️ Attention aux **populations comparées** : tous ces dossiers lisent bien
  `Database/no_duplicate` (et ses variantes), mais pas au même endroit du pipeline —
  `Machin_learning/` évalue par utilisateur sur le dataset complet **avec** les
  répétitions consécutives, alors que les pipelines VOMM évaluent sur un split de test
  **dédupliqué**. Il faut vérifier qu'on compare les modèles sur **le même split** avant
  de conclure ; `COMPARE_markov_vs_vomm.py` est le script qui le garantit.

## Points d'attention / dette technique

**Corrigé depuis :** doublons de `top_k_accuracy` / `map_k` et de
`convert_lat_lon_distance_to_meter` dans `utils.py` ; `predict_movement` supprimée ;
`VOMM_single_prediction.py` réécrit ; `N_CELLS` lu depuis `cell_map_start_1.json` au lieu
d'être codé en dur ; `loss.py` et `test.py` (vides) supprimés ; borne de récursion du
back-off VOMM alignée sur l'ordre 0.

**Restant :**
- Pas de procédure d'évaluation finale du Transformer (métriques de test comparables au VOMM).
- Beaucoup de configurations de run sont commentées dans les pipelines : l'historique
  des expériences n'est pas rejouable en un clic.
- Trois marges de « gap » différentes (4h+30s / 4h+60s / 4h+10min) selon les dossiers.
- `utils.py` existe en trois copies (racine, `important_cells_work/`, `simple_predictor/`).

---

# 4. Pistes pour la suite

*(Ces pistes sont reprises et discutées dans la conversation.)*

1. **Tableau de comparaison unifié.** Le livrable qui manque : évaluer **tous** les
   modèles (Markov d'ordre fixe, VOMM, VOMM+boosts, Transformer, LSTM move/stay) sur
   **le même split de test** et les mêmes métriques (ACC@1/3/5, MAP@k, log-loss).
   Aujourd'hui chacun a ses chemins et ses réglages ; on ne peut pas les départager.

2. **Aller au bout du Transformer.** Il n'est configuré que pour 3 epochs. Le lancer
   pleinement, récupérer son ACC@1 et le comparer au VOMM, dira si le coût du deep
   learning est justifié sur seulement 2 semaines de données.

3. **Prédicteur en deux étapes.** Étant donné les 72 % de self-transitions, un étage
   1 « bouge / bouge pas » (le classifieur move/stay déjà écrit) suivi d'un étage 2
   « où ? » (VOMM, seulement quand un mouvement est prédit) pourrait nettement
   améliorer l'ACC@1 par rapport à un VOMM seul.

4. **Évaluer les poids temporels.** `train_time_weight.py` produit des poids mais rien
   ne mesure encore leur apport face au VOMM nu. À brancher dans le pipeline
   d'évaluation.

5. **Nettoyage.** ✅ Fait : fichiers vides supprimés, `VOMM_single_prediction` réécrit,
   `predict_movement` supprimée, doublons de métriques éliminés, `N_CELLS` dérivé du
   mapping. Reste à relancer les métriques VOMM après la correction du back-off.

6. **Un seul dataset de référence.** Fixer une bonne fois le split train/test (et la
   variante `cd_142_dataset` vs `no_duplicate`) partagé par tous les modèles, sinon
   les comparaisons resteront incomparables.
