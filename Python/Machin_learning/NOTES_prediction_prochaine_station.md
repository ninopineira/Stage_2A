# Notes — Prédiction de la prochaine station (baseline Markov)

Résumé de la session de travail : mise en place d'une première approche de
prédiction de la prochaine cellule (station de base) visitée par un utilisateur.

## Contexte du projet

- Données : CDR anonymisés (Prague / CVUT), 15 jours (2014-03-12 → 2014-03-26),
  un fichier CSV par jour dans `Database/no_duplicate`.
- Chaque ligne = un utilisateur sur **une seule journée** : `line[8::2]` = cellules
  visitées dans l'ordre, `line[9::2]` = horodatages (secondes depuis minuit).
- Travail déjà réalisé et réutilisé ici :
  - Matrices de transition (`transition_matrix.py`), entropies par utilisateur
    (`transition_emtropy.py`), sauvées dans `results/numpy/user_entropies_no_merge.npy`.
  - Borne théorique de prévisibilité **P^max** (inégalité de Fano, benchmark
    Song et al. = 0.93) dans `maximal_previsibility.py`.

## Point clef découvert

**Les `user_id` ne sont PAS stables d'un jour à l'autre** (vérifié : intersection
vide entre les ids du 12/03 et du 13/03 — l'anonymisation est refaite chaque jour).
→ Impossible de construire une séquence multi-jours par utilisateur. Chaque
"utilisateur" n'existe que sur une journée (typiquement 10–200 records).
→ Toute la modélisation personnelle est donc confinée à une seule journée.

## Ce qui a été construit

### 1. `maximal_previsibility.py` (refactor)
Le code de script (chargement, calcul, plot) a été déplacé sous
`if __name__ == "__main__"` et une fonction réutilisable `load_pmax_dataframe()`
a été extraite, pour pouvoir importer `compute_pmax` / `P^max` sans effet de bord.

### 2. `markov_baseline.py` — baseline Markov à split aléatoire
- Pour chaque utilisateur : transitions `(cell_from, cell_to)` avec état `outside`
  de part et d'autre d'un gap > 4h10 (seuil `4*3600+60`, cohérent avec le reste du projet).
- Split **aléatoire** 70% train / 30% test des transitions (seed fixe).
- Modèle = fréquences empiriques `P(cell_to | cell_from)` sur le train ; prédiction
  par **argmax** (pas un tirage) ; repli sur la cellule la plus fréquente si
  `cell_from` inconnu (cold start).
- Sortie : `results/numpy/markov_accuracy_no_merge.npy` (par user : jour, accuracy,
  n_train, n_test, n_records).
- **Limite identifiée** : ce split aléatoire répond à la question "est-ce que je
  capture mon P^max sur un échantillon quelconque de mes trajets ?", **PAS** au
  vrai objectif du projet (utiliser uniquement le passé).

### 3. `plot_markov_vs_pmax.py` — illustrations de la baseline
Graphes : scatter accuracy vs P^max (+ droite y=x), histogrammes superposés,
accuracy/P^max par tranche de nombre de records, accuracy/P^max par jour
(week-ends en rouge), histogramme de l'écart P^max − accuracy.

### 4. `plot_markov_sequential_by_day.py` — vue agrégée jour par jour
Pour chaque jour : accuracy moyenne du modèle séquentiel à côté du P^max moyen
sur les mêmes utilisateurs (barres groupées, valeurs affichées, week-ends en
rouge), plus un tableau imprimé (nb d'utilisateurs, moyenne, médiane, écart moyen).

**Abandonné** : `markov_examples.py` (figures par utilisateur — trajectoire de la
journée, accuracy cumulée, matrice personnelle). Illisible en pratique : les
utilisateurs visitent trop de cellules distinctes pour qu'une journée tienne dans
un graphe interprétable. Remplacé par la vue agrégée ci-dessus.

### 5. `markov_sequential_prediction.py` — **la vraie voie** (causal / online)
C'est l'approche qui correspond au vrai objectif : **prédire la prochaine station
en ne connaissant que le passé**.
- On parcourt les transitions de l'utilisateur dans l'**ordre chronologique**.
- À la transition `i`, on prédit avec la matrice construite **uniquement** sur les
  transitions `0..i-1` (passé strict), puis on met à jour la matrice avec la
  transition `i` avant de continuer → le modèle apprend **au fil de l'eau (online)**.
- La toute première transition (aucun historique) est exclue du calcul d'accuracy.
- Sortie : `results/numpy/markov_sequential_accuracy_no_merge.npy`.
- `INPUT_DIR` pointe sur `Database/no_duplicate` (dataset complet). Le passage par
  `Database/sample_for_training` n'était qu'un premier test rapide, et il donnait
  des chiffres trompeurs (voir plus bas).

## Correction importante : le N de l'inégalité de Fano

**Bug trouvé et corrigé.** `P^max` était calculé avec `N` = **nombre de records**
de l'utilisateur, alors que l'inégalité de Fano attend `N` = **nombre d'états
distincts** (cellules distinctes visitées, + `outside` si l'utilisateur a quitté
la zone). Un `N` surestimé gonfle mécaniquement `P^max`.

Le fichier `plot_hist_entropy.py` était même incohérent avec lui-même : sa
fonction `compute_entropy_metrics` documente et utilise correctement
`N = nombre de stations distinctes`, mais son bloc `__main__` reprenait le
nombre de records du `.npy`.

Mesure de l'impact (2731 utilisateurs du 2014-03-12) :

| | ancien (N = records) | corrigé (N = états distincts) |
|---|---|---|
| N moyen | 145.6 | 10.6 |
| P^max moyen | 0.767 | 0.580 |
| % users > 0.8 | 42.6 % | 8.1 % |

Fichiers modifiés :
- `transition_emtropy.py` : `entropy_for_user` renvoie désormais aussi le nombre
  d'états distincts, stocké comme 5e champ du `.npy`
  (`(day, entropy, rel_entropy, n_records, n_states)`).
- `maximal_previsibility.py` : utilise `n_states` pour Fano ; lève une erreur
  explicite si le `.npy` est à l'ancien format.
- `plot_hist_entropy.py`, `plot_entropies_by_number_of_records.py`,
  `plot_markov_vs_pmax.py` : adaptés au nouveau format.

✅ `transition_emtropy.py` a été relancé : `user_entropies_no_merge.npy` est
régénéré au nouveau format (1 377 475 utilisateurs, 43 s). Les résultats P^max
produits/présentés **avant** cette correction sont à revoir.

## Deuxième bug : l'intervalle de recherche de brentq

Révélé par la correction ci-dessus. `compute_pmax` cherchait la racine de Fano
sur `[1e-10, 1-1e-10]`, mais la fonction `H(p) + (1-p)·log2(N-1) - S` y a le
**même signe aux deux bornes** : elle vaut `log2(N-1) - S` près de 0 et `-S` près
de 1. Dès que `S > log2(N-1)` — donc **toujours pour N = 2**, où `log2(1) = 0` —
`brentq` lève `ValueError` et la fonction renvoyait silencieusement `NaN`.

Avec l'ancien N (gonflé), le cas était rare ; avec le N correct (petit), il
touchait **37,8 % des utilisateurs**, concentrés sur les petits N (2, 3, 4…).

Correction : borner sur `[1/N, 1]`. La fonction de Fano culmine en `p = 1/N` (où
elle vaut `log2(N)`) puis décroît jusqu'à 0 en `p = 1` ; comme `0 ≤ S ≤ log2(N)`,
le changement de signe y est garanti, et c'est la racine qui a un sens
(prévisibilité au moins égale au hasard). Après correction : **0 NaN**.

Corrigé dans `maximal_previsibility.py` et dans la copie de `compute_pmax` de
`plot_hist_entropy.py`.

## Comparaison des deux méthodes (sans P^max)

`plot_markov_methods_comparison.py` met les deux protocoles en parallèle, sur les
mêmes 740 029 utilisateurs (présents dans les deux runs, entropie > 0) :

| mesure | séquentiel (passé seul) | split aléatoire 70/30 |
|---|---|---|
| accuracy moyenne | **0.555** | **0.592** |
| accuracy médiane | 0.571 | 0.634 |

- écart moyen : **+0.036** en faveur du split aléatoire ;
- corrélation entre les deux méthodes : **0.864** ;
- le séquentiel l'emporte quand même chez **35,5 %** des utilisateurs.

L'écart est faible et remarquablement stable (+0.030 à +0.040 selon le jour) :
voir seulement le passé coûte environ 3,6 points d'accuracy. C'est le prix
honnête de la contrainte causale, et il est modeste — le split aléatoire, qui
pioche des transitions de plus tard dans la journée, ne gagne pas tant que ça.

Trois figures, sauvegardées dans `results/plots/` :
- `markov_methods_by_day.png` — barres groupées jour par jour (week-ends en rouge) ;
- `markov_methods_difference.png` — histogramme de l'écart par utilisateur ;
- `markov_methods_calendar.png` — calendrier de hexbins séquentiel vs aléatoire.

## Résultats sur le dataset complet (`Database/no_duplicate`)

Configuration retenue : seuil de gap **4h10** (`4*3600+600`) partout, utilisateurs
à **entropie nulle exclus** (ils n'ont jamais quitté une seule cellule, Fano leur
donne P^max = 1 par convention et ils gonflent les moyennes).

803 530 utilisateurs prédits → **740 029 conservés** (63 501 à entropie nulle
écartés).

| mesure | valeur |
|---|---|
| accuracy séquentielle (causale) | **0.555** (médiane 0.571) |
| accuracy split aléatoire | 0.592 |
| P^max moyen | **0.575** |
| corrélation accuracy / P^max | **0.844** |
| utilisateurs avec accuracy > P^max | 38,8 % |

L'accuracy causale (0.555) passe **sous** le P^max (0.575) sur cette population
représentative, et l'écart moyen est positif **tous les jours** — le graphe jour
par jour est cohérent, contrairement à ce qu'on obtenait sur l'échantillon à
forte entropie. La corrélation de 0.844 entre accuracy et P^max est un bon
signe : le modèle réussit bien là où la théorie prédit qu'il peut réussir.

**Effet week-end net** : accuracy 0.593 le week-end contre 0.544 en semaine
(P^max 0.609 vs 0.564). Les gens sont plus prévisibles le week-end. À rapprocher
du travail de classification qui montrait déjà des profils week-end distincts.

Il reste 38,8 % d'utilisateurs au-dessus de leur P^max, ce qui est la trace de la
limite S_unc documentée plus bas (Markov exploite l'ordre, que S_unc ignore).

## Résultats antérieurs sur l'échantillon à forte entropie

Sur `Database/sample_for_training` (7 500 utilisateurs, 15 jours × 500) :

| mesure | valeur |
|---|---|
| accuracy séquentielle (causale) | **0.258** |
| accuracy split aléatoire | 0.295 |
| P^max moyen | 0.243 |

Comme prévu, l'évaluation causale est plus basse que le split aléatoire (0.258 vs
0.295) : le modèle part de zéro et n'a que le passé.

Vue jour par jour : `plot_markov_sequential_by_day.py` — l'accuracy est très
stable d'un jour à l'autre (0.234 → 0.277), sans écart week-end/semaine marqué.

### ⚠️ Deux réserves importantes sur ces chiffres

**1. La population évaluée est la plus difficile possible.**
`sample_for_training.py` sélectionne les utilisateurs à **plus forte entropie**.
Les 7 500 évalués sont donc au **99,6e percentile** d'entropie du dataset :

| | échantillon évalué | dataset complet |
|---|---|---|
| n_states moyen | 28.6 | 5.8 |
| S_unc moyen | 4.370 | 1.480 |
| P^max moyen | 0.243 | 0.656 |

L'accuracy de 0.258 n'est donc **pas représentative** : c'est le pire cas. Pour un
chiffre représentatif il faut faire tourner `markov_sequential_prediction.py` sur
`Database/no_duplicate` (remettre `INPUT_DIR` dessus).

**2. Comparer cette accuracy à ce P^max n'est pas légitime.**
48,8 % des utilisateurs ont une accuracy **supérieure** à leur P^max. Ce n'est pas
une erreur numérique : `P^max` est ici dérivé de **S_unc**, l'entropie
*non corrélée* (distribution des fréquences de visite seulement). Fano appliqué à
S_unc donne Π^unc = le plafond d'un prédicteur **sans mémoire**, qui ne connaît
que « à quelle fréquence je vais où ». Or le modèle de Markov exploite l'**ordre**
(`P(suivante | actuelle)`), exactement l'information que S_unc jette. Le dépasser
est donc attendu — c'est même le point central de Song et al. : Π^unc < Π^max
parce que les corrélations temporelles portent de l'information.

Un vrai plafond comparable à un prédicteur markovien demanderait l'entropie
**réelle** S_real (taux d'entropie tenant compte de l'ordre), estimée chez Song
et al. par un estimateur **Lempel-Ziv**.

**Décision prise** : on ne fait pas Lempel-Ziv (trop lourd pour le temps
disponible). On conserve la comparaison accuracy vs P^max issu de S_unc, comme
c'était fait avant, en gardant en tête cette limite : il faut la présenter comme
une comparaison au plafond d'un prédicteur *sans mémoire*, et non comme une borne
absolue que le modèle ne pourrait pas dépasser.

## Conclusions

1. **Deux évaluations distinctes** ont été clarifiées :
   - *Split aléatoire* (`markov_baseline.py`) = validation par rapport à la borne
     théorique P^max, informatif mais pas causal.
   - *Séquentiel / online* (`markov_sequential_prediction.py`) = le vrai objectif ;
     n'utilise que le passé, accuracy attendue plus basse (moins d'info en début
     de journée), mais c'est la mesure honnête pour ce projet.
2. Ces scripts sont une **baseline de départ** (Markov d'ordre 1, statistiques de
   bigrammes cellule→cellule, sans heure ni jour de semaine ni info inter-users).
   À améliorer ensuite.

## Mesure de contexte (constatée, non exploitée pour l'instant)

Sur l'échantillon `sample_for_training` (forte entropie), **43,6 % des enregistrements
consécutifs sont dans la même cellule** (le nettoyage « no_duplicate » ne supprime pas
les répétitions consécutives). ⚠️ Ce chiffre n'est **pas** représentatif : sur
l'ensemble de `no_duplicate` (15 jours), la vraie valeur est **72,0 %**. Conséquence
mesurée sur le 2014-03-12 de l'échantillon (42 133 transitions, accuracy poolée) :

| stratégie | accuracy |
|---|---|
| implémentation actuelle (fallback = cellule la plus fréquente) | 0.313 |
| « l'utilisateur ne bouge pas » (baseline triviale) | 0.434 |
| Markov + fallback « ne bouge pas » | 0.425 |
| Markov + fallback + départage des ex æquo vers « rester » | 0.437 |

Deux choses à retenir : **32,9 % des prédictions sont en cold-start** (cellule de
départ jamais vue), et sur cette population (la plus difficile du dataset) le
Markov d'ordre 1 n'apporte quasiment rien face à la baseline triviale. À garder
sous le coude si les résultats sur `no_duplicate` complet paraissent faibles.

## Pistes d'amélioration (pour la suite)

- Relancer l'évaluation sur `Database/no_duplicate` (population représentative)
  et pas seulement sur l'échantillon à forte entropie, qui est le pire cas.
- Corriger le fallback cold-start (prédire la cellule actuelle plutôt que le mode
  global) : +0.11 d'accuracy sur le jour testé.
- Markov d'ordre supérieur (contexte des 2–3 dernières cellules).
- Intégrer le temps (heure de la journée) comme feature.
- Lissage / back-off (ex. combiner modèle personnel + modèle population de
  `transition_matrix.py`) pour mieux gérer le cold start.
- Comparer systématiquement l'accuracy séquentielle à P^max par utilisateur et
  quantifier l'écart restant.
- À terme : modèle séquentiel type LSTM sur `Database/sample_for_training`.

## État d'exécution

Les scripts ont été écrits et vérifiés syntaxiquement, mais **pas encore exécutés
en entier** sur les données au moment de ces notes (exécution interrompue). Étapes
restantes : lancer `markov_sequential_prediction.py`, vérifier les chiffres
(accuracy moyenne/médiane, comparaison vs P^max et vs split aléatoire), puis
itérer sur les améliorations.
