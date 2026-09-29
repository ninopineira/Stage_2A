# Prévisibilité et entropie — synthèse pour le rapport

Document de fond : il rassemble les **thèses**, les **formules** et le **raisonnement**
qui sous-tendent le code de prédiction et de prévisibilité (dossiers
`Machin_learning/` et `simple_predictor/`). Objectif : disposer du contenu
scientifique (pas seulement du code) pour la rédaction du rapport et la soutenance.

---

## 1. Cadre et objet

On prédit la **prochaine cellule** (antenne) visitée par un utilisateur à partir de
son historique de la journée (CDR anonymisés, région de Karlovy Vary, 15 jours). Deux questions
distinctes structurent tout le travail :

- **Prédiction** : quel modèle prédit le mieux la prochaine cellule ?
- **Prévisibilité** : quelle est la borne *théorique* de ce qui est prédictible pour
  chaque utilisateur, et à quel point les modèles s'en approchent ?

### Faits structurants sur les données (à énoncer dans le rapport)

- **Les `user_id` ne sont pas stables d'un jour à l'autre** (anonymisation refaite
  chaque jour). Conséquence : toute modélisation par utilisateur est **confinée à une
  seule journée** (~10–200 records typiquement). Aucune séquence multi-jours.
- **72,0 % des enregistrements consécutifs sont dans la même cellule**
  (self-transitions, `no_duplicate`, 15 jours ; l'ancien chiffre de 43,6 % venait de
  l'échantillon `sample_for_training` et n'est pas représentatif). La prédiction triviale
  « il reste où il est » obtient **0,652** d'ACC@1 : elle bat le Markov d'ordre 1 (0,647). Tout l'enjeu est de **détecter les vrais déplacements**.
- **Deux représentations** : *avec* self-transitions (`no_duplicate`) ou *dédupliquée*
  (self-transitions retirées). Le choix change radicalement les chiffres et doit être
  **cohérent** entre le train, le test et l'entropie (voir §6, piège n°1).

---

## 2. Les modèles de prédiction

### 2.1 Markov d'ordre 1
`P(cellule suivante | cellule actuelle)`, fréquences empiriques. C'est la baseline.

### 2.2 VOMM (Variable-Order Markov Model)
Markov d'**ordre variable** (jusqu'à 5) avec **back-off** par *discounting* absolu
(le principe de base de Kneser-Ney, sans ses comptes de continuation). Pour un contexte `h` et une cible `w` :

$$P(w\mid h) = \frac{\max(\text{count}(h,w) - d,\ 0)}{\text{count}(h)} \;+\; \lambda(h)\cdot P(w\mid h')$$

- `d` = *discount* (≈ 0.9), retire une masse de probabilité à chaque transition vue ;
- `λ(h) = d·(\text{nb de suffixes distincts après } h)/\text{count}(h)` = poids de repli ;
- `h'` = contexte raccourci d'une cellule ; la récursion descend jusqu'à l'unigramme.

**Intuition** : si « A→B » a été vu 10 fois, un Markov naïf dit `P(B|A)=1` (trop
confiant). Le discount réserve un peu de masse aux cellules jamais vues après A, via
le terme de repli. C'est le principe du *discounting* absolu, emprunté au langage (et à la
base de Kneser-Ney, dont nous n'utilisons pas les comptes de continuation).

### 2.3 ⚠️ Le décalage d'ordre du VOMM (corrigé en août 2026)
**Avant la correction**, le cas de base de la récursion était `if order == 1 → unigramme`.
Donc `VOMM(max_order=1)` était un **modèle unigramme** (0 cellule de contexte), pas un
Markov d'ordre 1, avec une ACC@1 ≈ **0.015** (il prédisait la cellule globalement la plus
fréquente). **Aujourd'hui** la récursion descend jusqu'à l'ordre 0 : `VOMM(max_order=k)`
conditionne bien sur k cellules (avec lissage). Pour comparer « par ordre », on utilise
quand même un **Markov d'ordre k pur** (`NaiveMarkovChain(order=k)`), sans lissage.

⚠️ **Piège de `NaiveMarkovChain`** : quand le contexte n'a jamais été vu, `predict_next`
renvoie la **dernière cellule** (« reste où il est », juste dans 72 % des transitions). Aux
ordres élevés, la plupart des contextes sont inédits : l'ACC@1 monte de 0,649 (ordre 1) à
0,741 (ordre 5) **à cause de ce repli**, pas grâce au contexte.

---

## 3. Thèses centrales sur la prédiction (résultats)

> **Thèse 1 — L'ordre 1 capte l'essentiel.** Le VOMM (ordre 5 + back-off) ne dépasse le
> Markov d'ordre 1 que de **2,6 points** d'ACC@1 (0,674 contre 0,647), sur le même test.
> Les ordres supérieurs n'apportent presque rien sur une journée. Et la règle triviale
> « reste où il est » fait déjà **0,652** sur les mêmes points : le Markov d'ordre 1 ne la
> bat pas, le VOMM ne la dépasse que de 2,2 points. *(À énoncer avec la figure de
> comparaison Markov vs VOMM.)*

> **Thèse 2 — La contrainte causale coûte peu.** En évaluation *séquentielle/online*
> (n'utiliser que le passé), l'accuracy moyenne du Markov d'ordre 1 est **0.555**
> (médiane 0.571) contre **0.592** en split aléatoire — soit ~3,6 points de coût pour
> l'honnêteté causale.

> **Thèse 3 — Effet week-end net.** Accuracy **0.593 le week-end** contre **0.544 en
> semaine** (les gens sont plus routiniers/prévisibles le week-end).

> **Thèse 4 — Corrélation accuracy / prévisibilité = 0.844** (Markov séquentiel vs
> P^max) : le modèle réussit là où la théorie prédit qu'il *peut* réussir.

---

## 4. La prévisibilité théorique : entropie + inégalité de Fano

### 4.1 Les trois entropies (par utilisateur)

- **Entropie non corrélée `S_unc`** — n'utilise que la distribution des fréquences de
  visite (ignore l'ordre) :
  $$S_{unc} = -\sum_c p(c)\,\log_2 p(c),\qquad p(c) = \frac{\text{nb de visites de } c}{\text{total}}$$

- **Entropie conditionnelle d'ordre k `S_cond^{(k)}`** — conditionne sur les k cellules
  précédentes :
  $$S_{cond}^{(k)} = -\sum_{a_1,\dots,a_k} p(a_1,\dots,a_k)\sum_b p(b\mid a_1,\dots,a_k)\,\log_2 p(b\mid a_1,\dots,a_k)$$

- **Entropie réelle `S_real`** — le vrai taux d'entropie (toutes corrélations),
  estimé par Lempel-Ziv (§5.3).

Emboîtement : $S_{real} \le \dots \le S_{cond}^{(2)} \le S_{cond}^{(1)} \le S_{unc}$.
Plus on tient compte de l'ordre, plus l'entropie baisse.

**Exemple** : un utilisateur `A→B→A→B→A→B`. `S_unc = 1 bit` (50/50, « imprévisible »
vu comme distribution) mais `S_cond^{(1)} = 0` (après A toujours B, après B toujours A →
parfaitement prévisible dès qu'on connaît l'ordre). C'est exactement l'information que
`S_unc` jette.

### 4.2 L'inégalité de Fano → P^max

Étant donné une entropie `S` et un nombre d'états `N`, la prévisibilité maximale `P^max`
est la solution de :
$$S = H(P^{max}) + (1-P^{max})\log_2(N-1),\qquad H(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

- `P^max` = probabilité maximale de deviner correctement la **prochaine** cellule (top-1)
  qu'*aucun* prédicteur ne peut dépasser, compte tenu de l'entropie du processus.
- On la résout numériquement (bissection sur `[1/N, 1]`, voir piège n°2 ci-dessous).
- Song et al. (2010) trouvent `P^max ≈ 0.93` sur leurs données.

### 4.3 ⚠️ Deux bugs corrigés (à présenter comme des points méthodologiques)

> **Bug A — le N de Fano.** `P^max` était calculé avec `N` = **nombre de records** de
> l'utilisateur, alors que Fano attend `N` = **nombre d'états distincts** (cellules
> distinctes + `outside`). Un `N` surestimé **gonfle** P^max. Impact mesuré (2731
> utilisateurs) : P^max moyen **0.767 → 0.580**, part des utilisateurs > 0.8 :
> **42,6 % → 8,1 %**.

> **Bug B — l'intervalle de `brentq`.** La racine de Fano était cherchée sur
> `[1e-10, 1-1e-10]`, où la fonction a le **même signe aux deux bornes** dès que
> `S > log_2(N-1)` — donc **toujours pour N=2** → `NaN`. Avec le N corrigé (petit), ça
> touchait **37,8 %** des utilisateurs. Correctif : borner sur `[1/N, 1]` (la fonction y
> décroît de `log_2(N)-S ≥ 0` à `-S ≤ 0`, changement de signe garanti). Résultat : **0 NaN**.

---

## 5. Trois estimateurs d'entropie et leurs limites

C'est le cœur méthodologique. On veut estimer l'entropie *conditionnelle* (donc P^max)
par utilisateur, mais sur **une seule journée courte** — ce qui est difficile.

### 5.1 Plug-in (maximum de vraisemblance, in-sample) — **s'effondre**

On estime la distribution conditionnelle *et* on mesure l'entropie **sur les mêmes
données**. Problème : à ordre k, la plupart des contextes de longueur k ne sont vus
**qu'une seule fois** → `p(b|contexte)=1` → entropie de la ligne = 0 → `S_cond^{(k)}→0`
→ `P^max→1` pour presque tout le monde.

> **Thèse 5 — L'effondrement est un artefact de sur-apprentissage.** À ordre élevé,
> l'entropie plug-in « explique » parfaitement le passé parce qu'elle regarde les mêmes
> données — ce qui ne dit **rien** sur le futur. Techniquement : l'estimateur plug-in est
> **biaisé négativement** en régime de sous-échantillonnage, et aucune vitesse de
> convergence ne peut lui être garantie indépendamment de la distribution (Antos &
> Kontoyiannis 2001).

C'est aussi pourquoi une **accuracy peut dépasser un P^max issu de `S_unc`** : `S_unc`
donne le plafond d'un prédicteur *sans mémoire*, or les modèles markoviens exploitent
l'ordre (que `S_unc` ignore). Le dépasser est **attendu** (c'est le point de Song et al.).

### 5.2 Held-out (validation croisée) — **ne s'effondre pas** (méthode retenue)

On estime `p̂(·|contexte)` sur une partie (train, lissée), on mesure la **surprise** sur
la partie retenue :
$$\hat S_{ho} = -\frac{1}{M}\sum_{i\in\text{held-out}} \log_2 \hat p(x_i\mid c_i)$$

- **Lissage obligatoire** (Laplace add-α + repli sur l'unigramme) sinon `log_2(0)=-\infty`.
- **Biais positif** : par l'identité $\hat S_{ho}\approx H(p,\hat p)=H(p)+\mathrm{KL}(p\|\hat p)\ge H(p)$,
  la cross-entropie **surestime** l'entropie → `P^max` **conservateur** (Brown et al. 1992).
- **Ne s'effondre pas** : quand k croît, les contextes deviennent inédits → `p̂` petit →
  surprise **élevée** → l'entropie *remonte*. C'est le comportement correct.
- Bonus : mesurée sur les **mêmes** données que l'accuracy → plafond et accuracy sur le
  même pied. Même principe que le VOMM (lisser puis se replier, Chen & Goodman 1999),
  mais **pas le même lissage** : Laplace ici (`ALPHA = 1.0`), *discounting* absolu dans
  le VOMM.

> **Thèse 6 — Le held-out donne un plafond honnête** qui, sur nos données, **ne monte
> pas** avec l'ordre (contrairement au plug-in), voire redescend : sur une journée, un
> contexte long généralise moins bien → plus de surprise → borne plus basse.

**Limite propre à notre étude** : le held-out **coupe la journée en deux**, or le
comportement de l'utilisateur **dépend du moment de la journée** (non-stationnarité) —
donc train et test ne sont pas i.i.d. Lu et al. (2013) montrent de même que la
stationnarité des trajectoires conditionne la validité des bornes (P^max borne l'accuracy
sur les trajectoires stationnaires, pas sur les autres). ⚠️ Ne **pas** attribuer cette
critique à Ikanovic & Mollgaard : leur « stationarity » désigne l'**immobilité** (les gens
restent au même endroit), pas la stationnarité statistique du processus.

### 5.3 Lempel-Ziv (LZ) — le bon estimateur, mais **trop lent pour nos séquences**

Estime le **taux d'entropie** directement (tous ordres) via la compressibilité, sans
choisir d'ordre. Pour chaque position `i` :
$$\Lambda_i = \text{longueur de la plus courte sous-chaîne démarrant en } i \text{ jamais vue avant } i$$
$$S_{LZ} = \left(\frac{1}{n}\sum_{i=1}^n \Lambda_i\right)^{-1}\log_2 n \qquad(\text{via } \Lambda_i \sim \log_2 n / S)$$

- **Ne s'effondre pas** : mesure la compressibilité réelle, pas une distribution d'ordre
  fixe. **Asymptotiquement non biaisé** (Kontoyiannis et al. 1998).
- **Version par ordre maximal** (pour obtenir un axe « ordre ») : on plafonne
  `Λ_i^{(k)} = \min(\Lambda_i, k+1)` (LZ n'exploite qu'au plus k symboles de contexte).
  `P^max_LZ^{(k)}` **monte** avec k, mais honnêtement (le contexte long aide *vraiment* à
  comprimer, ce n'est pas de la mémorisation).

> **Thèse 7 — LZ converge trop lentement pour une journée.** Le taux de convergence des
> estimateurs par longueurs de correspondance est de l'ordre **(log n)^(-1/2)** — extrêmement
> lent. Song et al. avaient **des mois** par utilisateur ; nous, ~44 records/jour en
> moyenne, 13 en médiane.
> Vérification jouet : sur 80 symboles aléatoires sur 8 cellules, LZ donne **2.49 bits** au
> lieu de `log_2(8)=3` — biais visible dès n=80. Donc LZ est **le** référent théorique mais
> **inexploitable par utilisateur-jour** ici.

### 5.4 Tableau récapitulatif

| Estimateur | Idée | S'effondre à haut ordre ? | Biais | Sur 1 journée courte |
|---|---|---|---|---|
| Plug-in | in-sample | **oui** (→ P^max=1) | négatif | trompeur |
| Held-out | cross-entropie retenue | non | positif (conservateur) | robuste, mais viole la stationnarité |
| Lempel-Ziv | compressibilité | non | ~nul (asympt.) | **bruité** (convergence lente) |

**Conclusion méthodologique** : aucun estimateur ne donne un P^max *par utilisateur-jour*
pleinement satisfaisant. Held-out et LZ sous-estiment (held-out par le lissage + la
non-stationnarité, LZ par le bruit). On garde donc `S_unc`/`S_cond` en **assumant** que
c'est le plafond d'un prédicteur sans mémoire / d'ordre k, et non une borne absolue.

---

## 6. Pièges et cohérences (à respecter dans le rapport)

- **Piège n°1 — même représentation partout.** L'entropie doit être calculée sur la
  **même** séquence que celle où l'accuracy est mesurée. Sur le test dédupliqué (sans
  self-transitions), le top-1 des modèles (« rester ») est *toujours* faux → Markov
  s'effondre à ~0 et `S_cond` dégénère. On travaille donc en représentation **`no_duplicate`
  (avec self-transitions)** pour que tout soit cohérent — c'est aussi la tâche de Song et
  al. (prédire la prochaine localisation, où *rester* est une réponse valide).
- **Piège n°2 — micro vs macro-averaging.** L'accuracy poolée par prédiction sur-pondère
  les gros utilisateurs (souvent des appareils M2M). Pour la prévisibilité *par
  utilisateur*, on moyenne par utilisateur, et on **exclut les M2M** (`n_records > 512`).
- **Piège n°3 — causalité.** Toute feature d'entropie servant à la prédiction doit être
  calculée sur l'**historique jusqu'à l'instant courant** (sinon fuite du futur). Le
  `move_rate` global fuiterait ; sa version « jusqu'à maintenant » est causale.

---

## 7. Intégrer l'entropie *dans* la prédiction (demande du tuteur)

### 7.1 Distinction conceptuelle (thèse)

> **Thèse 8 — « Ajouter du bruit selon l'entropie » ≠ « entropie comme feature ».**
> Moduler la confiance par du bruit (temperature scaling) **ne change pas l'argmax** →
> n'améliore **pas** l'ACC@1 (seulement la calibration / permet l'abstention). En
> revanche, donner l'entropie comme **feature** à un modèle ML lui permet d'**apprendre**
> la modulation et peut *vraiment* améliorer la décision. C'est la même idée « le modèle
> module selon l'entropie », mais **apprise** au lieu d'imposée arbitrairement.

### 7.2 La bonne tâche : move/stay (binaire)

Une entropie *par utilisateur* dit *à quel point* il est prévisible, pas *où* il va.
Elle est donc utile pour **« va-t-il bouger ? »** (binaire), pas pour choisir la cellule.
On l'intègre comme **features causales** dans le classifieur move/stay (XGBoost) :

- `running_entropy` = `S_unc` des cellules vues jusqu'ici ;
- `running_cond_entropy` = `H(suivante|actuelle)` sur les transitions vues jusqu'ici ;
- `running_move_rate` = fraction de déplacements depuis le début de la journée ;
- `running_pmax` = Fano(`running_entropy`, nb de cellules distinctes vues) — la
  « prévisibilité à cet instant ».

Protocole d'évaluation : entraîner le **même XGBoost avec vs sans** ces features et
comparer (ROC-AUC, average precision, F1) → réponse chiffrée à « l'entropie aide-t-elle ? »,
+ SHAP pour l'importance. *(Attente honnête : vu la thèse 1, le gain peut être modeste ;
même « l'entropie n'aide pas, le mouvement se joue localement » est un résultat.)*

---

## 8. Chiffres clés obtenus (à citer)

| Mesure | Valeur |
|---|---|
| Self-transitions (no_duplicate, 15 jours) | 72,0 % *(43,6 % = échantillon `sample_for_training`, non représentatif)* |
| Baseline « reste où il est » (ACC@1, 20 000 utilisateurs) | 0.652 |
| Markov ordre 1 / VOMM (ACC@1, mêmes points) | 0.647 / 0.674 |
| Markov ordre 1, causal (accuracy moy. / médiane) | 0.555 / 0.571 |
| Markov ordre 1, split aléatoire | 0.592 |
| Effet week-end (accuracy) | 0.593 (WE) vs 0.544 (semaine) |
| Corrélation accuracy / P^max | 0.844 |
| VOMM vs Markov ordre 1 (ACC@1) | +0.026 seulement |
| Trois estimateurs de P^max à l'ordre 5 (plug-in / held-out / LZ) | 0.98 / 0.57 / 0.63 |
| Réplication de Song, 1 094 958 utilisateurs-jours d'entropie non nulle (P^max / S_rel) | 0.58 / 0.77 |
| P^max par période (matin / journée / soir) | 0.633 / 0.553 / 0.614 |
| Correction bug N de Fano (P^max moyen) | 0.767 → 0.580 |
| Correction bug brentq (NaN) | 37,8 % → 0 % |
| `VOMM(max_order=1)` **avant** la correction du back-off (= unigramme, ACC@1) | ~0.015 |

---

## 9. Références — les 12 retenues dans les rapports

Uniquement des travaux **lus en entier** (PDF dans `../Biblio/papiers/`). Ne pas citer
d'autre papier sans l'avoir lu : Smith 2014, Cuttone 2018, Paninski 2003 et Miller 1955 ont
été retirés des rapports pour cette raison.

**Mobilité et prédiction de la prochaine localisation**
- Song, Qu, Blumm, Barabási (2010). *Limits of Predictability in Human Mobility*.
  **Science 327(5968):1018–1021**. DOI 10.1126/science.1177170. — Fano + Lempel-Ziv, P^max≈0.93.
- Gambs, Killijian, Núñez del Prado Cortez (2012). *Next place prediction using mobility
  Markov chains*. **MPM 2012**. — ancêtre du Markov d'ordre 1.
- González, Hidalgo, Barabási (2008). *Understanding individual human mobility patterns*.
  **Nature 453:779–782**. — les gens reviennent à un petit nombre de lieux.
- Lu, Wetter, Bharti, Tatem, Bengtsson (2013). *Approaching the Limit of Predictability in
  Human Mobility*. **Scientific Reports 3:2923**. — CDR Côte d'Ivoire : P^max 0,88, Markov
  0,91 (au-dessus de la borne), ordres > 1 inutiles, corrélation accuracy/P^max 0,80.
- Ikanovic & Mollgaard (2017). *An alternative approach to the limits of predictability in
  human mobility*. **EPJ Data Science 6:12**. — la borne reflète surtout l'**immobilité** ;
  0,71 pour la prochaine localisation *distincte*, Markov d'ordre 1 à 0,40.

**Modèles de séquences et lissage**
- Begleiter, El-Yaniv, Yona (2004). *On Prediction Using Variable Order Markov Models*.
  **JAIR 22:385–421**. — VOMM.
- Chen & Goodman (1999). *An empirical study of smoothing techniques for language modeling*.
  **Computer Speech & Language 13(4):359–394**. — held-out, perplexité, *discounting* absolu,
  Kneser-Ney.
- Ke, He, Liu (2021). *Rethinking Positional Encoding in Language Pre-training*. **ICLR
  2021** (arXiv 2006.15595). — TUPE.

**Théorie de l'information et estimation de l'entropie**
- Cover & Thomas. *Elements of Information Theory* (2ᵉ éd., Wiley, 2006). — Fano ; cross-entropie.
- Kontoyiannis, Algoet, Suhov, Wyner (1998). *Nonparametric entropy estimation for
  stationary processes…* **IEEE Trans. IT 44(3):1319–1327**. — estimateur LZ ; taux (log n)^(-1/2).
- Antos & Kontoyiannis (2001). *Convergence properties of functional estimates for discrete
  distributions*. **Random Structures & Algorithms 19(3–4):163–193**. — pas de vitesse de
  convergence universelle pour le plug-in.
- Brown et al. (1992). *An estimate of an upper bound for the entropy of English*.
  **Computational Linguistics 18(1):31–40**. — cross-entropie held-out = borne supérieure.

---

## 10. Fichiers du code correspondants (pour référence)

- `Machin_learning/` : entropies (`transition_emtropy.py`), P^max de Fano
  (`maximal_previsibility.py`), Markov causal (`markov_sequential_prediction.py`).
- `simple_predictor/PREDICTABILITY_vs_accuracy.py` : accuracy vs P^max (ordre 1, par utilisateur).
- `simple_predictor/PREDICTABILITY_by_order_calendar.py` : par ordre (plug-in), calendriers.
- `simple_predictor/PREDICTABILITY_heldout_bound.py` : borne held-out.
- `simple_predictor/PREDICTABILITY_lempelziv_bound.py` : borne LZ par ordre maximal.
- `simple_predictor/COMPARE_markov_vs_vomm.py` : Markov ordre 1 vs VOMM (mêmes points).
- `simple_predictor/models.py` : `NaiveMarkovChain`, `VOMM`, `MovementPredictorV2` (+ features
  d'entropie causales), `_pmax_fano`.
- `simple_predictor/MOVEMENT_PREDICTION_*` : classifieur move/stay + features d'entropie.
