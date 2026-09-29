from utils import HelperVOMM
import math
import json
import math
import numpy as np
import collections
from pathlib import Path
from utils import HelperData, HelperVOMM


def _pmax_fano(S, N):
    """Fano maximum-predictability bound: solve S = H(p) + (1-p)*log2(N-1) for the
    max p on [1/N, 1] (self-contained bisection, no scipy). Used by the causal
    entropy features of MovementPredictorV2. Mirrors
    Machin_learning/maximal_previsibility.compute_pmax."""
    if N <= 1 or S <= 0:
        return 1.0
    log2N = math.log2(N)
    if S >= log2N:
        return 1.0 / N

    def fano(p):
        Hp = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        return Hp + (1 - p) * math.log2(N - 1) - S

    lo, hi = 1.0 / N, 1 - 1e-12   # fano(lo) >= 0, fano(hi) <= 0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fano(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ================ #
# Naive predictors #
# ================ #

class NaiveMarkovChain:
    def __init__(self, order=1, 
                 train_data = None):
        assert train_data is not None, "The following arguments are needed : train_data."
        self.order = str(order)
        self.train_data = train_data  # train_data[k] = dict: context -> Counter(next_state)
        self.context_totals = {
            context: sum(counter.values())
            for context, counter in self.train_data[self.order].items()
        }

        self.sorted_predictions = {}
        # Précalcul des probas
        for context, counter in self.train_data[self.order].items():
            total = sum(counter.values())
            
            sorted_preds = sorted(
                ((k, v / total) for k, v in counter.items()),
                key=lambda x: x[1],
                reverse=True
            )
            
            self.sorted_predictions[context] = sorted_preds
    
    def predict_next(self, context, last,top_k=1):
        if context in self.sorted_predictions:
            return self.sorted_predictions[context][:top_k], self.sorted_predictions[context]  # if length < top_k , outputs the entire list
        else:
            return [(last, 1)] * top_k , [(last, 1)]

class NaiveMarkovChainWithTimeWeight:
    def __init__(self, order=1, 
                 train_data = None, time_weight = None):
        """
        
        =======================================================================================================
        -- train_data[k] = dict: context -> Counter(next_state)
        Counter(next_state) : key -> value
        
        with k being a context length
        context is a str such as "cellID-cellID" 
        
        key is a str "cellID" and value is an int (the number of time the suffix has been seen in the data
        considering the context)
        =======================================================================================================
        
        =======================================================================================================
        -- time_weight : k,v
        k are the keys [str(i) for i in range(24)]
        v are dicts with keys being every 369 cells and the values being a float representing the weight of the specific cell
        at a specific hour of the day.
        =======================================================================================================
        """
        
        
        
        assert train_data is not None, "train_data must not be None."
        assert time_weight is not None, "time_weight must not be None."
        
        self.order = str(order)
        self.train_data = train_data
        self.time_weight = time_weight
        self.context_totals = {
            context: sum(counter.values())
            for context, counter in self.train_data[self.order].items()
        }

        self.sorted_predictions = {}
        # Précalcul des probas
        for context, counter in self.train_data[self.order].items():
            total = sum(counter.values())
            
            # No need to sort them because we have to apply time weight
            non_sorted_preds = [(k, v / total) for k, v in counter.items()]
            self.sorted_predictions[context] = non_sorted_preds
    

    
    
    

    def predict_next(self, context : str, last : str, time : str , top_k=1):
        if context in self.sorted_predictions:
            return self.weight_prediction(context = context, time = time, top_k = top_k) # if length < top_k , outputs the entire list
        else:
            return [(last, 1)] * top_k

# ============================ #
# Variable Order Markov Models #
# ============================ #

class VOMM:
    def __init__(self, max_order=5, discount=0.75,
                 train_data = None, context_totals_train_data_path : Path | None = None, unique_train_data_path : Path | None = None, 
                 unigram_train_data_path : Path | None = None, DATASET_DIR : Path | None = None):
        self.max_order = max_order
        self.discount = discount
        self.helper = HelperVOMM()
        self.cache = {} # VERY useful to increase the speed of predictions
        
        
        if context_totals_train_data_path.exists() and unique_train_data_path.exists() and unigram_train_data_path.exists():
            print("Found features in the directories, skipping computation...")
            with open(context_totals_train_data_path, 'r', encoding='utf-8') as f:
                context_totals_train_data = json.load(f)
            with open(unique_train_data_path, 'r', encoding='utf-8') as f:
                unique_train_data = json.load(f)
            with open(unigram_train_data_path, 'r', encoding='utf-8') as f:
                unigram_counts = json.load(f)
        else:
            print("Features not found in directories, computing them...")
            helper = HelperVOMM()
            helper_data = HelperData()
            context_totals_train_data, unique_train_data = helper.prepare_ngrams(train_data=train_data)
            unigram_counts = helper.prepare_unigram_counts(dataset_filepath=DATASET_DIR)
            helper_data.save_dict(context_totals_train_data,   BASE_DIR=context_totals_train_data_path.parent,  inner_dir=None, file_name=context_totals_train_data_path.name)
            helper_data.save_dict(unique_train_data,  BASE_DIR=unique_train_data_path.parent, inner_dir=None, file_name=unique_train_data_path.name )
            helper_data.save_dict(unigram_counts, BASE_DIR=unigram_train_data_path.parent, inner_dir=None, file_name=unigram_train_data_path.name)
            print("End of computation")
        
        
        if train_data is not None and context_totals_train_data is not None and unique_train_data is not None  \
            and unigram_counts is not None:
            # counts[k] = dict: context -> Counter(next_state)
            self.train_data = {}
            self.context_totals = {}
            self.unique_followers = {}
            self.unigram_counts = {}
            for context_length,context_dict in train_data.items():
                order = int(context_length)
                self.train_data[order] = {}
                self.context_totals[order] = {}
                self.unique_followers[order] = {}
                for context,counter_dict in context_dict.items():
                    context_tuple = self.helper.convert_prefix_string_to_tuple(context)
                    self.train_data[order][context_tuple] = counter_dict
                    self.context_totals[order][context_tuple] = context_totals_train_data[context_length].pop(context)
                    self.unique_followers[order][context_tuple] = unique_train_data[context_length].pop(context)
            self.unigram_counts = unigram_counts
            self.total_unigrams     = sum(unigram_counts.values())
            train_data, context_totals_train_data, unique_train_data = None, None, None
        else:
            raise ValueError("The following arguments are needed : counts, context_totals, unique_followers.")

    def prob(self, context, target):
        return self._recursive_prob(context, target, len(context))

    def _recursive_prob(self, context, target, order):
        """
        inputs :
        - context   : list of strings
        - target    : the target cell which one must calculate its score to be the next cell after the given context
        - order     : length of the context considered, first equal to context length and goes down recursively
        until reaching 0 (the unigram), which is the base case of the recursion.

        The recursion stops at order 0, NOT at order 1: order 1 is the real first-order Markov
        step (train_data[1] = counts of "one cell -> next cell"), which must be discounted and
        backed off like any other order. Stopping at order 1 would skip that level entirely and
        jump straight from order 2 to the context-free unigram.
        """
        if order > self.max_order:
            order = self.max_order

        if order == 0: # unigram : base case, no context left
            return self.unigram_counts.get(target, 0) / self.total_unigrams if self.total_unigrams > 0 else 0

        context = context[-order:] # Extracting the end of the sequence (end of the sequence length = order)

        if context not in self.train_data[order]:
            return self._recursive_prob(context[1:], target, order-1)

        count_hw = self.train_data[order][context][target] if target in self.train_data[order][context] else 0 # If the pair does not exists probability is 0
        count_h = self.context_totals[order][context]

        # discounted probability
        first_term = max(count_hw - self.discount, 0) / count_h

        # lambda (backoff weight)
        num_unique = self.unique_followers[order][context]
        lambda_h = (self.discount * num_unique) / count_h

        # backoff computed with less context (recursively until context of size 0 is reached)
        backoff = self._recursive_prob(context[1:], target, order-1)

        return first_term + lambda_h * backoff # Weighted sum of the probabilities

    def predict_next(self, context, candidates = None):
        context = tuple(context)
        
        if context in self.cache:
            return self.cache[context]
        
        if candidates is None:
            candidates = set()
            for order in range(1, self.max_order+1):
                sub_context = context[-order:]
                if sub_context in self.train_data[order]:
                    candidates.update(self.train_data[order][sub_context].keys())
        
        scores = [(c, self.prob(context, c)) for c in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        self.cache[context] = scores
        return scores

class VOMM_V4:
    def __init__(self, max_order=5, discount=0.75, 
                 train_data = None,
                 context_totals_train_data_path : Path | None = None, unique_train_data_path : Path | None = None, 
                 unigram_train_data_path : Path | None = None, DATASET_DIR : Path | None = None):
        self.max_order = max_order
        self.discount = discount
        self.helper = HelperVOMM()
        
        self.prob_cache = {}  # (context, candidate) → float

        
        if context_totals_train_data_path.exists() and unique_train_data_path.exists() and unigram_train_data_path.exists():
            print("Found features in the directories, skipping computation...")
            with open(context_totals_train_data_path, 'r', encoding='utf-8') as f:
                context_totals_train_data = json.load(f)
            with open(unique_train_data_path, 'r', encoding='utf-8') as f:
                unique_train_data = json.load(f)
            with open(unigram_train_data_path, 'r', encoding='utf-8') as f:
                unigram_counts = json.load(f)
        else:
            print("Features not found in directories, computing them...")
            helper = HelperVOMM()
            helper_data = HelperData()
            context_totals_train_data, unique_train_data = helper.prepare_ngrams(train_data=train_data)
            unigram_counts = helper.prepare_unigram_counts(dataset_filepath=DATASET_DIR)
            helper_data.save_dict(context_totals_train_data,   BASE_DIR=context_totals_train_data_path.parent,  inner_dir=None, file_name=context_totals_train_data_path.name)
            helper_data.save_dict(unique_train_data,  BASE_DIR=unique_train_data_path.parent, inner_dir=None, file_name=unique_train_data_path.name )
            helper_data.save_dict(unigram_counts, BASE_DIR=unigram_train_data_path.parent, inner_dir=None, file_name=unigram_train_data_path.name)
            print("End of computation")
        
        
        if train_data is not None and context_totals_train_data is not None and unique_train_data is not None  \
            and unigram_counts is not None:
            # counts[k] = dict: context -> Counter(next_state)
            self.train_data = {}
            self.context_totals = {}
            self.unique_followers = {}
            self.unigram_counts = {}
            for context_length,context_dict in train_data.items():
                order = int(context_length)
                self.train_data[order] = {}
                self.context_totals[order] = {}
                self.unique_followers[order] = {}
                for context,counter_dict in context_dict.items():
                    context_tuple = self.helper.convert_prefix_string_to_tuple(context)
                    self.train_data[order][context_tuple] = counter_dict
                    self.context_totals[order][context_tuple] = context_totals_train_data[context_length].pop(context)
                    self.unique_followers[order][context_tuple] = unique_train_data[context_length].pop(context)
            self.unigram_counts = unigram_counts
            self.total_unigrams     = sum(unigram_counts.values())
            train_data, context_totals_train_data, unique_train_data = None, None, None
        else:
            raise ValueError("The following arguments are needed : counts, context_totals, unique_followers.")

    def prob(self, context, target):
        return self._recursive_prob(context, target, len(context))

    def _recursive_prob(self, context, target, order):
        """
        inputs : 
        - context   : list of strings
        - target    : the target cell which one must calculate its score to be the next cell after the given context
        - order     : length of the context considered, first equal to context length and goes down recursively
        until reaching 0 (the unigram), which is the base case of the recursion.
        """
        if order > self.max_order:
            order = self.max_order

        if order == 0: # unigram : base case, no context left
            return self.unigram_counts.get(target, 0) / self.total_unigrams if self.total_unigrams > 0 else 0
        
        context = context[-order:] # Extracting the end of the sequence (end of the sequence length = order)
                
        if context not in self.train_data[order]:
            return self._recursive_prob(context[1:], target, order-1)
        
        count_hw = self.train_data[order][context][target] if target in self.train_data[order][context] else 0 # If the pair does not exists probability is 0
        count_h = self.context_totals[order][context]
        
        # discounted probability
        first_term = max(count_hw - self.discount, 0) / count_h
        
        # lambda (backoff weight)
        num_unique = self.unique_followers[order][context]
        lambda_h = (self.discount * num_unique) / count_h
        
        # backoff computed with less context (recursively until context of size 0 is reached)
        backoff = self._recursive_prob(context[1:], target, order-1) 
        
        return first_term + lambda_h * backoff # Weighted sum of the probabilities

    def predict_next_v4(self, context, timestamp, home_cell, activity_cell,
                    user_profile=None,
                    alpha_boost=0.3, alpha_profile=0.1,
                    candidates=None):
        context = tuple(context)

        # --- Résolution des candidats ---
        if candidates is None:
            candidates = set()
            for order in range(1, self.max_order + 1):
                sub = context[-order:]
                if sub in self.train_data[order]:
                    candidates.update(self.train_data[order][sub].keys())

        # --- Récupération des scores VOMM bruts (avec cache) ---
        base_scores = {}
        missing_candidates = []

        for c in candidates:
            # key = (context, c)
            # if key in self.prob_cache:
            #     base_scores[c] = self.prob_cache[key]
            # else:
            missing_candidates.append(c)

        for c in missing_candidates:
            score = self.prob(context, c)
            self.prob_cache[(context, c)] = score
            base_scores[c] = score

        # ================================================================ #
        # Log-space scoring + renormalisation                               #
        # log_score(c) = log(p_vomm(c)) + log(boost_temporal(c))           #
        #                               + log(boost_profile(c))            #
        # Ensuite softmax → distribution valide, somme = 1                 #
        # ================================================================ #

        LOG_EPS = math.log(1e-10)  # plancher pour éviter log(0)

        log_scores = {}
        for c, base in base_scores.items():
            log_p = math.log(base) if base > 0 else LOG_EPS

            log_scores[c] = log_p

        # Renormalisation numériquement stable (soustraction du max avant exp)
        max_log = max(log_scores.values())
        exp_scores = {c: math.exp(ls - max_log) for c, ls in log_scores.items()}
        total = sum(exp_scores.values())
        final_probs = {c: v / total for c, v in exp_scores.items()}

        scores = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        return scores

class VOMM_V5:
    def __init__(self, max_order=5, discount=0.90, 
                 train_data = None,
                 context_totals_train_data_path : Path | None = None, unique_train_data_path : Path | None = None, 
                 unigram_train_data_path : Path | None = None, DATASET_DIR : Path | None = None):
        self.max_order = max_order
        self.discount = discount
        self.helper = HelperVOMM()
        
        self.prob_cache = {}  # (context, candidate) → float

        
        if context_totals_train_data_path.exists() and unique_train_data_path.exists() and unigram_train_data_path.exists():
            print("Found features in the directories, skipping computation...")
            with open(context_totals_train_data_path, 'r', encoding='utf-8') as f:
                context_totals_train_data = json.load(f)
            with open(unique_train_data_path, 'r', encoding='utf-8') as f:
                unique_train_data = json.load(f)
            with open(unigram_train_data_path, 'r', encoding='utf-8') as f:
                unigram_counts = json.load(f)
        else:
            print("Features not found in directories, computing them...")
            helper = HelperVOMM()
            helper_data = HelperData()
            context_totals_train_data, unique_train_data = helper.prepare_ngrams(train_data=train_data)
            unigram_counts = helper.prepare_unigram_counts(dataset_filepath=DATASET_DIR)
            helper_data.save_dict(context_totals_train_data,   BASE_DIR=context_totals_train_data_path.parent,  inner_dir=None, file_name=context_totals_train_data_path.name)
            helper_data.save_dict(unique_train_data,  BASE_DIR=unique_train_data_path.parent, inner_dir=None, file_name=unique_train_data_path.name )
            helper_data.save_dict(unigram_counts, BASE_DIR=unigram_train_data_path.parent, inner_dir=None, file_name=unigram_train_data_path.name)
            print("End of computation")
        
        
        if train_data is not None and context_totals_train_data is not None and unique_train_data is not None  \
            and unigram_counts is not None:
            # counts[k] = dict: context -> Counter(next_state)
            self.train_data = {}
            self.context_totals = {}
            self.unique_followers = {}
            self.unigram_counts = {}
            for context_length,context_dict in train_data.items():
                order = int(context_length)
                self.train_data[order] = {}
                self.context_totals[order] = {}
                self.unique_followers[order] = {}
                for context,counter_dict in context_dict.items():
                    context_tuple = self.helper.convert_prefix_string_to_tuple(context)
                    self.train_data[order][context_tuple] = counter_dict
                    self.context_totals[order][context_tuple] = context_totals_train_data[context_length].pop(context)
                    self.unique_followers[order][context_tuple] = unique_train_data[context_length].pop(context)
            self.unigram_counts = unigram_counts
            self.total_unigrams = sum(unigram_counts.values())
            train_data, context_totals_train_data, unique_train_data = None, None, None
        else:
            raise ValueError("The following arguments are needed : counts, context_totals, unique_followers.")

    def temporal_boost(self, candidate_cell, timestamp, home_cell, activity_cell, alpha=0.3):
        """
        Retourne un facteur multiplicatif doux [1.0, 1+alpha].
        alpha=0.3 → boost max de 30%, calibré pour ne pas écraser le VOMM.
        """
        hour = timestamp // 3600  # heure locale (0-23)
        if hour < 23:
            hour+=1

        if candidate_cell == home_cell:
            # Nuit : 20h-00h et 00h-4h
            if hour >= 20 or hour < 4:
                return 1.0 + alpha
            # Transition douce : 4h-6h (retour progressif à 1.0)
            elif 4 <= hour < 6:
                fade = 1.0 - (hour - 4) / 2.0
                return 1.0 + alpha * fade
            # Transition douce : 18h-20h (montée vers le boost)
            elif 18 <= hour < 20:
                ramp = (hour - 18) / 2.0
                return 1.0 + alpha * ramp

        if candidate_cell == activity_cell:
            # Journée : 8h-18h (heures pleines de l'activité)
            if 8 <= hour < 18:
                return 1.0 + alpha
            # Transitions douces en bord de plage
            elif 5 <= hour < 8:
                ramp = (hour - 5) / 3.0
                return 1.0 + alpha * ramp
            elif 18 <= hour < 20:
                fade = 1.0 - (hour - 18) / 2.0
                return 1.0 + alpha * fade

        return 1.0

    def build_user_profile(self, user_history_cells, decay=0.95):
        """
        Profil de mobilité tenant compte de la concentration et du momentum récent.
        
        - Si l'utilisateur est très concentré (peu de cellules distinctes) → boost fort
        - Si l'utilisateur est dispersé (beaucoup de cellules distinctes) → boost faible
        - Si les dernières cellules divergent de l'historique → momentum de déplacement détecté,
        le poids bascule vers les cellules récentes plutôt que les habituelles
        """
        if not user_history_cells:
            return {}

        n = len(user_history_cells)
        n_distinct = len(set(user_history_cells))

        # ------------------------------------------------------------------ #
        # 1. Concentration : entropie normalisée de la distribution brute     #
        # H_norm ∈ [0, 1] : 0 = toujours la même cellule, 1 = tout équiprobable
        # concentration = 1 - H_norm ∈ [0, 1]                                #
        # ------------------------------------------------------------------ #
        counts = {}
        for c in user_history_cells:
            counts[c] = counts.get(c, 0) + 1

        H = 0.0
        for cnt in counts.values():
            p = cnt / n
            H -= p * math.log(p)

        H_max = math.log(n_distinct) if n_distinct > 1 else 1.0
        H_norm = H / H_max                     # ∈ [0, 1]
        concentration = 1.0 - H_norm           # 1 = très concentré, 0 = très dispersé

        # ------------------------------------------------------------------ #
        # 2. Momentum récent : les k dernières positions divergent-elles      #
        #    de l'historique global ?                                         #
        #    momentum ∈ [0, 1] : 1 = en plein déplacement, 0 = stable        #
        # ------------------------------------------------------------------ #
        k = min(5, n // 3 + 1)                 # fenêtre récente adaptative
        recent_cells = user_history_cells[-k:]
        recent_distinct = set(recent_cells)

        # Cellules récentes absentes de l'historique long (hors fenêtre récente)
        long_term_cells = set(user_history_cells[:-k])
        new_in_recent = recent_distinct - long_term_cells

        # Part de la fenêtre récente occupée par des cellules "nouvelles"
        novelty = sum(1 for c in recent_cells if c in new_in_recent) / k
        momentum = novelty                      # ∈ [0, 1]

        # ------------------------------------------------------------------ #
        # 3. Construction du profil avec decay positionnel                    #
        #    Le decay est modulé : fort si concentré+stable, faible sinon     #
        # ------------------------------------------------------------------ #
        # effective_decay ∈ [decay_min, decay] selon concentration et momentum
        decay_min = 0.5
        effective_decay = decay_min + (decay - decay_min) * concentration * (1.0 - momentum)

        profile = {}
        weight = 1.0
        for cell in reversed(user_history_cells):
            profile[cell] = profile.get(cell, 0) + weight
            weight *= effective_decay

        # ------------------------------------------------------------------ #
        # 4. Scale global du profil = concentration * (1 - momentum)         #
        #    → si dispersé ou en déplacement, le profil pèse peu dans        #
        #      predict_next_v4 via alpha_profile                              #
        # ------------------------------------------------------------------ #
        global_scale = concentration * (1.0 - momentum * 0.7)  # 0.7 : momentum ne tue pas tout

        total = sum(profile.values())
        return {
            c: (v / total) * global_scale
            for c, v in profile.items()
        }

    def prob(self, context, target):
        return self._recursive_prob(context, target, len(context))

    def _recursive_prob(self, context, target, order):
        """
        inputs :
        - context   : list of strings
        - target    : the target cell which one must calculate its score to be the next cell after the given context
        - order     : length of the context considered, first equal to context length and goes down recursively
        until reaching 0 (the unigram), which is the base case of the recursion.

        Same convention as VOMM and VOMM_V4: the recursion stops at order 0, so that order 1
        (train_data[1], the real first-order Markov step) is actually used in the backoff chain.
        """
        if order > self.max_order:
            order = self.max_order

        if order == 0: # unigram : base case, no context left
            return self.unigram_counts.get(target, 0) / self.total_unigrams if self.total_unigrams > 0 else 0

        context = context[-order:] # Extracting the end of the sequence (end of the sequence length = order)

        if context not in self.train_data[order]:
            return self._recursive_prob(context[1:], target, order-1)

        count_hw = self.train_data[order][context][target] if target in self.train_data[order][context] else 0 # If the pair does not exists probability is 0
        count_h = self.context_totals[order][context]

        # discounted probability
        first_term = max(count_hw - self.discount, 0) / count_h

        # lambda (backoff weight)
        num_unique = self.unique_followers[order][context]
        lambda_h = (self.discount * num_unique) / count_h

        # backoff computed with less context (recursively until context of size 0 is reached)
        backoff = self._recursive_prob(context[1:], target, order-1)

        return first_term + lambda_h * backoff # Weighted sum of the probabilities

    def predict_next_v5(self, context, timestamp, home_cell, activity_cell,
                    user_profile=None,
                    alpha_boost=0.3, alpha_profile=0.1,
                    candidates=None):
        context = tuple(context)

        # --- Résolution des candidats ---
        if candidates is None:
            candidates = set()
            for order in range(1, self.max_order + 1):
                sub_context = context[-order:]
                if sub_context in self.train_data[order]:
                    # Candidates is the set of all existing distinct suffixes for each sub context
                    candidates.update(self.train_data[order][sub_context].keys())

        # --- Récupération des scores VOMM bruts (avec cache) ---
        base_scores = {}
        missing_candidates = []

        for c in candidates:
            # key = (context, c)
            # if key in self.prob_cache:
            #     base_scores[c] = self.prob_cache[key]
            # else:
            missing_candidates.append(c)

        for c in missing_candidates:
            score = self.prob(context, c)
            self.prob_cache[(context, c)] = score
            base_scores[c] = score

        # ================================================================ #
        # Log-space scoring + renormalisation                               #
        # log_score(c) = log(p_vomm(c)) + log(boost_temporal(c))           #
        #                               + log(boost_profile(c))            #
        # Ensuite softmax → distribution valide, somme = 1                 #
        # ================================================================ #

        LOG_EPS = math.log(1e-10)  # plancher pour éviter log(0)

        log_scores = {}
        for c, base in base_scores.items():
            log_p = math.log(base) if base > 0 else LOG_EPS

            # Boost temporel → log(multiplicatif) = additif en log-space
            log_boost = 0.0
            if alpha_boost:
                boost = self.temporal_boost(c, timestamp, home_cell, activity_cell, alpha=alpha_boost)
                log_boost = math.log(boost)  # boost ∈ [1, 1+alpha] → log_boost ∈ [0, log(1+alpha)] ≥ 0

            # Boost profil utilisateur
            log_profile = 0.0
            if user_profile and c in user_profile:
                log_profile = math.log(1.0 + alpha_profile * user_profile[c])

            log_scores[c] = log_p + log_boost + log_profile

        # Renormalisation numériquement stable (soustraction du max avant exp)
        max_log = max(log_scores.values())
        exp_scores = {c: math.exp(ls - max_log) for c, ls in log_scores.items()}
        total = sum(exp_scores.values())
        final_probs = {c: v / total for c, v in exp_scores.items()}

        scores = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        return scores





# =================== #
# Mouvement Predictor #
# =================== #
class MovementPredictor:
    """
    Predict either a user moves or not between 2 records (1 = move) or (0 = no move),
    based on its daily history (depuis minuit) with records and timestamps.
    """

    # ------------------------------------------------------------------ #
    #  HELPERS INTERNES                                                   #
    # ------------------------------------------------------------------ #

    def get_first_cells(self, day_cells_np, day_timestamps_np, threshold=14410):
        # Recherche dichotomique ultra-rapide sur le tableau NumPy
        idx = np.searchsorted(day_timestamps_np, threshold)
        return day_cells_np[:idx]

    def _get_anchor_cell(self, day_cells: list, day_timestamps : list) -> str | None:
        """
        Cellule d'ancrage = première cellule stable de la journée.
        On prend soit toutes les cellules dans les 4 premières heures de la journée s'il y en a plus
        de 5 (si une personne rentre chez elles vers 3h du matin par exemple ou autre),
        ou bien on prend les 5 premiers records dans le cas échéants
        S'il n'y a pas de majorité absolue, la personne n'a pas de cell d'ancrage (None puis géré plus tard)
        """
        
        first_cells = self.get_first_cells(day_cells_np=np.array(day_cells), day_timestamps_np=np.array(day_timestamps)
                                           ,threshold=14410)
        
        min_window = 5
        if len(first_cells) > min_window:
            counts = collections.Counter(first_cells)
        else:
            counts = collections.Counter(day_cells[:min_window])
        return None if counts.most_common(1)[0][1] == 1 else counts.most_common(1)[0][0]

    def _get_transition_indices(self, day_cells: list) -> list[int]:
        """Indices où day_cells[i] != day_cells[i-1]."""
        return [i for i in range(1, len(day_cells)) if day_cells[i] != day_cells[i - 1]]

    def _first_real_departure_index(self, day_cells: list, anchor: str | None) -> int | None:
        """
        Index du premier record où l'utilisateur quitte la cellule d'ancrage
        et ne revient pas immédiatement (durée ≥ 2 records consécutifs hors anchor).
        Filtre les micro-transitions parasites (rebond sur antenne voisine).
        """
        if anchor is None or len(day_cells) < 3:
            return 0 # Si pas d'anchor, la personne n'a pas d'ancre donc est partie directement
        i = 0
        while i < len(day_cells):
            if day_cells[i] != anchor:
                # Vérifier qu'on reste hors ancrage au moins 1 record de plus
                if i + 1 < len(day_cells) and day_cells[i + 1] != anchor:
                    return i
                # Micro-transition ignorée
            i += 1
        return None

    # ------------------------------------------------------------------ #
    #  FEATURE 1 — Statut par rapport à la cellule d'ancrage             #
    # ------------------------------------------------------------------ #

    def is_at_anchor_cell(self, day_cells: list) -> int:
        """
        1 si l'utilisateur est actuellement dans sa cellule d'ancrage, 0 sinon.
        Signal fort : toujours à 1 en Phase 0 et Phase 2, toujours 0 en Phase 1.
        """
        anchor = self.current_anchor_cell
        if anchor is None or not day_cells:
            return -1  # Si anchor est None, on dit que la personne n'a pas d'ancre (return 2) 
        return int(day_cells[-1] == anchor)

    # ------------------------------------------------------------------ #
    #  FEATURE 2 — Déclenchement du mouvement                           #
    # ------------------------------------------------------------------ #

    def has_departed_today(self) -> int:
        """
        1 si l'utilisateur a déjà quitté sa cellule d'ancrage au moins une fois.
        Discrimine Phase 0 (0) de Phase 1 et Phase 2 (1).
        """
        dep_idx = self.first_real_departure_index
        return int(dep_idx is not None)

    def first_departure_elapsed_h(self, day_cells: list, day_timestamps: list) -> float:
        """
        Temps écoulé depuis le premier vrai départ (en heures).
        0.0 si pas encore parti. Capture "à quelle étape du trip en est-on".
        Un utilisateur qui a quitté son ancre il y a 6h est probablement
        en fin de trajet ; s'il vient de partir, il est en début.
        """
        if not day_cells or not day_timestamps:
            return 0.0
        dep_idx = self.first_real_departure_index
        if dep_idx is None:
            return 0.0
        return (day_timestamps[-1] - day_timestamps[dep_idx]) / 3600

    def time_in_anchor_before_departure_h(self, day_cells: list, day_timestamps: list) -> float:
        """
        Durée passée dans la cellule d'ancrage AVANT le premier départ (en heures).
        Capture la "durée de la phase stationnaire initiale" — un proxy de
        l'heure de départ réelle même sans connaître l'heure absolue.
        Retourne la durée totale observée si pas encore parti.
        """
        if not day_cells or not day_timestamps:
            return 0.0
        dep_idx = self.first_real_departure_index
        if dep_idx is None:
            # Pas encore parti : toute la séquence est dans l'ancre
            return (day_timestamps[-1] - day_timestamps[0]) / 3600
        return (day_timestamps[dep_idx] - day_timestamps[0]) / 3600

    # ------------------------------------------------------------------ #
    #  FEATURE 3 — Retour à l'ancre                                      #
    # ------------------------------------------------------------------ #

    def returned_to_anchor(self, day_cells: list) -> int:
        """
        1 si l'utilisateur est reparti ET est revenu dans sa cellule d'ancrage.
        Signal fort d'arrêt définitif (Phase 2).
        Condition : a quitté l'ancre ET est actuellement de retour dans l'ancre.
        """
        anchor = self.current_anchor_cell
        dep_idx = self.first_real_departure_index
        if dep_idx is None:
            return 0  # jamais parti → pas de "retour"
        return int(day_cells[-1] == anchor)

    def time_since_return_h(self, day_cells: list, day_timestamps: list) -> float:
        """
        Temps depuis le retour à l'ancre après le premier départ (en heures).
        0.0 si pas encore revenu. Plus cette valeur est grande, plus
        la probabilité de rester est élevée.
        """
        if not day_cells or not day_timestamps:
            return 0.0
        anchor = self.current_anchor_cell
        dep_idx = self.first_real_departure_index
        if dep_idx is None or day_cells[-1] != anchor:
            return 0.0
        # Trouver l'index du retour : dernier run de l'ancre depuis la fin
        return_idx = len(day_cells) - 1
        for i in range(len(day_cells) - 2, dep_idx, -1):
            if day_cells[i] != anchor:
                return_idx = i + 1
                break
        else:
            return_idx = dep_idx  # cas limite
        return (day_timestamps[-1] - day_timestamps[return_idx]) / 3600

    # ------------------------------------------------------------------ #
    #  FEATURE 4 — Phase du voyage (encodage ordinal)                    #
    # ------------------------------------------------------------------ #

    def trip_phase(self, day_cells: list) -> int:
        """
        Encodage de la phase courante :
            0 = ancré (pas encore parti)
            1 = en déplacement (parti, pas encore revenu)
            2 = rentré (parti puis revenu à l'ancre)
        Feature ordinale très utile pour les arbres de décision.
        """
        anchor = self.current_anchor_cell
        dep_idx = self.first_real_departure_index
        if dep_idx is None:
            return 0
        if day_cells[-1] != anchor:
            return 1
        return 2

    # ------------------------------------------------------------------ #
    #  FEATURE 5 — Accélération locale du mouvement                      #
    # ------------------------------------------------------------------ #

    def local_acceleration(self, day_cells: list) -> float:
        """
        Ratio entre le taux de transition récent (fenêtre courte)
        et le taux de transition global de la journée.

        > 1.0  → le mouvement s'accélère  (début de déplacement)
        ≈ 1.0  → rythme stable
        < 1.0  → le mouvement décélère    (début de stabilisation)

        Particulièrement utile pour détecter le "déclenchement" :
        quand quelqu'un passe de stationnaire à mobile, ce ratio
        explose avant même que first_departure_elapsed_h soit significatif.
        """
        if len(day_cells) < 5:
            return 1.0

        lasts = day_cells[-5:]

        old_trans = 0
        old_trans = 1 if lasts[2] != lasts[3] else 0
        if lasts[3] != lasts[4]:
            old_trans = 2
        elif old_trans != 0:
            old_trans = 0.5


        recent_trans = 0
        recent_trans = 1 if lasts[2] != lasts[3] else 0
        if lasts[3] != lasts[4]:
            recent_trans = 2
        elif recent_trans != 0:
            recent_trans = 0.5


        if old_trans == 0 and recent_trans != 0:
            return 2.0
        elif old_trans == 0 and recent_trans == 0:
            return 0.0
        else:
            return recent_trans - old_trans / old_trans

    # ------------------------------------------------------------------ #
    #  FEATURE 6 — Nombre de trips complets                              #
    # ------------------------------------------------------------------ #

    def n_complete_trips(self, day_cells: list) -> int:
        """
        Nombre de fois où l'utilisateur est parti ET revenu à l'ancre.
        Capture les journées multi-trips (matin + après-midi) vs mono-trip.
        Un utilisateur à n_complete_trips=1 en Phase 2 a très peu de
        chances de repartir ; à n_complete_trips=0 en Phase 1, il est
        en plein premier déplacement.
        """
        anchor = self.current_anchor_cell
        if anchor is None or not day_cells:
            return 0

        trips = 0
        in_trip = False
        for cell in day_cells:
            if not in_trip and cell != anchor:
                in_trip = True
            elif in_trip and cell == anchor:
                trips += 1
                in_trip = False
        return trips

    def current_trip_duration_h(self, day_cells: list, day_timestamps: list) -> float:
        """
        Durée du trip en cours (en heures).
        0.0 si l'utilisateur est ancré ou vient de rentrer.
        Utile combiné à first_departure_elapsed_h : si les deux sont
        élevés, l'utilisateur est loin dans son déplacement.
        """
        if not day_cells or not day_timestamps:
            return 0.0
        anchor = self.current_anchor_cell
        if day_cells[-1] == anchor:
            return 0.0  # ancré ou rentré

        # Trouver le début du trip courant (dernière fois qu'on était dans l'ancre)
        trip_start_idx = None
        for i in range(len(day_cells) - 2, -1, -1):
            if day_cells[i] == anchor:
                trip_start_idx = i + 1
                break

        if trip_start_idx is None:
            # Jamais été dans l'ancre → trip depuis le début
            trip_start_idx = 0

        return (day_timestamps[-1] - day_timestamps[trip_start_idx]) / 3600

    # ------------------------------------------------------------------ #
    #  FEATURE 7 — Diversité cellulaire hors-ancre                      #
    # ------------------------------------------------------------------ #

    def out_of_anchor_entropy(self, day_cells: list) -> float:
        """
        Entropie de Shannon des cellules visitées HORS de l'ancre, normalisée.
        ∈ [0, 1]

        - 0.0 : toujours dans l'ancre, ou toujours dans une seule cellule hors-ancre
        - 1.0 : distribution uniforme sur toutes les cellules hors-ancre

        Sépare les "navetteurs" (A→B→A, entropie basse) des "explorateurs"
        (A→B→C→D→A, entropie haute). Un navetteur en Phase 1 avec entropie
        basse est probablement en train de rentrer ; un explorateur avec
        entropie haute est encore loin de son ancre.
        """
        anchor = self.current_anchor_cell
        out_cells = [c for c in day_cells if c != anchor]
        if not out_cells:
            return 0.0
        counts = collections.Counter(out_cells)
        n = len(out_cells)
        n_distinct = len(counts)
        if n_distinct == 1:
            return 0.0
        H = -sum((c / n) * math.log(c / n) for c in counts.values())
        H_max = math.log(n_distinct)
        return H / H_max









    # ------------------------------------------------------------------ #
    #  Features assembly                                                 #
    # ------------------------------------------------------------------ #

    def extract_features(self, day_cells: list, day_timestamps: list) -> dict:
        """
        Agrège toutes les features en un seul vecteur pour le classifieur.

        Orthogonalité des features :
        ┌─────────────────────────────────────┬──────────────────────────────────────────┐
        │ Feature                             │ Ce qu'elle capture                       │
        ├─────────────────────────────────────┼──────────────────────────────────────────┤
        │ is_at_anchor_cell                   │ Position actuelle / ancre                │
        │ has_departed_today                  │ A-t-on décollé ? (binaire)               │
        │ trip_phase                          │ Phase globale 0/1/2                      │
        │ first_departure_elapsed_h           │ Depuis combien de temps en trip ?        │
        │ time_in_anchor_before_departure_h   │ Durée phase stationnaire initiale        │
        │ returned_to_anchor                  │ Est-on revenu ? (binaire)                │
        │ time_since_return_h                 │ Depuis combien de temps rentré ?         │
        │ local_acceleration                  │ Le mouvement accélère ou décélère ?      │
        │ n_complete_trips                    │ Combien de trips faits dans la journée ? │
        │ current_trip_duration_h             │ Durée du trip courant                    │
        │ out_of_anchor_entropy               │ Diversité des cellules hors-ancre        │
        └─────────────────────────────────────┴──────────────────────────────────────────┘
        """
        
        self.current_anchor_cell = self._get_anchor_cell(day_cells=day_cells, day_timestamps=day_timestamps)
        self.first_real_departure_index = self._first_real_departure_index(day_cells=day_cells, anchor=self.current_anchor_cell)
        
        return {
            # -- Position vs ancre --
            "is_at_anchor_cell":                self.is_at_anchor_cell(day_cells),
            "trip_phase":                       self.trip_phase(day_cells),
            # -- Déclenchement du mouvement --
            "has_departed_today":               self.has_departed_today(),
            "first_departure_elapsed_h":        self.first_departure_elapsed_h(day_cells, day_timestamps),
            "time_in_anchor_before_dep_h":      self.time_in_anchor_before_departure_h(day_cells, day_timestamps),
            # -- Retour à l'ancre --
            "returned_to_anchor":               self.returned_to_anchor(day_cells),
            "time_since_return_h":              self.time_since_return_h(day_cells, day_timestamps),
            # -- Dynamique du mouvement --
            "local_acceleration":               self.local_acceleration(day_cells),
            "n_complete_trips":                 self.n_complete_trips(day_cells),
            "current_trip_duration_h":          self.current_trip_duration_h(day_cells, day_timestamps),
            # -- Profil de mobilité --
            "out_of_anchor_entropy":            self.out_of_anchor_entropy(day_cells),
        }


    # ------------------------------------------------------------------ #
    #  Score                                                             #
    # ------------------------------------------------------------------ #
    # NOTE : une méthode heuristique `predict_movement` existait ici. Elle combinait
    # à la main des features (current_streak, is_home_now, momentum, concentration,
    # recent_variability, inter_transition_rhythm_h...) issues d'une ANCIENNE version
    # de extract_features et qui ne sont plus produites : elle levait un TypeError dès
    # l'appel (elle passait 4 arguments à extract_features qui n'en prend que 2).
    # Elle a été supprimée : la décision move/stay est apprise par XGBoost /
    # RandomForest / LSTM dans les scripts MOVEMENT_PREDICTION_fit_features*.py et
    # MOVEMENT_PREDICTION_train_lstm.py, à partir du vecteur rendu par extract_features.


class MovementPredictorV2:
    """
    V2 — Features recentrées sur les signaux discrets de changement d'état.

    Enseignements de la V1 :
    - Les features temporelles absolues (elapsed_h, duration_h) ont importance ≈ 0
      car sans heure absolue elles sont non-informatives.
    - Les 5 features dominantes sont toutes des signaux de RÉGIME : est-ce que
      le comportement est en train de changer RIGHT NOW ?
    - Stratégie V2 : maximiser les features de type "signal de rupture local"
      et réduire les features continues au profit de features discrètes/ordinales.
    """

    # ------------------------------------------------------------------ #
    #  CACHE : calculs partagés entre features                           #
    # ------------------------------------------------------------------ #

    def _compute_cache(self, day_cells: list, day_timestamps: list):
        """
        Pré-calcule les dérivées coûteuses une seule fois.
        Toutes les features lisent self._cache plutôt que de recalculer.
        """
        anchor = self._get_anchor_cell(day_cells)
        dep_idx = self._first_real_departure_index(day_cells, anchor)

        # Transitions globales (indices où ça change)
        trans_indices = [
            i for i in range(1, len(day_cells))
            if day_cells[i] != day_cells[i - 1]
        ]

        # Nombre de trips complets
        trips, in_trip = 0, False
        for cell in day_cells:
            if not in_trip and cell != anchor:
                in_trip = True
            elif in_trip and cell == anchor:
                trips += 1
                in_trip = False

        self._cache = {
            "anchor":        anchor,
            "dep_idx":       dep_idx,
            "trans_indices": trans_indices,
            "n_trips":       trips,
            "in_trip":       in_trip,  # True si trip en cours non clôturé
        }

    # ------------------------------------------------------------------ #
    #  HELPERS                                                           #
    # ------------------------------------------------------------------ #

    def _get_anchor_cell(self, day_cells: list) -> str | None:
        if not day_cells:
            return None
        window = min(5, len(day_cells))
        counts = collections.Counter(day_cells[:window])
        return counts.most_common(1)[0][0]

    def _first_real_departure_index(self, day_cells: list, anchor: str | None) -> int | None:
        """
        Premier départ non-parasite : quitte l'ancre ET reste hors-ancre
        au moins 1 record de plus (filtre les bounces d'antenne).
        """
        if anchor is None or len(day_cells) < 3:
            return None
        i = 0
        while i < len(day_cells):
            if day_cells[i] != anchor:
                if i + 1 < len(day_cells) and day_cells[i + 1] != anchor:
                    return i
            i += 1
        return None

    # ================================================================== #
    #  GROUPE A — Signaux de phase (conservés de V1, reformulés)        #
    # ================================================================== #

    def trip_phase(self, day_cells: list) -> int:
        """
        0 = ancré (jamais parti)
        1 = en déplacement (parti, pas revenu)
        2 = rentré (parti puis revenu à l'ancre)
        """
        anchor  = self._cache["anchor"]
        dep_idx = self._cache["dep_idx"]
        if dep_idx is None:
            return 0
        return 2 if day_cells[-1] == anchor else 1

    def is_at_anchor_cell(self, day_cells: list) -> int:
        """1 si cellule courante == cellule d'ancrage."""
        anchor = self._cache["anchor"]
        if anchor is None or not day_cells:
            return 1
        return int(day_cells[-1] == anchor)

    def returned_to_anchor(self, day_cells: list) -> int:
        """1 si parti ET revenu (Phase 2 exactement)."""
        dep_idx = self._cache["dep_idx"]
        if dep_idx is None:
            return 0
        return int(day_cells[-1] == self._cache["anchor"])

    def has_departed_today(self) -> int:
        """1 si au moins un vrai départ observé."""
        return int(self._cache["dep_idx"] is not None)

    # ================================================================== #
    #  GROUPE B — Signaux locaux de rupture (NOUVEAUX)                  #
    # ================================================================== #

    def moving(self, day_cells: list) -> int:
        """
        1 si les 2 dernières transitions sont toutes des changements de cellule.
        Signal fort de mobilité active en ce moment.
        Reprend la feature 'moving' du script d'extraction.
        """
        if len(day_cells) < 3:
            return 0
        return int(
            day_cells[-1] != day_cells[-2] and
            day_cells[-2] != day_cells[-3]
        )

    def is_oscillating(self, day_cells: list, window: int = 6) -> int:
        """
        1 si les `window` derniers records alternent entre exactement 2 cellules.

        Discrimine le bruit réseau (A↔B↔A↔B) d'un vrai déplacement.
        Une oscillation → label stay très probable, réduit les faux positifs.
        Ex : [A, B, A, B, A, B] → 1  /  [A, B, C, B, A, B] → 0
        """
        if len(day_cells) < window:
            return 0
        recent = day_cells[-window:]
        if len(set(recent)) != 2:
            return 0
        return int(all(recent[i] != recent[i + 1] for i in range(len(recent) - 1)))

    def phase_transition_signal(self, day_cells: list) -> int:
        """
        Détecte si on est EXACTEMENT au moment d'une bascule de phase.

        -1 = vient de rentrer (dernier record = retour à l'ancre après trip)
         0 = régime stable (ancré depuis longtemps OU en déplacement continu)
        +1 = vient de partir (premier record hors-ancre après phase 0)

        C'est la feature la plus directement liée au label : si +1, move très
        probable ; si -1, stay quasi-certain.
        """
        anchor  = self._cache["anchor"]
        dep_idx = self._cache["dep_idx"]
        n = len(day_cells)

        if n < 2 or anchor is None:
            return 0

        prev, curr = day_cells[-2], day_cells[-1]

        # Vient de rentrer : transition vers l'ancre après avoir été dehors
        if curr == anchor and prev != anchor and dep_idx is not None:
            return -1

        # Vient de partir : transition hors-ancre et c'est le début du départ
        # (dep_idx pointe exactement sur le record courant ou le précédent)
        if curr != anchor and prev == anchor:
            # Vérifier que ce n'est pas une micro-transition (le record suivant
            # n'existe pas encore → on ne peut pas filtrer, on signale quand même)
            return 1

        return 0

    def consecutive_non_anchor_records(self, day_cells: list) -> int:
        """
        Nombre de records consécutifs hors-ancre depuis la fin.
        0 = actuellement dans l'ancre.
        Capture "depuis combien de records on est dehors" — plus robuste
        que current_trip_duration_h (qui est quasi nul en importance V1).

        Ex : ancre=A, cells=[A,A,B,C,B,C] → 4
        """
        anchor = self._cache["anchor"]
        if anchor is None or not day_cells or day_cells[-1] == anchor:
            return 0
        count = 0
        for cell in reversed(day_cells):
            if cell != anchor:
                count += 1
            else:
                break
        return count

    def consecutive_anchor_records(self, day_cells: list) -> int:
        """
        Nombre de records consécutifs DANS l'ancre depuis la fin.
        0 = actuellement hors de l'ancre.
        Complément symétrique de consecutive_non_anchor_records.
        Long run dans l'ancre → stay très probable.

        Ex : ancre=A, cells=[B,C,A,A,A,A] → 4
        """
        anchor = self._cache["anchor"]
        if anchor is None or not day_cells or day_cells[-1] != anchor:
            return 0
        count = 0
        for cell in reversed(day_cells):
            if cell == anchor:
                count += 1
            else:
                break
        return count

    def local_acceleration(self, day_cells: list) -> float:
        """
        Ratio taux-de-transition-récent / taux-de-transition-global.
        > 1 → accélération (départ imminent)
        < 1 → décélération (stabilisation)
        Conservée de V1 car top-5 importance.
        """
        if len(day_cells) < 4:
            return 1.0
        global_trans = len(self._cache["trans_indices"])
        global_rate  = global_trans / (len(day_cells) - 1) if len(day_cells) > 1 else 0.0

        window = min(6, len(day_cells))
        recent = day_cells[-window:]
        recent_trans = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        recent_rate  = recent_trans / (len(recent) - 1) if len(recent) > 1 else 0.0

        if global_rate == 0.0:
            return 2.0 if recent_trans > 0 else 0.0
        return recent_rate / global_rate

    # ================================================================== #
    #  GROUPE C — Densité et diversité locale (NOUVEAUX)                #
    # ================================================================== #

    def recent_record_density(self, day_timestamps: list, window_s: int = 600) -> int:
        """
        Nombre de records dans les `window_s` dernières secondes (défaut 10 min).

        Un pic de densité précède souvent un déplacement : quand quelqu'un
        commence à se déplacer, il traverse plusieurs antennes rapidement
        → beaucoup d'enregistrements en peu de temps.
        Valeur haute + is_at_anchor=1 → départ imminent probable.
        """
        if not day_timestamps:
            return 0
        cutoff = day_timestamps[-1] - window_s
        return sum(1 for t in day_timestamps if t >= cutoff)

    def last_n_distinct_cells(self, day_cells: list, n: int = 5) -> int:
        """
        Nombre de cellules distinctes parmi les `n` derniers records.
        ∈ [1, n]

        1 = totalement stationnaire localement
        n = cellule différente à chaque record → mouvement intense

        Complète local_acceleration : capture la diversité spatiale
        récente, pas seulement le taux de transition.
        """
        if not day_cells:
            return 0
        return len(set(day_cells[-n:]))

    def anchor_dominance_ratio(self, day_cells: list) -> float:
        """
        Part des records dans la cellule COURANTE sur l'ensemble de la journée.
        ∈ [0, 1]

        0.9 → utilisateur très ancré dans sa cellule actuelle → stay
        0.1 → utilisateur de passage → move ou retour probable

        Différent de is_at_anchor_cell (binaire) : capture le DEGRÉ
        d'attachement à la cellule courante, pas juste si c'est l'ancre.
        """
        if not day_cells:
            return 1.0
        current = day_cells[-1]
        return day_cells.count(current) / len(day_cells)

    def n_complete_trips(self) -> int:
        """
        Nombre de trips complets (départ + retour à l'ancre) dans la journée.
        Capturé dans le cache.
        """
        return self._cache["n_trips"]

    # ================================================================== #
    #  GROUPE D — Entropie hors-ancre (conservée V1, reformulée)        #
    # ================================================================== #

    def out_of_anchor_entropy(self, day_cells: list) -> float:
        """
        Entropie de Shannon normalisée des cellules hors-ancre. ∈ [0, 1]
        0 = navetteur (A→B→A, une seule cellule hors-ancre)
        1 = explorateur (A→B→C→D→E→A, distribution uniforme)
        """
        anchor    = self._cache["anchor"]
        out_cells = [c for c in day_cells if c != anchor]
        if not out_cells:
            return 0.0
        counts    = collections.Counter(out_cells)
        n         = len(out_cells)
        n_distinct = len(counts)
        if n_distinct == 1:
            return 0.0
        H     = -sum((c / n) * math.log(c / n) for c in counts.values())
        H_max = math.log(n_distinct)
        return H / H_max

    # ================================================================== #
    #  GROUPE E — Entropie causale (calculée sur l'historique connu)     #
    # ================================================================== #

    def running_entropy(self, day_cells: list) -> float:
        """S_unc so far: Shannon entropy (bits) of the visit-frequency distribution
        of the cells seen up to now. Low = concentrated, high = spread out."""
        n = len(day_cells)
        if n == 0:
            return 0.0
        h = 0.0
        for c in collections.Counter(day_cells).values():
            p = c / n
            h -= p * math.log2(p)
        return h

    def running_cond_entropy(self, day_cells: list) -> float:
        """H(next | current) so far, from the order-1 transitions seen up to now
        (bits). Low = routine transitions (A->B->A->B), high = erratic."""
        if len(day_cells) < 2:
            return 0.0
        trans = collections.defaultdict(collections.Counter)
        for a, b in zip(day_cells, day_cells[1:]):
            trans[a][b] += 1
        total = len(day_cells) - 1
        s = 0.0
        for counter in trans.values():
            row = sum(counter.values())
            p_ctx = row / total
            h = 0.0
            for cnt in counter.values():
                p = cnt / row
                h -= p * math.log2(p)
            s += p_ctx * h
        return s

    def running_move_rate(self, day_cells: list) -> float:
        """Fraction of steps so far that changed cell (causal move rate)."""
        if len(day_cells) < 2:
            return 0.0
        moves = sum(1 for i in range(1, len(day_cells)) if day_cells[i] != day_cells[i - 1])
        return moves / (len(day_cells) - 1)

    def running_pmax(self, day_cells: list) -> float:
        """Fano predictability from the running entropy and the number of distinct
        cells seen so far — 'how predictable is this user up to now'. This is the
        feature that most directly embodies 'the model modulates by the entropy'."""
        return _pmax_fano(self.running_entropy(day_cells), len(set(day_cells)))

    # ================================================================== #
    #  FEATURES ASSEMBLY                                                 #
    # ================================================================== #

    def extract_features(self, day_cells: list, day_timestamps: list) -> dict:
        """
        ┌──────────────────────────────────────┬─────────────────────────────────────────┬──────────┐
        │ Feature                              │ Signal                                  │ Type     │
        ├──────────────────────────────────────┼─────────────────────────────────────────┼──────────┤
        │ trip_phase                           │ Phase globale 0/1/2                     │ Ordinal  │
        │ is_at_anchor_cell                    │ Dans l'ancre en ce moment ?             │ Binaire  │
        │ returned_to_anchor                   │ Revenu après départ ?                   │ Binaire  │
        │ has_departed_today                   │ A-t-on décollé ?                        │ Binaire  │
        │ moving                               │ 2 transitions consécutives ?            │ Binaire  │
        │ is_oscillating                       │ Bruit réseau A↔B↔A ?                    │ Binaire  │
        │ phase_transition_signal              │ Bascule de phase RIGHT NOW ?            │ {-1,0,1} │
        │ consecutive_non_anchor_records       │ Records consécutifs hors-ancre          │ Int      │
        │ consecutive_anchor_records           │ Records consécutifs dans l'ancre        │ Int      │
        │ local_acceleration                   │ Mouvement qui s'accélère ?              │ Float    │
        │ recent_record_density                │ Pic d'enregistrements récents           │ Int      │
        │ last_n_distinct_cells                │ Diversité cellulaire locale             │ Int      │
        │ anchor_dominance_ratio               │ Degré d'ancrage dans cellule courante   │ Float    │
        │ n_complete_trips                     │ Trips complets dans la journée          │ Int      │
        │ out_of_anchor_entropy                │ Navetteur vs explorateur                │ Float    │
        └──────────────────────────────────────┴─────────────────────────────────────────┴──────────┘
        """
        self._compute_cache(day_cells, day_timestamps)

        return {
            # -- Signaux de phase --
            "trip_phase":                       self.trip_phase(day_cells),
            "is_at_anchor_cell":                self.is_at_anchor_cell(day_cells),
            "returned_to_anchor":               self.returned_to_anchor(day_cells),
            "has_departed_today":               self.has_departed_today(),
            # -- Signaux locaux de rupture --
            "moving":                           self.moving(day_cells),
            "is_oscillating":                   self.is_oscillating(day_cells),
            "phase_transition_signal":          self.phase_transition_signal(day_cells),
            "consecutive_non_anchor_records":   self.consecutive_non_anchor_records(day_cells),
            "consecutive_anchor_records":       self.consecutive_anchor_records(day_cells),
            "local_acceleration":               self.local_acceleration(day_cells),
            # -- Densité et diversité locale --
            "recent_record_density":            self.recent_record_density(day_timestamps),
            "last_n_distinct_cells":            self.last_n_distinct_cells(day_cells),
            "anchor_dominance_ratio":           self.anchor_dominance_ratio(day_cells),
            # -- Profil journalier --
            "n_complete_trips":                 self.n_complete_trips(),
            "out_of_anchor_entropy":            self.out_of_anchor_entropy(day_cells),
            # -- Entropie causale (predictability so far) --
            "running_entropy":                  self.running_entropy(day_cells),
            "running_cond_entropy":             self.running_cond_entropy(day_cells),
            "running_move_rate":                self.running_move_rate(day_cells),
            "running_pmax":                     self.running_pmax(day_cells),
        }















