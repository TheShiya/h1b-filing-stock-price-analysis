"""
Import needed packages
"""
import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from pyjarowinkler import distance
import Levenshtein


SUFFIXES = {
    'inc', 'sezc', 'co', 'vc', 'ag', 'llc', 'inc', 'ltd', 'ltda', 'ab', 'llp', 'sa', 'lp', 'uk', 'sgr', 's', 'et al',
    'gp', 'n a', 'hk', 'oy', 'us', 'as', 'am', 'nv', 'plc', 'p', 'na', 'usa', 'pte', 'l l c', 'l p', 'l l p',
}


def standardize_name(s: str):
    """
    Standardize names by removing/replacing special characters and strip common suffixes
    """
    try:
        # Transform string to lower case, remove symbols and spaces
        s = s.lower()
        s = re.sub(r'[^\w]', ' ', s)
        s = re.sub(r' +', ' ', s)
        s = s.strip()
        # Deal with common apostrophe abbreviations
        s = re.sub(r'\bint\'l\b', 'international', s)
        s = re.sub(r'\bu\.s\.\b', 'us', s)
        # Remove up to 2 common suffixes that are words e.g. 'abc llc'
        for _ in range(2):
            s = re.sub(' ({})$'.format('|'.join(SUFFIXES)), '', s)
        # Remove leading/trailing the's
        s = re.sub(r'^the | the$', '', s)
        # Remove extra white spaces inside string
        s = " ".join(s.split())
        return s
    except Exception as E:
        print(E)
        return s


def jarowinkler_sim(s, t, scaling=0.1, exact_bonus=0.8):
    sim = distance.get_jaro_distance(s, t, winkler=True, scaling=scaling)
    sim = (1 - exact_bonus) * sim + exact_bonus * (s == t)  # Reward exact matches
    return sim


def levenshtein_sim(s, t, exact_bonus=0.8):
    sim = 1 - Levenshtein.distance(s, t) / max(len(s), len(t))
    sim = (1 - exact_bonus) * sim + exact_bonus * (s == t)  # Reward exact matches
    return sim


class StringComparer:

    def __init__(self, sources: list, targets: list, beta=0.5, min_word_weight=0.1):
        """
        :param sources:
        :param targets:
        :param beta:
        :param min_word_weight:
        """
        self.sources = list(set(sources))
        self.targets = list(set([x for x in list(targets) if type(x) == str]))

        # Create inverse frequency weight dictionary
        unigram_count = pd.Series(Counter(' '.join(self.sources + self.targets).split()))
        self.weight_dict = (min_word_weight + 1 / unigram_count**beta).to_dict()

    def slow_compare(self, s, t):
        """
        Slow but probably more accurate metric, should be used as a last resort
        :param s: source string
        :param t: target string
        :return: float similarity score
        """
        s_grams = s.split()  # unigrams + first unigram + first bigram
        t_grams = t.split()  # unigrams + first unigram + first bigram

        # Assign word(s) in s to most similar word in t
        assignments = {idx: 0 for idx in range(len(t_grams))}
        for sg in s_grams:
            best = np.argmax([jarowinkler_sim(sg, tg) for tg in t_grams[:]])
            assignments[best] = max(assignments[best], levenshtein_sim(sg, t_grams[best]))
        weights_raw = []
        t_len = len(t_grams)
        for i, tg in enumerate(t_grams):
            if ' ' in tg:  # if tg is a bigram, take the average weight of each unigram
                first, second = tg.split()
                weights_raw.append((self.weight_dict[first] + self.weight_dict[second])/2)
            else:
                weights_raw.append(self.weight_dict[tg] * (t_len - i)/t_len)
        weights_raw = np.array(weights_raw)
        weights_normalized = weights_raw / weights_raw.sum()
        score_array = np.array(list(assignments.values())) * weights_normalized
        total_score = score_array.sum()
        return total_score

    def get_matches(self, threshold=0.75, num_results=1, include_exact=True):
        """
        Main matching algorithm that tries to match every source to a target screens. Uses
        increasingly more expensive methods until a good similarity score is returned.
        :param threshold:
        :param num_results:
        :param include_exact:
        :return:
        """
        sims = dict({})
        count = 0
        num_sources = len(self.sources)
        start = time.time()
        for s in self.sources:
            for t in self.targets:
                if (s, t) in sims:
                    continue
                if t.split()[0] not in s.split():
                    continue
                if s == t:
                    if include_exact:
                        sims[(s, t)] = 1
                    continue
                levenshtein = levenshtein_sim(s, t)
                if levenshtein >= 0.9:
                    sims[(s, t)] = levenshtein
                    continue
                sim = self.slow_compare(s, t)  # As a last resort, use self.slow_compare()
                if sim <= threshold:
                    continue
                else:
                    sims[(s, t)] = min(sim, self.slow_compare(t, s))
            count += 1
        if len(sims) == 0:
            return pd.DataFrame({'s': [], 't': [], 'sim': []})
        print('String matching complete - {:.2f}s'.format(time.time() - start))
        # Prepare output DataFrame
        pairs_sim = pd.Series(sims).reset_index()
        pairs_sim.columns = ['s', 't', 'sim']
        pairs_sim = pairs_sim[pairs_sim['sim'] >= threshold].sort_values('sim', ascending=False)
        pairs_sim = pairs_sim.groupby('s').head(num_results)
        return pairs_sim
