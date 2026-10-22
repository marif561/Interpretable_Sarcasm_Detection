"""Contextual Collocation Augmentation (C2A), paper Sect. 3.3.

The most frequent bigrams are extracted with ``CountVectorizer`` and, whenever a text
contains one of them, it is appended to the text as a single token
("as a single token" -> "as a single token single_token"). Downstream TF-IDF or a
transformer tokenizer then sees the collocation as one explicit feature:

    TF-IDF_aug(t, d) = TF-IDF(t, d) + Σ_{b ∈ B_d} TF-IDF(b, d)

The augmenter is fitted on the training split only, so no information from the test
set leaks into the collocation vocabulary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)[^\s]+"  # whitespace tokens — keeps Devanagari words and emojis intact


class CollocationAugmenter(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list of texts -> list of C2A-augmented texts.

    Parameters
    ----------
    top_k : number of most frequent bigrams to use as collocations
    min_df : ignore bigrams that occur in fewer documents than this
    joiner : string used to merge the two words into one token
    """

    def __init__(self, top_k: int = 1000, min_df: int = 2, joiner: str = "_"):
        self.top_k = top_k
        self.min_df = min_df
        self.joiner = joiner

    def fit(self, texts: Sequence[str], y=None):
        cv = CountVectorizer(ngram_range=(2, 2), token_pattern=TOKEN_PATTERN, lowercase=False,
                             min_df=self.min_df)
        try:
            counts = cv.fit_transform(texts)
        except ValueError:  # no bigram reaches min_df
            self.bigrams_, self.counts_, self._set = [], {}, set()
            return self
        freq = np.asarray(counts.sum(0)).ravel()
        vocab = cv.get_feature_names_out()
        order = np.argsort(-freq, kind="stable")[: self.top_k]
        self.bigrams_ = [vocab[i] for i in order]
        self.counts_ = {vocab[i]: int(freq[i]) for i in order}
        self._set = set(self.bigrams_)
        return self

    def _found(self, text: str) -> List[str]:
        toks = text.split()
        return [f"{a} {b}" for a, b in zip(toks, toks[1:]) if f"{a} {b}" in self._set]

    def transform(self, texts: Sequence[str]) -> List[str]:
        out = []
        for t in texts:
            extra = [bg.replace(" ", self.joiner) for bg in self._found(t)]
            out.append(f"{t} {' '.join(extra)}".strip() if extra else t)
        return out

    def explain(self, texts: Sequence[str], y: Sequence[int], top_n: int = 20,
                alpha: float = 1.0) -> List[Dict]:
        """Interpretability: which collocations are most associated with each class.

        Returns the bigrams with the largest smoothed log-odds of appearing in sarcastic (1)
        vs. non-sarcastic (0) texts, with their document counts per class.
        """
        y = np.asarray(y)
        pos = np.zeros(len(self.bigrams_))
        neg = np.zeros(len(self.bigrams_))
        idx = {b: i for i, b in enumerate(self.bigrams_)}
        for t, label in zip(texts, y):
            for b in set(self._found(t)):
                (pos if label == 1 else neg)[idx[b]] += 1
        n_pos, n_neg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
        log_odds = np.log((pos + alpha) / (n_pos + alpha)) - np.log((neg + alpha) / (n_neg + alpha))
        order = np.argsort(-np.abs(log_odds))[:top_n]
        return [{"bigram": self.bigrams_[i], "log_odds_sarcastic": round(float(log_odds[i]), 3),
                 "sarcastic_docs": int(pos[i]), "non_sarcastic_docs": int(neg[i])} for i in order]


def coverage(aug: CollocationAugmenter, texts: Sequence[str]) -> float:
    """Share of texts that receive at least one collocation token."""
    return float(np.mean([bool(aug._found(t)) for t in texts])) if len(texts) else 0.0


def augment_texts(train: Sequence[str], others: Optional[Sequence[Sequence[str]]] = None, **kw):
    aug = CollocationAugmenter(**kw).fit(train)
    return aug, aug.transform(train), [aug.transform(o) for o in (others or [])]
