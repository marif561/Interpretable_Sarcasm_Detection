"""Dataset loading and the 80/20 split used in the paper (seed 42).

English: Hugging Face / Kaggle *News Headlines Dataset for Sarcasm Detection* (Misra & Arora),
columns ``headline``, ``is_sarcastic`` (JSON lines or CSV).
Hindi: Kaggle sarcasm-labelled Hindi Twitter corpus, columns ``hindi_text``, ``is_sarcastic``.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import CLEANERS

DEFAULT_TEXT_COL = {"en": "headline", "hi": "hindi_text"}


def read_any(path: str) -> pd.DataFrame:
    if path.endswith((".json", ".jsonl")):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")


def load_dataset(path: str, lang: str, text_col: str | None = None, label_col: str = "is_sarcastic") -> pd.DataFrame:
    df = read_any(path)
    text_col = text_col or DEFAULT_TEXT_COL[lang]
    for c in (text_col, label_col):
        if c not in df:
            raise KeyError(f"column {c!r} not in {path}; columns: {list(df.columns)}")
    df = df[[text_col, label_col]].dropna().drop_duplicates(subset=[text_col])
    out = pd.DataFrame({"raw": df[text_col].astype(str).values, "label": df[label_col].astype(int).values})
    out["text"] = out["raw"].map(CLEANERS[lang])
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)
