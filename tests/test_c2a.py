from pathlib import Path

import pandas as pd
import pytest

from c2a.augment import CollocationAugmenter, coverage
from c2a.data import load_dataset, split
from c2a.metrics import evaluate, misclassified
from c2a.preprocess import clean_en, clean_hi
from c2a.train_ml import main as ml_main

S = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_cleaning():
    assert clean_en("Area Man's 'Big' Win!! https://x.co/1") == "area man s big win"
    out = clean_hi("वाह क्या बात है!!! 😂 @user #tag https://t.co/x")
    assert out == "वाह क्या बात है 😂 tag"  # Devanagari and emoji kept, punctuation and links removed


def test_augmenter_appends_collocations_as_single_tokens():
    train = ["as a single token", "as a single word", "a single token again"]
    aug = CollocationAugmenter(top_k=2, min_df=2).fit(train)
    assert aug.bigrams_ == ["a single", "as a"] or set(aug.bigrams_) == {"a single", "as a"}
    out = aug.transform(["as a single token"])[0]
    assert out.startswith("as a single token ") and "a_single" in out.split() and "as_a" in out.split()
    assert aug.transform(["nothing here"]) == ["nothing here"]  # unchanged when no collocation present


def test_augmenter_fits_on_train_only_and_handles_no_bigrams():
    aug = CollocationAugmenter(min_df=5).fit(["one two", "three four"])
    assert aug.bigrams_ == [] and aug.transform(["one two"]) == ["one two"]
    assert coverage(aug, ["one two"]) == 0.0


def test_explain_ranks_sarcastic_collocations():
    texts = ["wah kya baat hai monday", "wah kya baat hai rain", "news report today", "news report weather"]
    aug = CollocationAugmenter(top_k=10, min_df=2).fit(texts)
    top = aug.explain(texts, [1, 1, 0, 0], top_n=10)
    by = {r["bigram"]: r["log_odds_sarcastic"] for r in top}
    assert by["kya baat"] > 0 and by["news report"] < 0


def test_load_and_split():
    en = load_dataset(str(S / "english_headlines.jsonl"), "en")
    hi = load_dataset(str(S / "hindi_tweets.csv"), "hi")
    assert len(en) == 40 and set(en["label"]) == {0, 1}
    assert any("😂" in t for t in hi["text"])
    tr, te = split(en)
    assert len(te) == 8 and te["label"].sum() == 4


def test_metrics_and_error_analysis():
    res = evaluate([0, 0, 1, 1], [0, 1, 1, 1], scores=[0.1, 0.6, 0.8, 0.9])
    assert res["per_class"]["1"]["recall"] == 1.0 and res["roc_auc"] == 1.0
    df = pd.DataFrame({"text": list("abcd"), "label": [0, 0, 1, 1]})
    assert misclassified(df, [0, 1, 1, 1])["text"].tolist() == ["b"]


@pytest.mark.parametrize("lang,file", [("en", "english_headlines.jsonl"), ("hi", "hindi_tweets.csv")])
def test_ml_end_to_end(tmp_path, lang, file):
    ml_main(["--data", str(S / file), "--lang", lang, "--models", "lr", "ridge", "nb",
             "--min-df", "2", "--out", str(tmp_path)])
    table = pd.read_csv(tmp_path / lang / "per_class_results.csv")
    assert set(table["setting"]) == {"tfidf", "c2a+tfidf"} and len(table) == 2 * 3 * 2
    assert (tmp_path / lang / "top_collocations.csv").exists()
