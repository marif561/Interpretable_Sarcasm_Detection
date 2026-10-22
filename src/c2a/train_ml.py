"""Traditional ML on TF-IDF, with and without C2A (paper Sect. 4.1, Table 1).

Classifiers: Gradient Boosting, Logistic Regression, Ridge, Multinomial Naive Bayes, Random
Forest (scikit-learn defaults). For each one the script reports per-class precision / recall /
F1, ROC-AUC, the confusion matrix, and saves misclassified test samples for error analysis.
It also writes the collocations most associated with sarcasm (interpretability).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB

from .augment import TOKEN_PATTERN, CollocationAugmenter, coverage
from .data import load_dataset, split
from .metrics import curves, evaluate, misclassified, save, summary_line

MODELS = {
    "gb": ("Gradient Boosting", lambda s: GradientBoostingClassifier(random_state=s)),
    "lr": ("Logistic Regression", lambda s: LogisticRegression(max_iter=2000, random_state=s)),
    "ridge": ("Ridge", lambda s: RidgeClassifier(random_state=s)),
    "nb": ("Naive Bayes", lambda s: MultinomialNB()),
    "rf": ("Random Forest", lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s)),
}


def scores_of(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV / JSONL file")
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default="is_sarcastic")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--top-k", type=int, default=1000, help="number of collocations used by C2A")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/ml")
    args = ap.parse_args(argv)

    df = load_dataset(args.data, args.lang, args.text_col, args.label_col)
    train, test = split(df, seed=args.seed)
    aug = CollocationAugmenter(top_k=args.top_k, min_df=args.min_df).fit(train["text"])
    inputs = {"tfidf": (train["text"].tolist(), test["text"].tolist()),
              "c2a+tfidf": (aug.transform(train["text"]), aug.transform(test["text"]))}
    out = Path(args.out) / args.lang
    print(f"{args.lang}: {len(train):,} train / {len(test):,} test · {len(aug.bigrams_)} collocations · "
          f"C2A coverage {coverage(aug, test['text']):.0%} of test texts")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aug.explain(train["text"], train["label"], top_n=30)).to_csv(out / "top_collocations.csv", index=False)

    rows = []
    for setting, (x_tr, x_te) in inputs.items():
        vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
        X_tr, X_te = vec.fit_transform(x_tr), vec.transform(x_te)
        for key in args.models:
            name, make = MODELS[key]
            clf = make(args.seed).fit(X_tr, train["label"])
            pred, sc = clf.predict(X_te), scores_of(clf, X_te)
            res = evaluate(test["label"], pred, sc)
            print(f"{setting:<10} {name:<20} {summary_line(res)}")
            save({**res, "curves": curves(test["label"], sc)}, out / setting / f"{key}.json")
            misclassified(test, pred).to_csv(out / setting / f"{key}_misclassified.csv", index=False)
            for c in ("0", "1"):
                pc = res["per_class"][c]
                rows.append({"setting": setting, "model": name, "class": int(c), "precision": pc["precision"],
                             "recall": pc["recall"], "f1": pc["f1-score"], "roc_auc": res.get("roc_auc")})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "per_class_results.csv", index=False)
    print(f"\nresults in {out}/")


if __name__ == "__main__":
    main()
