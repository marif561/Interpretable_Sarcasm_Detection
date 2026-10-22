"""Per-class precision / recall / F1, ROC-AUC, confusion matrix and misclassified samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


def evaluate(y_true, y_pred, scores: Optional[np.ndarray] = None) -> Dict:
    rep = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    res = {"per_class": {str(c): {k: rep[str(c)][k] for k in ("precision", "recall", "f1-score", "support")}
                         for c in (0, 1)},
           "accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"],
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
    if scores is not None and len(set(y_true)) == 2:
        res["roc_auc"] = float(roc_auc_score(y_true, scores))
    return res


def curves(y_true, scores) -> Dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": prec.tolist(), "recall": rec.tolist()}


def misclassified(test_df: pd.DataFrame, pred) -> pd.DataFrame:
    """Error analysis (paper Sect. 3.6): rows where the true and predicted labels differ."""
    df = test_df.assign(true_label=test_df["label"], predicted_label=pred)
    return df[df["true_label"] != df["predicted_label"]]


def summary_line(res: Dict) -> str:
    pc = res["per_class"]
    auc = f" · AUC {res['roc_auc']:.3f}" if "roc_auc" in res else ""
    return (f"F1 0/1 {pc['0']['f1-score']:.2f}/{pc['1']['f1-score']:.2f} · "
            f"P 0/1 {pc['0']['precision']:.2f}/{pc['1']['precision']:.2f} · "
            f"R 0/1 {pc['0']['recall']:.2f}/{pc['1']['recall']:.2f}{auc}")


def save(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
