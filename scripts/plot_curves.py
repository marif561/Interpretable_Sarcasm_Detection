"""Plot ROC and precision-recall curves (paper Figs. 2-3) from train_ml outputs.

Usage: python scripts/plot_curves.py outputs/ml
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/ml")
for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for setting in ("tfidf", "c2a+tfidf"):
        files = sorted((lang_dir / setting).glob("*.json"))
        if not files:
            continue
        fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10, 4))
        for f in files:
            r = json.loads(f.read_text())
            c = r["curves"]
            ax_pr.plot(c["recall"], c["precision"], label=f.stem)
            ax_roc.plot(c["fpr"], c["tpr"], label=f"{f.stem} (AUC {r.get('roc_auc', float('nan')):.2f})")
        ax_pr.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
        ax_roc.set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
        for ax in (ax_pr, ax_roc):
            ax.legend(fontsize=8)
        fig.suptitle(f"{lang_dir.name.upper()} · {setting}")
        fig.tight_layout()
        out = lang_dir / f"curves_{setting.replace('+', '_')}.png"
        fig.savefig(out, dpi=150)
        print("saved", out)
