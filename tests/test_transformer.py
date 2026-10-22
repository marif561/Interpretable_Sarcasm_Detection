"""Tiny end-to-end fine-tuning run (skipped without torch / transformers / network)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from c2a.train_transformer import main  # noqa: E402

S = Path(__file__).resolve().parents[1] / "data" / "sample"
TINY = "hf-internal-testing/tiny-random-BertForSequenceClassification"


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(TINY)
    except Exception as e:
        pytest.skip(f"cannot download {TINY}: {e}")
    return TINY


def test_finetune_with_c2a(tmp_path, tiny):
    main(["--data", str(S / "english_headlines.jsonl"), "--lang", "en", "--model", tiny, "--c2a",
          "--epochs", "1", "--out", str(tmp_path)])
    res = json.loads(next(tmp_path.rglob("metrics.json")).read_text())
    assert "roc_auc" in res and set(res["per_class"]) == {"0", "1"}
