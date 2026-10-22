# Data

Neither dataset is redistributed here. Download them from their sources and follow their licences.

| Dataset | Where | Expected columns |
|---|---|---|
| **News Headlines Dataset for Sarcasm Detection** (Misra & Arora, 2019; [arXiv:2212.06035](https://arxiv.org/abs/2212.06035)) | Hugging Face / Kaggle | `headline`, `is_sarcastic` (optionally `article_link`) |
| **Hindi sarcasm Twitter corpus** | Kaggle | `hindi_text`, `is_sarcastic` |

Both JSON-lines (`.json` / `.jsonl`) and CSV files are accepted. If your copy uses other column names, pass `--text-col` and `--label-col`. Duplicates and empty rows are dropped, and the data is split 80 / 20 with seed 42.

## Samples

[`sample/`](sample/) has 40 **invented** English headlines and 40 **invented** Hindi tweets, with emojis, in the expected formats. They exist only so the code and tests run offline; they are not from the real datasets.
