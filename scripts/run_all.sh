#!/usr/bin/env bash
# Reproduce Tables 1 and 2: traditional ML and transformers, with and without C2A.
# Usage: bash scripts/run_all.sh <english_headlines.json> <hindi_tweets.csv>
set -euo pipefail
EN=${1:?"English headline dataset required"}
HI=${2:?"Hindi Twitter dataset required"}

python -m c2a.train_ml --data "$EN" --lang en
python -m c2a.train_ml --data "$HI" --lang hi

for M in bert-base-uncased roberta-base distilbert-base-uncased; do
  python -m c2a.train_transformer --data "$EN" --lang en --model "$M"
  python -m c2a.train_transformer --data "$EN" --lang en --model "$M" --c2a
done
for M in bert-base-multilingual-cased xlm-roberta-base sentence-transformers/LaBSE; do
  python -m c2a.train_transformer --data "$HI" --lang hi --model "$M"
  python -m c2a.train_transformer --data "$HI" --lang hi --model "$M" --c2a
done
python scripts/plot_curves.py outputs/ml
