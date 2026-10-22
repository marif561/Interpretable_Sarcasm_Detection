"""Fine-tune a transformer with and without C2A (paper Sect. 3.3.2 / 4.2, Table 2).

English models in the paper: bert-base-uncased, roberta-base, distilbert-base-uncased.
Hindi models: bert-base-multilingual-cased, xlm-roberta-base, sentence-transformers/LaBSE.
With ``--c2a`` the input text is augmented with collocation tokens before tokenisation, so
the encoder sees E_aug(S) = [E(w1), …, E(b), …, E(wn)].

The paper does not list fine-tuning hyperparameters; the defaults below (lr 2e-5, batch 16,
3 epochs, max length 128) are standard values for sentence classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from .augment import CollocationAugmenter
from .data import load_dataset, split
from .metrics import curves, evaluate, misclassified, save, summary_line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lang", choices=["en", "hi"], required=True)
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--c2a", action="store_true", help="append collocation tokens (C2A)")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default="is_sarcastic")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/transformers")
    args = ap.parse_args(argv)

    df = load_dataset(args.data, args.lang, args.text_col, args.label_col)
    train, test = split(df, seed=args.seed)
    if args.c2a:
        aug = CollocationAugmenter(top_k=args.top_k).fit(train["text"])
        train["input"], test["input"] = aug.transform(train["text"]), aug.transform(test["text"])
    else:
        train["input"], test["input"] = train["text"], test["text"]

    tok = AutoTokenizer.from_pretrained(args.model)

    def ds(d):
        return Dataset.from_dict({"text": d["input"].tolist(), "labels": d["label"].tolist()}).map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length), batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "not_sarcastic", 1: "sarcastic"},
        label2id={"not_sarcastic": 0, "sarcastic": 1}, ignore_mismatched_sizes=True)
    tag = f"{args.model.split('/')[-1]}_{'c2a' if args.c2a else 'nofe'}"
    out = Path(args.out) / args.lang / tag
    targs = TrainingArguments(output_dir=str(out / "checkpoints"), learning_rate=args.lr,
                              per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
                              num_train_epochs=args.epochs, save_strategy="no", report_to=[], seed=args.seed,
                              logging_steps=50, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=targs, train_dataset=ds(train), data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    logits = trainer.predict(ds(test)).predictions
    probs = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
    pred = logits.argmax(-1)
    res = evaluate(test["label"], pred, probs)
    print(f"{tag} [{args.lang}] {summary_line(res)}")
    save({**res, "curves": curves(test["label"], probs)}, out / "metrics.json")
    misclassified(test, pred).to_csv(out / "misclassified.csv", index=False)


if __name__ == "__main__":
    main()
