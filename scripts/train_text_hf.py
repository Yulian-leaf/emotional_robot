import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import accuracy_score, f1_score


CANONICAL_MAP = {
    # text dataset style
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "sad": "sad",
    "sadness": "sad",
    "joy": "happy",
    "happy": "happy",
    # love 没有对应 7 类，默认并入 happy（你也可以改成 neutral）
    "love": "happy",
    "neutral": "neutral",
}


def parse_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line:
        return None

    # 你当前数据：text;label
    if ";" in line:
        a, b = line.split(";", 1)
        a, b = a.strip(), b.strip()
        # 判断哪个更像 label
        def is_label(x: str) -> bool:
            return len(x) <= 24 and re.fullmatch(r"[A-Za-z0-9_\-]+", x or "") is not None

        if is_label(a) and not is_label(b):
            return b, a
        if is_label(b) and not is_label(a):
            return a, b
        # 默认：text;label
        return a, b

    for sep in ["\t", ",", "||"]:
        if sep in line:
            a, b = line.split(sep, 1)
            a, b = a.strip(), b.strip()
            # 默认 label 在后面
            return a, b

    m = re.match(r"^\s*([A-Za-z0-9_\-]{1,24})\s*[:：]\s*(.+)$", line)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    return None


def load_split(path: str, canonical: bool) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            text, label = parsed
            label = label.strip()
            if canonical:
                label = CANONICAL_MAP.get(label.lower(), label.lower())
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


@dataclass
class TextArgs:
    data_dir: str
    output_dir: str
    base_model: str
    canonical: bool
    epochs: int
    lr: float
    batch: int
    max_length: int
    smoke_test: bool


def compute_metrics_fn(pred):
    logits = pred.predictions
    y = pred.label_ids
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="weighted"),
    }


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="txt_Emotional_Dataset")
    p.add_argument("--output_dir", default=os.path.join("models", "text-emotion"))
    p.add_argument("--base_model", default=os.environ.get("BASE_TEXT_MODEL", "j-hartmann/emotion-english-distilroberta-base"))
    p.add_argument("--canonical", type=int, default=int(os.environ.get("TEXT_CANONICAL", "0")))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--smoke_test", type=int, default=0)
    args_ns = p.parse_args()

    args = TextArgs(
        data_dir=args_ns.data_dir,
        output_dir=args_ns.output_dir,
        base_model=args_ns.base_model,
        canonical=bool(args_ns.canonical),
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        batch=args_ns.batch,
        max_length=args_ns.max_length,
        smoke_test=bool(args_ns.smoke_test),
    )

    train_path = os.path.join(args.data_dir, "train.txt")
    val_path = os.path.join(args.data_dir, "val.txt")
    test_path = os.path.join(args.data_dir, "test.txt")

    train_df = load_split(train_path, canonical=args.canonical)
    val_df = load_split(val_path, canonical=args.canonical)
    test_df = load_split(test_path, canonical=args.canonical)

    if len(train_df) == 0:
        raise RuntimeError(f"Train split is empty after parsing: {train_path}")

    labels = sorted(set(train_df["label"].unique()) | set(val_df["label"].unique()) | set(test_df["label"].unique()))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    train_ds = TextDataset(train_df, tok, label2id, args.max_length)
    val_ds = TextDataset(val_df, tok, label2id, args.max_length) if len(val_df) else None

    if args.smoke_test:
        # 只跑一次前向，验证能跑通
        sample = [train_ds[i] for i in range(min(2, len(train_ds)))]
        batch = tok.pad(sample, return_tensors="pt")
        _ = model(**{k: v for k, v in batch.items() if k != "labels"})
        print("SMOKE_TEST_OK")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    def collate(features):
        return tok.pad(features, return_tensors="pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate) if val_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        if val_loader is not None:
            model.eval()
            all_logits = []
            all_y = []
            with torch.no_grad():
                for batch in val_loader:
                    y = batch["labels"].numpy().tolist()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    all_logits.append(out.logits.detach().cpu().numpy())
                    all_y.extend(y)
            logits = np.concatenate(all_logits, axis=0)
            metrics = compute_metrics_fn(type("P", (), {"predictions": logits, "label_ids": np.array(all_y)})())
            if metrics.get("f1", 0.0) > best_f1:
                best_f1 = metrics["f1"]
                model.save_pretrained(args.output_dir)
                tok.save_pretrained(args.output_dir)
        else:
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)

    print("Saved text model to:", os.path.abspath(args.output_dir))
    print("Set backend env:")
    print("  TEXT_MODEL_ID=", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
