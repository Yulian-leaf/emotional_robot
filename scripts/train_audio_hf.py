import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
import wave
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from sklearn.metrics import accuracy_score, f1_score


@dataclass
class AudioArgs:
    csv: str
    output_dir: str
    base_model: str
    target_sr: int
    epochs: int
    lr: float
    batch: int
    smoke_test: bool
    max_train_samples: int


def compute_metrics_fn(pred):
    logits = pred.predictions
    y = pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="weighted"),
    }


class RAVDESSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id: Dict[str, int], target_sr: int):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.target_sr = target_sr
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self):
        return len(self.df)

    def _resample_if_needed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        # 使用纯 torch 的 resample（不依赖音频后端）
        return torchaudio.functional.resample(wav, sr, self.target_sr)

    def _load_wav_wave(self, path: str) -> (torch.Tensor, int):
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        if sampwidth != 2:
            raise RuntimeError(f"Unsupported WAV sampwidth={sampwidth} for {path}")

        x = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            x = x.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        wav = torch.from_numpy(x.astype(np.float32) / 32768.0)
        return wav, sr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["label"]

        wav, sr = self._load_wav_wave(path)
        wav = self._resample_if_needed(wav, sr)

        return {"audio": wav.numpy(), "labels": self.label2id[label]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join("data", "ravdess.csv"))
    p.add_argument("--output_dir", default=os.path.join("models", "audio-ser"))
    p.add_argument("--base_model", default=os.environ.get("BASE_AUDIO_MODEL", "superb/hubert-large-superb-er"))
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=0)
    args_ns = p.parse_args()

    args = AudioArgs(
        csv=args_ns.csv,
        output_dir=args_ns.output_dir,
        base_model=args_ns.base_model,
        target_sr=args_ns.target_sr,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        batch=args_ns.batch,
        smoke_test=bool(args_ns.smoke_test),
        max_train_samples=args_ns.max_train_samples,
    )

    df = pd.read_csv(args.csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise RuntimeError("CSV must contain columns: path,label")

    # 简单随机切分（可后续改成 actor-based split）
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * 0.1))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.iloc[: args.max_train_samples]

    labels = sorted(set(df["label"].unique()))
    label2id: Dict[str, int] = {l: i for i, l in enumerate(labels)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    feat = AutoFeatureExtractor.from_pretrained(args.base_model)
    model = AutoModelForAudioClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    train_ds = RAVDESSDataset(train_df, label2id, args.target_sr)
    val_ds = RAVDESSDataset(val_df, label2id, args.target_sr)

    if args.smoke_test:
        # 只跑一次前向，验证数据与模型
        sample = train_ds[0]
        inputs = feat(sample["audio"], sampling_rate=args.target_sr, return_tensors="pt")
        _ = model(**inputs)
        print("SMOKE_TEST_OK")
        return

    def collate(features: List[Dict]):
        audios = [f["audio"] for f in features]
        labels = [f["labels"] for f in features]
        batch = feat(audios, sampling_rate=args.target_sr, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for batch in val_loader:
                y = batch["labels"].detach().cpu().numpy().tolist()
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                all_logits.append(out.logits.detach().cpu().numpy())
                all_y.extend(y)
        logits = np.concatenate(all_logits, axis=0)
        metrics = compute_metrics_fn(type("P", (), {"predictions": logits, "label_ids": np.array(all_y)})())
        if metrics.get("f1", 0.0) > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(args.output_dir)
            feat.save_pretrained(args.output_dir)

    print("Saved audio model to:", os.path.abspath(args.output_dir))
    print("Set backend env:")
    print("  AUDIO_MODEL_ID=", os.path.abspath(args.output_dir))
    print("  AUDIO_ALLOW_TEXT_FALLBACK=0  (recommended)")


if __name__ == "__main__":
    main()
