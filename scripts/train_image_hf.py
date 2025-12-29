import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ImageArgs:
    data_dir: str
    output_dir: str
    base_model: str
    epochs: int
    lr: float
    batch: int
    smoke_test: bool
    max_train_samples: int
    max_eval_samples: int


def compute_metrics_fn(pred):
    logits = pred.predictions
    y = pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="weighted"),
    }


class FERFolderDataset(Dataset):
    def __init__(self, root_dir: str):
        self.ds = ImageFolder(root_dir)

    @property
    def class_to_idx(self):
        return self.ds.class_to_idx

    @property
    def classes(self):
        return self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        if not isinstance(img, Image.Image):
            img = img.convert("RGB")
        return {"image": img, "labels": label}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="img_FER")
    p.add_argument("--output_dir", default=os.path.join("models", "image-fer"))
    p.add_argument("--base_model", default=os.environ.get("BASE_IMAGE_MODEL", "google/vit-base-patch16-224"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--smoke_test", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    args_ns = p.parse_args()

    args = ImageArgs(
        data_dir=args_ns.data_dir,
        output_dir=args_ns.output_dir,
        base_model=args_ns.base_model,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        batch=args_ns.batch,
        smoke_test=bool(args_ns.smoke_test),
        max_train_samples=int(args_ns.max_train_samples),
        max_eval_samples=int(args_ns.max_eval_samples),
    )

    train_root = os.path.join(args.data_dir, "train")
    test_root = os.path.join(args.data_dir, "test")
    if not os.path.isdir(train_root):
        raise RuntimeError("Expected folder: data_dir/train/<class>/...")

    train_ds = FERFolderDataset(train_root)
    eval_ds = FERFolderDataset(test_root) if os.path.isdir(test_root) else None

    if args.max_train_samples and args.max_train_samples > 0:
        train_ds.ds.samples = train_ds.ds.samples[: args.max_train_samples]
        train_ds.ds.imgs = train_ds.ds.samples

    if eval_ds is not None and args.max_eval_samples and args.max_eval_samples > 0:
        eval_ds.ds.samples = eval_ds.ds.samples[: args.max_eval_samples]
        eval_ds.ds.imgs = eval_ds.ds.samples

    label_names = train_ds.classes
    label2id: Dict[str, int] = {n: i for i, n in enumerate(label_names)}
    id2label: Dict[int, str] = {i: n for n, i in label2id.items()}

    proc = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    if args.smoke_test:
        b = [train_ds[i] for i in range(min(2, len(train_ds)))]
        x = proc(images=[item["image"].convert("RGB") for item in b], return_tensors="pt")
        _ = model(**x)
        print("SMOKE_TEST_OK")
        return

    def collate(features):
        images = [f["image"].convert("RGB") for f in features]
        labels = [f["labels"] for f in features]
        batch = proc(images=images, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, collate_fn=collate) if eval_ds is not None else None

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

        if eval_loader is not None:
            model.eval()
            all_logits = []
            all_y = []
            with torch.no_grad():
                for batch in eval_loader:
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
                proc.save_pretrained(args.output_dir)
        else:
            model.save_pretrained(args.output_dir)
            proc.save_pretrained(args.output_dir)

    print("Saved image model to:", os.path.abspath(args.output_dir))
    print("Set backend env:")
    print("  IMAGE_MODEL_ID=", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
