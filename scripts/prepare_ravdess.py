import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple


RAVDESS_EMOTION_ID_TO_LABEL: Dict[str, str] = {
    # third field in filename: 03-01-XX-...
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

# 将 calm 映射到 7 类（可选）
CANONICAL_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}


def iter_wavs(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                yield os.path.join(dirpath, fn)


def parse_ravdess_label(wav_path: str) -> Optional[str]:
    base = os.path.basename(wav_path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    emo_id = parts[2]
    return RAVDESS_EMOTION_ID_TO_LABEL.get(emo_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ravdess_dir", default="audio_RAVDESS")
    p.add_argument("--out_csv", default=os.path.join("data", "ravdess.csv"))
    p.add_argument("--canonical", type=int, default=1)
    args = p.parse_args()

    ravdess_dir = args.ravdess_dir
    out_csv = args.out_csv
    canonical = bool(args.canonical)

    rows: List[Tuple[str, str]] = []
    for wav in iter_wavs(ravdess_dir):
        label = parse_ravdess_label(wav)
        if label is None:
            continue
        if canonical:
            label = CANONICAL_MAP.get(label, label)
        rows.append((os.path.abspath(wav), label))

    if not rows:
        raise RuntimeError(f"No wav files found under {ravdess_dir}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

    print("Wrote:", os.path.abspath(out_csv))
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
