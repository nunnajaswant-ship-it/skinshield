"""Smoke test for train_dermlip.py — runs the full 3-phase pipeline on a tiny
synthetic dataset with a random-init ViT-B-32 (no downloads, CPU).

Usage:
    python scripts/smoke_test.py
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import train_dermlip as t

CLASSES = ["melanoma", "psoriasis", "atopic_dermatitis"]
SPLITS = {"train": 9, "val": 6, "test": 6}  # 3/2/2 per class


def make_dataset(root: str):
    if os.path.exists(root):
        shutil.rmtree(root)
    rng = np.random.default_rng(0)
    rows = []
    for split, n in SPLITS.items():
        per_class = n // len(CLASSES)
        for cls in CLASSES:
            for i in range(per_class):
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                fname = f"{cls}_{i}.jpg"
                d = os.path.join(root, "images", split, cls)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(img).save(os.path.join(d, fname))
                rows.append({
                    "filename": fname,
                    "truncated_caption": f"A clinical photograph showing {cls.replace('_', ' ')} on the skin.",
                    "label": cls,
                    "source": "smoke",
                })
    for split in SPLITS:
        start = {"train": 0, "val": SPLITS["train"], "test": SPLITS["train"] + SPLITS["val"]}[split]
        end = start + SPLITS[split]
        df = pd.DataFrame(rows[start:end])
        df.to_csv(os.path.join(root, f"{split}.csv"), index=False)
    print(f"[smoke] dataset written to {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/smoke_test")
    ap.add_argument("--out", default="data/smoke_out")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    make_dataset(args.data)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    # Mirror main() but with the tiny model
    t.args = argparse.Namespace(out=args.out)
    os.makedirs(args.out, exist_ok=True)
    class_list = sorted(t.CLASS_NAMES.keys())

    print("=" * 70)
    print("SMOKE TEST — SkinShield training pipeline")
    print("=" * 70)

    train_df = t.load_split(args.data, "train")
    val_df = t.load_split(args.data, "val")
    test_df = t.load_split(args.data, "test")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model, transform, tokenizer, device = t.load_model("ViT-B-32")
    print(f"  model device={device}")

    # Phase 1
    t.zero_shot(model, transform, tokenizer, test_df, args.data, device)

    # Features
    feats = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        paths = [t.image_path_for(r, args.data, split) for _, r in df.iterrows()]
        img = t.encode_images(model, transform, paths, device, batch_size=4)
        txt = t.encode_texts(model, tokenizer, df["truncated_caption"].tolist(), device)
        feats[split] = {
            "img": img.numpy(), "txt": txt.numpy(),
            "fused": t.fused_features(img, txt).numpy(),
            "y": [class_list.index(c) for c in df["label"]],
        }
        print(f"  {split}: img {img.shape}, txt {txt.shape}, fused {feats[split]['fused'].shape}")

    # Phase 2
    fused_model, fused_report = t.probe_pipeline(
        feats["train"]["fused"], feats["train"]["y"],
        feats["val"]["fused"], feats["val"]["y"],
        feats["test"]["fused"], feats["test"]["y"],
        class_list, args.epochs, 4, device, "fused_image_text")
    with open(os.path.join(args.out, "probe_report.json"), "w") as f:
        import json
        json.dump(fused_report, f, indent=2)
    torch_save = __import__("torch").save
    torch_save(fused_model.state_dict(), os.path.join(args.out, "probe_head.pt"))
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(t.CLASS_NAMES, f, indent=2)
    t.print_report(fused_report)
    t.plot_confusion(fused_report, class_list, os.path.join(args.out, "confusion_matrix.png"))

    # Phase 3
    ablation = {}
    for tag, key in [("image_only", "img"), ("text_only", "txt"), ("fused_image_text", "fused")]:
        _, rep = t.probe_pipeline(
            feats["train"][key], feats["train"]["y"],
            feats["val"][key], feats["val"]["y"],
            feats["test"][key], feats["test"]["y"],
            class_list, args.epochs, 4, device, tag)
        ablation[tag] = {"balanced_accuracy": rep["balanced_accuracy"],
                         "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"]}
    with open(os.path.join(args.out, "ablation_report.json"), "w") as f:
        json.dump(ablation, f, indent=2)
    print("\n  ABLATION SUMMARY:")
    for tag, m in ablation.items():
        print(f"    {tag:<20} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    print("\nSMOKE TEST PASSED — all phases completed without error")
    return 0


if __name__ == "__main__":
    sys.exit(main())