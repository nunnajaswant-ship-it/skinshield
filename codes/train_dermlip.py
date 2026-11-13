"""
SkinShield — DermLIP training & evaluation (12-class locked set).

Runs on Google Colab (free T4 GPU). Three phases:
    Phase 1: Zero-shot classification (no training) — establishes the floor.
    Phase 2: MLP head on frozen DermLIP fused features (image+text) — primary model.
    Phase 3: Ablation — image-only vs text-only vs fused (professor-facing proof).

USAGE
    pip install open_clip_torch torch torchvision scikit-learn matplotlib pandas
    python train_dermlip.py --data /content/curated [--model redlessone/DermLIP_PanDerm-base-w-PubMed-256]
                            [--epochs 20] [--batch-size 64] [--out /content/skinshield_out]

INPUT (--data)
    train.csv / val.csv / test.csv  with columns: filename, truncated_caption, label, ...
    images/{split}/{class}/{basename}  (as produced by curate_derm1m.py)

OUTPUT (--out)
    zero_shot_report.json      Phase 1 metrics
    probe_report.json          Phase 2 metrics (fused)
    ablation_report.json       Phase 3 metrics (image / text / fused)
    confusion_matrix.png       fused model confusion matrix
    probe_head.pt              trained MLP head (encoders stay frozen)
    class_names.json
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Class names (human-readable, used for zero-shot prompts)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    "melanoma": "melanoma",
    "basal_cell_carcinoma": "basal cell carcinoma",
    "squamous_cell_carcinoma": "squamous cell carcinoma",
    "atopic_dermatitis": "atopic dermatitis",
    "psoriasis": "psoriasis",
    "hidradenitis_suppurativa": "hidradenitis suppurativa",
    "cutaneous_lupus": "cutaneous lupus erythematosus",
    "pemphigus_vulgaris": "pemphigus vulgaris",
    "bullous_pemphigoid": "bullous pemphigoid",
    "epidermolysis_bullosa": "epidermolysis bullosa",
    "ichthyosis": "ichthyosis",
    "cutaneous_t_cell_lymphoma": "cutaneous T-cell lymphoma",
}

PROMPT_TEMPLATES = [
    "This is a skin image of {}.",
    "A skin image of {}.",
    "{} is shown in this skin image.",
    "Clinical photograph of {}.",
]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(data_dir: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    # Drop rows with empty text (should be rare after curation)
    before = len(df)
    df = df[df["truncated_caption"].notna() & (df["truncated_caption"].astype(str).str.len() > 5)]
    if len(df) < before:
        print(f"  [warn] {split}: dropped {before - len(df)} rows with empty/short text")
    return df.reset_index(drop=True)


def image_path_for(row, data_dir: str, split: str) -> str:
    return os.path.join(data_dir, "images", split, row["label"], os.path.basename(row["filename"]))


def check_images(df: pd.DataFrame, data_dir: str, split: str, sample: int = 200) -> float:
    """Pre-flight check: verify image files exist. Returns fraction present."""
    n = min(sample, len(df))
    idx = np.random.choice(len(df), n, replace=False)
    missing = 0
    for i in idx:
        row = df.iloc[i]
        if not os.path.exists(image_path_for(row, data_dir, split)):
            missing += 1
    frac = 1.0 - missing / n
    print(f"  [check] {split}: {n - missing}/{n} sample images present ({frac:.0%})")
    if frac < 0.95:
        print(f"  [check] *** WARNING: {missing}/{n} sample images MISSING in {split} ***")
        print(f"  [check] Expected path example: {image_path_for(df.iloc[idx[0]], data_dir, split)}")
        print(f"  [check] If images are missing, the model trains on zero tensors and image "
              f"features are useless. Fix the data before trusting results.")
    return frac


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str, token: str = ""):
    import open_clip

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    if "/" not in model_id:
        # Standard open_clip model name (e.g. ViT-B-32) — random init, no download.
        # Used for local smoke tests; production uses the HF DermLIP repos below.
        print(f"[model] loading standard open_clip model '{model_id}' (random init)")
        model, _, transform = open_clip.create_model_and_transforms(model_id, pretrained=None)
        tokenizer = open_clip.get_tokenizer(model_id)
        model = model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"[model] device={device}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        return model, transform, tokenizer, device

    import json
    from huggingface_hub import hf_hub_download

    # --- Patch the model config: strip keys unsupported by the installed
    #     open_clip version (e.g. 'pretrain_path' in DermLIP's vision_cfg).
    #     The architecture is unchanged; only the pretrained-tower init path
    #     is removed, and the full safetensors weights load normally after.
    #     NOTE: DermLIP_ViT-B-16 uses a standard config (no patching needed).
    #     DermLIP_PanDerm needs the Derm1M open_clip fork instead.
    config_path = hf_hub_download(repo_id=model_id, filename="open_clip_config.json",
                                  token=token or None)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    patched = False
    for section in ("vision_cfg", "text_cfg"):
        sub = cfg.get(section, {})
        for key in ("pretrain_path", "hf_model_name", "hf_tokenizer_name", "hf_pooler_type"):
            if key in sub:
                print(f"[model] stripping '{key}' from {section}")
                del sub[key]
                patched = True
    if patched:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"[model] loading hf-hub:{model_id} ...")
    model, _, transform = open_clip.create_model_and_transforms("hf-hub:" + model_id)
    tokenizer = open_clip.get_tokenizer("hf-hub:" + model_id)
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[model] device={device}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, transform, tokenizer, device


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_images(model, transform, paths, device, batch_size=64):
    feats = []
    n_failed = 0
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
                n_failed += 1
        batch = torch.stack(imgs).to(device)
        feats.append(model.encode_image(batch).cpu().float())
    if n_failed:
        print(f"  [warn] encode_images: {n_failed}/{len(paths)} images FAILED to load "
              f"(zero tensors used — image features are garbage)")
    return torch.cat(feats)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=256):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        feats.append(model.encode_text(tokens).cpu().float())
    return torch.cat(feats)


def fused_features(img_feat, txt_feat):
    """concat([img, txt, img*txt, |img-txt|]) — multimodal bilinear trick."""
    return torch.cat([img_feat, txt_feat, img_feat * txt_feat, (img_feat - txt_feat).abs()], dim=1)


# ---------------------------------------------------------------------------
# Phase 1: zero-shot
# ---------------------------------------------------------------------------
@torch.no_grad()
def zero_shot(model, transform, tokenizer, test_df, data_dir, device):
    print("\n[Phase 1] Zero-shot classification ...")
    class_list = sorted(CLASS_NAMES.keys())
    # Class text features: mean over prompt templates
    text_feats = []
    for cls in class_list:
        prompts = [t.format(CLASS_NAMES[cls]) for t in PROMPT_TEMPLATES]
        feats = encode_texts(model, tokenizer, prompts, device)
        text_feats.append(feats.mean(dim=0))
    text_feats = torch.stack(text_feats)  # [C, D]
    text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

    paths = [image_path_for(r, data_dir, "test") for _, r in test_df.iterrows()]
    img_feats = encode_images(model, transform, paths, device)
    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)

    sims = img_feats @ text_feats.t()  # [N, C]
    preds = [class_list[i] for i in sims.argmax(dim=1).tolist()]
    report = evaluate(test_df["label"].tolist(), preds, class_list)
    report["method"] = "zero_shot"
    with open(os.path.join(args.out, "zero_shot_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)
    return report


# ---------------------------------------------------------------------------
# Phase 2/3: MLP head on frozen features
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=512, n_classes=12, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(X_train, y_train, X_val, y_val, n_classes, epochs, batch_size, device):
    in_dim = X_train.shape[1]
    model = MLPHead(in_dim, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val, best_state = -1, None
    n = len(X_tr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(X_tr[idx])
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f} val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, best_val


def probe_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, class_list,
                   epochs, batch_size, device, tag):
    print(f"\n[Probe] {tag}: training MLP head (frozen encoders) ...")
    model, best_val = train_head(X_train, y_train, X_val, y_val,
                                 len(class_list), epochs, batch_size, device)
    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = [class_list[i] for i in model(X_te).argmax(1).tolist()]
    # y_test is integer class indices; evaluate() expects string labels
    y_test_labels = [class_list[i] for i in y_test]
    report = evaluate(y_test_labels, preds, class_list)
    report["method"] = tag
    report["best_val_acc"] = best_val
    return model, report


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, class_list):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_recall_fscore_support)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = {}
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                       labels=class_list, zero_division=0)
    for i, cls in enumerate(class_list):
        per_class[cls] = {"precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1[i]), 4)}
    cm = confusion_matrix(y_true, y_pred, labels=class_list)
    return {"accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()}


def print_report(report):
    print(f"    accuracy={report['accuracy']}  balanced_acc={report['balanced_accuracy']}  macro_f1={report['macro_f1']}")
    for cls, m in report["per_class"].items():
        print(f"      {cls:<28} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")


def plot_confusion(report, class_list, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_list)))
    ax.set_yticks(range(len(class_list)))
    ax.set_xticklabels([CLASS_NAMES[c] for c in class_list], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([CLASS_NAMES[c] for c in class_list], fontsize=8)
    for i in range(len(class_list)):
        for j in range(len(class_list)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — fused image+text model")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global args
    parser = argparse.ArgumentParser(description="DermLIP training for SkinShield 12-class")
    parser.add_argument("--data", default="/content/curated", help="curated dataset dir")
    parser.add_argument("--model", default="redlessone/DermLIP_ViT-B-16",
                        help="DermLIP model id. NOTE: PanDerm needs the Derm1M open_clip fork; "
                             "ViT-B-16 loads with standard open_clip_torch.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                        help="HF read token (required for gated DermLIP repos)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap each split to N rows (0=all). For quick end-to-end tests.")
    parser.add_argument("--out", default="/content/skinshield_out")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    class_list = sorted(CLASS_NAMES.keys())

    print("=" * 70)
    print("SkinShield — DermLIP training (12-class)")
    print("=" * 70)

    # --- Load data -----------------------------------------------------------
    print("\n[data] loading splits ...")
    train_df = load_split(args.data, "train")
    val_df = load_split(args.data, "val")
    test_df = load_split(args.data, "test")
    if args.limit:
        train_df, val_df, test_df = train_df.head(args.limit).copy(), val_df.head(args.limit).copy(), test_df.head(args.limit).copy()
        if not train_df.empty:
            # keep class coverage even in tiny test runs
            train_df = pd.concat([train_df.head(48)] + [train_df[train_df["label"] == c].head(4) for c in set(train_df["label"])]).drop_duplicates().reset_index(drop=True)
        print(f"\n[data] TEST MODE --limit={args.limit}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("  train class balance:", dict(Counter(train_df["label"]).most_common()))

    # --- Pre-flight image check (fail loudly if images are missing) ------------
    print("\n[check] verifying image files exist ...")
    fracs = [check_images(train_df, args.data, "train"),
             check_images(val_df, args.data, "val"),
             check_images(test_df, args.data, "test")]
    if min(fracs) < 0.95:
        print("\n[check] *** ABORTING: image files are missing. ***")
        print("[check] The curated data must contain images/{split}/{class}/ files.")
        print("[check] Re-run the curation notebook (or fix the data path) and try again.")
        return 1

    # --- Load model ------------------------------------------------------------
    model, transform, tokenizer, device = load_model(args.model, args.token)

    # --- Phase 1: zero-shot -----------------------------------------------------
    if not args.skip_zero_shot:
        zero_shot(model, transform, tokenizer, test_df, args.data, device)

    # --- Extract features for probe phases ---------------------------------------
    print("\n[features] extracting image + text features (frozen encoders) ...")
    feats = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        paths = [image_path_for(r, args.data, split) for _, r in df.iterrows()]
        img = encode_images(model, transform, paths, device, args.batch_size)
        txt = encode_texts(model, tokenizer, df["truncated_caption"].tolist(), device)
        feats[split] = {
            "img": img.numpy(),
            "txt": txt.numpy(),
            "fused": fused_features(img, txt).numpy(),
            "y": [class_list.index(c) for c in df["label"]],
        }
        print(f"  {split}: img {img.shape}, txt {txt.shape}, fused {feats[split]['fused'].shape}")

    # --- Phase 2: fused probe (primary model) -------------------------------------
    fused_model, fused_report = probe_pipeline(
        feats["train"]["fused"], feats["train"]["y"],
        feats["val"]["fused"], feats["val"]["y"],
        feats["test"]["fused"], feats["test"]["y"],
        class_list, args.epochs, args.batch_size, device, "fused_image_text")
    with open(os.path.join(args.out, "probe_report.json"), "w") as f:
        json.dump(fused_report, f, indent=2)
    torch.save(fused_model.state_dict(), os.path.join(args.out, "probe_head.pt"))
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print_report(fused_report)
    plot_confusion(fused_report, class_list, os.path.join(args.out, "confusion_matrix.png"))

    # --- Phase 3: ablation ----------------------------------------------------------
    if not args.skip_ablation:
        print("\n[Phase 3] Ablation: image-only vs text-only vs fused ...")
        ablation = {}
        for tag, key in [("image_only", "img"), ("text_only", "txt"), ("fused_image_text", "fused")]:
            _, rep = probe_pipeline(
                feats["train"][key], feats["train"]["y"],
                feats["val"][key], feats["val"]["y"],
                feats["test"][key], feats["test"]["y"],
                class_list, args.epochs, args.batch_size, device, tag)
            ablation[tag] = {"balanced_accuracy": rep["balanced_accuracy"],
                             "macro_f1": rep["macro_f1"],
                             "accuracy": rep["accuracy"]}
        with open(os.path.join(args.out, "ablation_report.json"), "w") as f:
            json.dump(ablation, f, indent=2)
        print("\n  ABLATION SUMMARY (test set):")
        for tag, m in ablation.items():
            print(f"    {tag:<20} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    print("\nDONE. Outputs in:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())