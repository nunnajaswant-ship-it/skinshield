"""
SkinShield — export trained model to ONNX for Jetson deployment.

Exports three ONNX graphs (matching the training-time feature pipeline exactly):
    image_branch.onnx   image [1,3,224,224]        -> img_feat [1,512]
    text_branch.onnx    tokens [1,77] int64        -> txt_feat [1,512]
    head.onnx           fused [1,2048]             -> logits [1,12]
where fused = concat([img, txt, img*txt, |img-txt|]) — identical to
train_dermlip.fused_features, and the head is the trained MLPHead.

USAGE (Colab, after training):
    python export_onnx.py --data /content/curated \
        --model redlessone/DermLIP_ViT-B-16 \
        --head /content/skinshield_out/probe_head.pt \
        --out /content/skinshield_export

OUTPUT (--out):
    image_branch.onnx, text_branch.onnx, head.onnx
    class_names.json
    calibration/  (N random training images for optional INT8 calibration)
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_dermlip import CLASS_NAMES, MLPHead, fused_features, load_model


class ImageEncoder(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, image):
        return self.clip.encode_image(image)


class TextEncoder(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, tokens):
        return self.clip.encode_text(tokens)


class FusedInference(nn.Module):
    """Full inference graph: image + text tokens -> logits (matches training)."""

    def __init__(self, clip_model, head):
        super().__init__()
        self.clip = clip_model
        self.head = head

    def forward(self, image, text_tokens):
        img_feat = self.clip.encode_image(image)
        txt_feat = self.clip.encode_text(text_tokens)
        fused = fused_features(img_feat, txt_feat)
        return self.head(fused)


def export_onnx(model, head, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    clip = model.float().to(device).eval()
    head = head.float().to(device).eval()

    # --- image branch -------------------------------------------------------
    img_enc = ImageEncoder(clip)
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        img_enc, (dummy_img,),
        os.path.join(out_dir, "image_branch.onnx"),
        input_names=["image"], output_names=["img_feat"],
        opset_version=17,
    )
    print("[export] image_branch.onnx OK")

    # --- text branch ---------------------------------------------------------
    txt_enc = TextEncoder(clip)
    dummy_tokens = torch.randint(0, 49408, (1, 77), dtype=torch.long, device=device)
    torch.onnx.export(
        txt_enc, (dummy_tokens,),
        os.path.join(out_dir, "text_branch.onnx"),
        input_names=["tokens"], output_names=["txt_feat"],
        opset_version=17,
    )
    print("[export] text_branch.onnx OK")

    # --- head ----------------------------------------------------------------
    dummy_fused = torch.randn(1, 2048, device=device)
    torch.onnx.export(
        head, (dummy_fused,),
        os.path.join(out_dir, "head.onnx"),
        input_names=["fused"], output_names=["logits"],
        opset_version=17,
    )
    print("[export] head.onnx OK")

    # --- class names ----------------------------------------------------------
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print("[export] class_names.json OK")


def export_calibration(data_dir, out_dir, n=64, seed=42):
    """Copy n random training images for optional INT8 calibration."""
    random.seed(seed)
    train_csv = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_csv):
        print("[export] no train.csv found — skipping calibration images")
        return
    import pandas as pd
    df = pd.read_csv(train_csv)
    cal_dir = os.path.join(out_dir, "calibration")
    os.makedirs(cal_dir, exist_ok=True)
    sample = df.sample(min(n, len(df)), random_state=seed)
    copied = 0
    for _, row in sample.iterrows():
        src = os.path.join(data_dir, "images", "train", row["label"],
                           os.path.basename(row["filename"]))
        if os.path.exists(src):
            shutil.copy(src, os.path.join(cal_dir, f"cal_{copied:04d}.jpg"))
            copied += 1
    print(f"[export] calibration images: {copied}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SkinShield model to ONNX")
    parser.add_argument("--data", default="/content/curated")
    parser.add_argument("--model", default="redlessone/DermLIP_ViT-B-16")
    parser.add_argument("--head", default="/content/skinshield_out/probe_head.pt")
    parser.add_argument("--out", default="/content/skinshield_export")
    parser.add_argument("--calib-images", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("SkinShield — ONNX export")
    print("=" * 70)

    clip, _, _, _ = load_model(args.model)
    head = MLPHead(2048, n_classes=len(CLASS_NAMES))
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    print(f"[export] head loaded from {args.head}")

    export_onnx(clip, head, args.out, device)
    export_calibration(args.data, args.out, args.calib_images)

    print("\nDONE. ONNX files in:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())