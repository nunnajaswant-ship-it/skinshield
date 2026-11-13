import os, json, re
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

MF   = r"E:\SkinShield\manifest.csv"
OUT  = r"E:\SkinShield\zeroshot"
os.makedirs(OUT, exist_ok=True)
TOK  = os.environ.get("HF_TOKEN")
NCLASS, PER = 150, 40

print("=" * 66)
print("  ZERO-SHOT BREADTH across the Derm1M ontology (no training)")
print("=" * 66, flush=True)

df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["condition"] = H.str.split(",").str[-1].str.strip()
df = df[df["condition"] != "no definitive diagnosis"].copy()
df["cname"] = df["condition"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()

top = list(df["condition"].value_counts().head(NCLASS).index)
d = df[df["condition"].isin(top)].groupby("condition", group_keys=False).head(PER).reset_index(drop=True)
classes = sorted(d["condition"].unique())
d["y"] = d["condition"].map({c: i for i, c in enumerate(classes)})
print(f"[zs] conditions={len(classes)}  images={len(d)}", flush=True)

# ---- image features ----
ip = os.path.join(OUT, "img.npy")
model, _, transform = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_ViT-B-16")
tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermLIP_ViT-B-16")
model.eval()

if os.path.exists(ip):
    img = np.load(ip)
    print("[zs] image features loaded from cache", flush=True)
else:
    paths = d["img_path"].tolist()
    img = np.zeros((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), 16), desc="images"):
            b = []
            for p in paths[i:i+16]:
                try:
                    b.append(transform(Image.open(p).convert("RGB")))
                except Exception:
                    b.append(torch.zeros(3, 224, 224))
            img[i:i+len(b)] = model.encode_image(torch.stack(b)).float().numpy()
    np.save(ip, img)
img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)

# ---- text prompts per condition ----
TEMPLATES = ["This is a skin image of {}.",
             "A clinical photo of {}.",
             "{} is shown in this skin image."]
cname = {c: d.loc[d["condition"] == c, "cname"].iloc[0] for c in classes}
txt = np.zeros((len(classes), 512), dtype=np.float32)
with torch.no_grad():
    for i, c in enumerate(classes):
        f = model.encode_text(tokenizer([t.format(cname[c]) for t in TEMPLATES])).float().numpy()
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        txt[i] = f.mean(0)
txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)

# ---- zero-shot ----
logits = img @ txt.T
y = d["y"].values
top1 = logits.argmax(1)
acc1 = accuracy_score(y, top1)
top5 = np.argsort(-logits, axis=1)[:, :5]
acc5 = float(np.mean([y[i] in top5[i] for i in range(len(y))]))

print("\n=== ZERO-SHOT RESULTS (no training, pure image+text prompts) ===", flush=True)
print(f"  conditions : {len(classes)}")
print(f"  images     : {len(d)}")
print(f"  top-1 acc  : {acc1:.4f}")
print(f"  top-5 acc  : {acc5:.4f}")

per_acc = {}
for i, c in enumerate(classes):
    m = y == i
    if m.sum() >= 10:
        per_acc[c] = float((top1[m] == i).mean())
print("\n  best 10 conditions (top-1):", flush=True)
for c, a in sorted(per_acc.items(), key=lambda x: -x[1])[:10]:
    print(f"    {c:42s} {a:.2f}")
print("\n  worst 10 conditions (top-1):", flush=True)
for c, a in sorted(per_acc.items(), key=lambda x: x[1])[:10]:
    print(f"    {c:42s} {a:.2f}")

json.dump({"n_classes": len(classes), "n_images": len(d), "top1": float(acc1),
           "top5": float(acc5), "per_class": per_acc},
          open(os.path.join(OUT, "zeroshot_results.json"), "w"), indent=2)
print("\nDONE ->", os.path.join(OUT, "zeroshot_results.json"), flush=True)
print("=" * 66)
