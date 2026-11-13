import os, time, json
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MF   = r"E:\SkinShield\manifest.csv"
OUT  = r"E:\SkinShield\stage2"
os.makedirs(OUT, exist_ok=True)
TOK  = os.environ.get("HF_TOKEN")
NCLASS, CAP = 50, 2000

print("=" * 66)
print("  STAGE 2 - disease identification (multi-class)")
print("=" * 66, flush=True)

# ---- 1. target set: top-N NON-CONTAGIOUS conditions (exclude infectious) ----
df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["condition"]    = H.str.split(",").str[-1].str.strip()
df["is_infectious"] = H.str.contains("infectious") & ~H.str.contains("non-infectious")

d = df[(~df["is_infectious"]) & (df["condition"] != "no definitive diagnosis")].copy()
vc = d["condition"].value_counts()
top = list(vc.head(NCLASS).index)
d = d[d["condition"].isin(top)].copy()
d = d.groupby("condition", group_keys=False).head(CAP).reset_index(drop=True)

classes = sorted(d["condition"].unique())
c2i = {c: i for i, c in enumerate(classes)}
d["y"] = d["condition"].map(c2i)
print(f"[stage2] rows={len(d)}  classes={len(classes)}", flush=True)
print("[stage2] classes:", ", ".join(classes), flush=True)

idx_tr, idx_te = train_test_split(d.index, test_size=0.2, stratify=d["y"], random_state=42)
print(f"[stage2] train={len(idx_tr)}  test={len(idx_te)}", flush=True)

# ---- 2. features ----
ip, tp = os.path.join(OUT, "img.npy"), os.path.join(OUT, "txt.npy")
if os.path.exists(ip) and os.path.exists(tp):
    img, txt = np.load(ip), np.load(tp)
    print("[features] loaded from cache", flush=True)
else:
    print("[model] loading DermLIP ...", flush=True)
    model, _, transform = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_ViT-B-16")
    tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermLIP_ViT-B-16")
    model.eval()
    t0 = time.time()
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
            if i and i % 1600 == 0:
                np.save(ip, img)
    np.save(ip, img)
    texts = d["truncated_caption"].fillna("skin image").astype(str).tolist()
    txt = np.zeros((len(texts), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            txt[i:i+64] = model.encode_text(tokenizer(texts[i:i+64])).float().numpy()
    np.save(tp, txt)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)

img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
fused = np.concatenate([img, txt], axis=1)
y = d["y"].values

# ---- 3. multi-class heads ----
print("\n=== STAGE 2 RESULTS (test set) ===", flush=True)
res = {}
for name, X in [("image_only", img), ("text_only", txt), ("fused", fused)]:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1))
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict(X[idx_te])
    res[name] = dict(accuracy=float(accuracy_score(y[idx_te], p)),
                     balanced_accuracy=float(balanced_accuracy_score(y[idx_te], p)),
                     macro_f1=float(f1_score(y[idx_te], p, average="macro")))
    r = res[name]
    print(f"  {name:12s} acc={r['accuracy']:.4f}  bal_acc={r['balanced_accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}", flush=True)

# top-10 most-confused classes (fused)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1))
clf.fit(fused[idx_tr], y[idx_tr])
pte = clf.predict(fused[idx_te])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y[idx_te], pte)
print("\n  per-class F1 (fused) - lowest 10:", flush=True)
from sklearn.metrics import f1_score as f1c
per = f1c(y[idx_te], pte, average=None)
order = np.argsort(per)
for i in order[:10]:
    print(f"    {classes[i]:36s} f1={per[i]:.3f}", flush=True)
print("\n  per-class F1 (fused) - highest 5:", flush=True)
for i in order[::-1][:5]:
    print(f"    {classes[i]:36s} f1={per[i]:.3f}", flush=True)

json.dump({"models": res, "n_classes": len(classes), "rows": len(d),
           "classes": classes}, open(os.path.join(OUT, "stage2_results.json"), "w"), indent=2)
print("\nDONE. ->", os.path.join(OUT, "stage2_results.json"), flush=True)
print("=" * 66)
