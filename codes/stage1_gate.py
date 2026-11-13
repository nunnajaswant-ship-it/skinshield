import os, time, json
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

MF  = r"E:\SkinShield\manifest.csv"
OUT = r"E:\SkinShield\gate"
os.makedirs(OUT, exist_ok=True)
TOK = os.environ.get("HF_TOKEN")

print("==== Stage 1: contagion gate (infectious vs non-infectious) ====", flush=True)

# ---- 1. build the gate set: explicit infectious vs explicit non-infectious ----
df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["y"] = np.where(H.str.contains("infectious") & ~H.str.contains("non-infectious"), 1,
          np.where(H.str.contains("non-infectious"), 0, -1))
g = df[df["y"] >= 0].reset_index(drop=True)
g["subtype"] = g["hierarchical_disease_label"].astype(str).str.extract(r"infectious,\s*([^,]+)")[0].fillna("non-infectious")
print(f"[gate] rows={len(g)}  infectious={int((g.y==1).sum())}  non={int((g.y==0).sum())}", flush=True)

idx_tr, idx_te = train_test_split(g.index, test_size=0.2, stratify=g["y"], random_state=42)
print(f"[gate] train={len(idx_tr)}  test={len(idx_te)}", flush=True)

# ---- 2. features (cached) ----
ip, tp = os.path.join(OUT, "img.npy"), os.path.join(OUT, "txt.npy")
if os.path.exists(ip) and os.path.exists(tp):
    img, txt = np.load(ip), np.load(tp)
    print("[features] loaded from cache", flush=True)
else:
    print("[model] loading DermLIP (downloads ~600MB first time) ...", flush=True)
    model, _, transform = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_ViT-B-16")
    tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermLIP_ViT-B-16")
    model.eval()
    t0 = time.time()

    paths = g["img_path"].tolist()
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

    texts = g["truncated_caption"].fillna("skin image").astype(str).tolist()
    txt = np.zeros((len(texts), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            txt[i:i+64] = model.encode_text(tokenizer(texts[i:i+64])).float().numpy()
    np.save(tp, txt)
    print(f"[features] done in {time.time()-t0:.0f}s", flush=True)

img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
fused = np.concatenate([img, txt], axis=1)
y = g["y"].values

# ---- 3. heads + evaluation ----
print("\n=== STAGE 1 RESULTS (test set) ===", flush=True)
res = {}
for name, X in [("image_only", img), ("text_only", txt), ("fused", fused)]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict(X[idx_te])
    res[name] = dict(acc=float(accuracy_score(y[idx_te], p)),
                     bal_acc=float(balanced_accuracy_score(y[idx_te], p)),
                     f1=float(f1_score(y[idx_te], p)))
    print(f"  {name:12s} acc={res[name]['acc']:.4f}  bal_acc={res[name]['bal_acc']:.4f}  f1={res[name]['f1']:.4f}", flush=True)

# per-subtype recall for the infectious class (fused)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(fused[idx_tr], y[idx_tr])
pte = clf.predict(fused[idx_te])
sub = g.iloc[idx_te]["subtype"].values
yte = y[idx_te]
print("\n  infectious recall by subtype (fused):", flush=True)
for s in sorted(set(sub)):
    mask = sub == s
    if s != "non-infectious" and mask.sum() > 0:
        print(f"    {s:10s} n={int(mask.sum()):6d}  recall={recall_score(yte[mask], pte[mask], zero_division=0):.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, "gate_results.json"), "w"), indent=2)
print("\nDONE. ->", os.path.join(OUT, "gate_results.json"), flush=True)
