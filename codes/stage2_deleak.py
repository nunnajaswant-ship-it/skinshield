import os, re, json
import numpy as np, pandas as pd, torch, open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MF   = r"E:\SkinShield\manifest.csv"
OUT  = r"E:\SkinShield\stage2"
TOK  = os.environ.get("HF_TOKEN")
NCLASS, CAP = 50, 2000

print("=" * 70)
print("  DE-LEAK RE-EVALUATION  (remove diagnosis from captions)")
print("=" * 70, flush=True)

# ---- 1. rebuild the SAME stage-2 set (identical seed -> identical split) ----
df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["condition"]     = H.str.split(",").str[-1].str.strip()
df["is_infectious"] = H.str.contains("infectious") & ~H.str.contains("non-infectious")

d = df[(~df["is_infectious"]) & (df["condition"] != "no definitive diagnosis")].copy()
top = list(d["condition"].value_counts().head(NCLASS).index)
d = d[d["condition"].isin(top)].copy()
d = d.groupby("condition", group_keys=False).head(CAP).reset_index(drop=True)
classes = sorted(d["condition"].unique())
d["y"] = d["condition"].map({c: i for i, c in enumerate(classes)})
idx_tr, idx_te = train_test_split(d.index, test_size=0.2, stratify=d["y"], random_state=42)
y = d["y"].values
print(f"[data] rows={len(d)} classes={len(classes)}", flush=True)

# ---- 2. de-leak the captions ----
STOP = {"cell", "and", "the", "disease", "other", "disorder", "type", "types"}
def deleak(text, cond):
    t = str(text)
    words = [w for w in re.split(r"\W+", cond.lower()) if w]
    # full condition phrase (loose separators)
    if words:
        pat = r"\b" + r"[\W_]{0,4}".join(re.escape(w) for w in words) + r"\b"
        t = re.sub(pat, " ", t, flags=re.I)
    # distinctive individual words
    for w in words:
        if len(w) >= 5 and w not in STOP:
            t = re.sub(r"\b" + re.escape(w) + r"\w*", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) > 3 else "skin image"

d["clean_caption"] = [deleak(c, k) for c, k in zip(d["truncated_caption"].fillna(""), d["condition"])]
print("[deleak] example (before -> after):", flush=True)
for i in range(3):
    print("   BEFORE:", str(d["truncated_caption"].iloc[i])[:110], flush=True)
    print("   AFTER :", d["clean_caption"].iloc[i][:110], flush=True)

# ---- 3. features: reuse cached image; re-encode CLEAN text ----
img  = np.load(os.path.join(OUT, "img.npy"))
tra  = np.load(os.path.join(OUT, "txt.npy"))            # raw (leaky) text features
print("[model] loading DermLIP to encode clean captions ...", flush=True)
model, _, _ = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_ViT-B-16")
tokenizer = open_clip.get_tokenizer("hf-hub:redlessone/DermLIP_ViT-B-16")
model.eval()
texts = d["clean_caption"].tolist()
tcln = np.zeros((len(texts), 512), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(texts), 128):
        tcln[i:i+128] = model.encode_text(tokenizer(texts[i:i+128])).float().numpy()
np.save(os.path.join(OUT, "txt_clean.npy"), tcln)

norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
img, tra, tcln = norm(img), norm(tra), norm(tcln)

# ---- 4. compare 5 configurations ----
print("\n=== LEAKAGE IMPACT (identical 50-class test set) ===", flush=True)
print(f"  {'config':22s} {'acc':>8s} {'bal_acc':>9s} {'macro_f1':>9s}", flush=True)
print("  " + "-" * 52, flush=True)
res = {}
for name, X in [("image_only",          img),
                ("text_raw (LEAKY)",     tra),
                ("text_deleaked",        tcln),
                ("fused_raw (LEAKY)",    np.concatenate([img, tra],  axis=1)),
                ("fused_deleaked",       np.concatenate([img, tcln], axis=1))]:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict(X[idx_te])
    res[name] = dict(acc=float(accuracy_score(y[idx_te], p)),
                     bal_acc=float(balanced_accuracy_score(y[idx_te], p)),
                     macro_f1=float(f1_score(y[idx_te], p, average="macro")))
    r = res[name]
    print(f"  {name:22s} {r['acc']:8.4f} {r['bal_acc']:9.4f} {r['macro_f1']:9.4f}", flush=True)

json.dump(res, open(os.path.join(OUT, "deleak_results.json"), "w"), indent=2)
print("\nDONE ->", os.path.join(OUT, "deleak_results.json"), flush=True)
print("=" * 70)
