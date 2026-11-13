import os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MF  = r"E:\SkinShield\manifest.csv"
OUT = r"E:\SkinShield\stage2"
NCLASS, CAP = 50, 2000

print("=" * 70)
print("  STAGE 2+ : TRI-MODAL (image + text + clinical metadata) + strong heads")
print("=" * 70, flush=True)

# ---- rebuild SAME stage-2 set (identical split) ----
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

# ---- cached deep features ----
img  = np.load(os.path.join(OUT, "img.npy"))
tcln = np.load(os.path.join(OUT, "txt_clean.npy"))     # de-leaked text
norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
img, tcln = norm(img), norm(tcln)

# ---- NEW: structured clinical metadata ----
print("[meta] building clinical-metadata features ...", flush=True)
age = pd.to_numeric(d["age"], errors="coerce").fillna(-1).values.reshape(-1, 1)
g   = pd.get_dummies(d["gender"].astype(str)).values.astype(np.float32)
bl  = d["body_location"].astype(str)
topbl = bl.value_counts().head(30).index
bl  = bl.where(bl.isin(topbl), "other")
blf = pd.get_dummies(bl).values.astype(np.float32)
sc  = d["skin_concept"].astype(str)
cnt = Counter()
for v in sc:
    for t in v.split(","):
        if t.strip() and t.strip() != "No visual concepts":
            cnt[t.strip()] += 1
topc = [t for t, _ in cnt.most_common(60)]
idx = {t: i for i, t in enumerate(topc)}
M = np.zeros((len(d), len(topc)), dtype=np.float32)
for i, v in enumerate(sc):
    for t in v.split(","):
        t = t.strip()
        if t in idx:
            M[i, idx[t]] = 1
stf = pd.get_dummies(d["source_type"].astype(str)).values.astype(np.float32)
meta = np.hstack([age, g, blf, M, stf]).astype(np.float32)
print(f"[meta] meta feature dim = {meta.shape[1]}  (age 1 + gender + location 30 + concepts 60 + source)", flush=True)

# ---- heads ----
def train_eval(name, X, head):
    if head == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    elif head == "mlp":
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=120,
                                          early_stopping=True, random_state=42))
    else:
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, random_state=42)
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict(X[idx_te])
    return dict(acc=float(accuracy_score(y[idx_te], p)),
                bal_acc=float(balanced_accuracy_score(y[idx_te], p)),
                macro_f1=float(f1_score(y[idx_te], p, average="macro")))

configs = [
    ("image_only",            img,                                            "lr"),
    ("text_deleaked",         tcln,                                           "lr"),
    ("fused (img+txt)",       np.hstack([img, tcln]),                         "lr"),
    ("meta_only",             meta,                                           "lr"),
    ("TRI-MODAL (i+t+m)",     np.hstack([img, tcln, meta]),                   "lr"),
    ("TRI-MODAL + MLP",       np.hstack([img, tcln, meta]),                   "mlp"),
    ("TRI-MODAL + HistGBM",   np.hstack([img, tcln, meta]),                   "hgb"),
]
print(f"\n=== STAGE 2+ RESULTS (identical 50-class test set) ===", flush=True)
print(f"  {'configuration':24s} {'acc':>8s} {'bal_acc':>9s} {'macro_f1':>9s}", flush=True)
print("  " + "-" * 54, flush=True)
res = {}
for name, X, head in configs:
    r = train_eval(name, X, head)
    res[name] = r
    print(f"  {name:24s} {r['acc']:8.4f} {r['bal_acc']:9.4f} {r['macro_f1']:9.4f}", flush=True)

json.dump(res, open(os.path.join(OUT, "trimodal_results.json"), "w"), indent=2)
print("\nDONE ->", os.path.join(OUT, "trimodal_results.json"), flush=True)
print("=" * 70)
