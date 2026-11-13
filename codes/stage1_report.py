import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split

MF  = r"E:\SkinShield\manifest.csv"
OUT = r"E:\SkinShield\gate"

print("=" * 66)
print("  STAGE 1 — CONTAGION GATE  (clean report, cached features)")
print("=" * 66)

# --- rebuild the SAME gate set + split (same seed => identical split) ---
df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["y"] = np.where(H.str.contains("infectious") & ~H.str.contains("non-infectious"), 1,
          np.where(H.str.contains("non-infectious"), 0, -1))
g = df[df["y"] >= 0].reset_index(drop=True)

# FIXED subtype: only infectious rows get a pathogen subtype
g["subtype"] = "non-infectious"
inf = g["y"] == 1
ext = g.loc[inf, "hierarchical_disease_label"].astype(str).str.extract(r"infectious,\s*([^,]+)")[0]
g.loc[inf, "subtype"] = ext.values

idx_tr, idx_te = train_test_split(g.index, test_size=0.2, stratify=g["y"], random_state=42)
y = g["y"].values

# --- cached features (instant) ---
img = np.load(os.path.join(OUT, "img.npy"))
txt = np.load(os.path.join(OUT, "txt.npy"))
img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8)
fused = np.concatenate([img, txt], axis=1)

print(f"\n  gate set : {len(g)} rows  (infectious {int((g.y==1).sum())} / non-infectious {int((g.y==0).sum())})")
print(f"  split    : train {len(idx_tr)}  |  test {len(idx_te)}\n")

# --- ablation ---
results = {}
print("  MODEL          acc      bal_acc   f1")
print("  " + "-" * 42)
for name, X in [("image_only", img), ("text_only", txt), ("fused", fused)]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[idx_tr], y[idx_tr])
    p = clf.predict(X[idx_te])
    results[name] = dict(accuracy=float(accuracy_score(y[idx_te], p)),
                         balanced_accuracy=float(balanced_accuracy_score(y[idx_te], p)),
                         macro_f1=float(f1_score(y[idx_te], p)))
    r = results[name]
    print(f"  {name:12s}  {r['accuracy']:.4f}   {r['balanced_accuracy']:.4f}    {r['macro_f1']:.4f}")

# --- fused confusion matrix ---
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(fused[idx_tr], y[idx_tr])
pte = clf.predict(fused[idx_te])
yte = y[idx_te]
cm = confusion_matrix(yte, pte)
print("\n  Confusion matrix (fused)  [rows=true, cols=pred]")
print("                    non-inf   infectious")
print(f"    non-infectious   {cm[0,0]:6d}     {cm[0,1]:6d}")
print(f"    infectious       {cm[1,0]:6d}     {cm[1,1]:6d}")

# --- CLEAN per-pathogen recall (only infectious subtypes) ---
sub = g.iloc[idx_te]["subtype"].values
print("\n  Infectious detection recall by pathogen (fused):")
subres = {}
for s in ["bacterial", "fungal", "viral", "parasitic"]:
    m = sub == s
    if m.sum() > 0:
        rec = recall_score(yte[m], pte[m], zero_division=0)
        subres[s] = dict(n=int(m.sum()), recall=float(rec))
        print(f"    {s:10s} n={int(m.sum()):5d}   recall={rec:.3f}")

print("\n  DONE -> clean results saved to gate_results_clean.json")

json.dump({"models": results, "subtype_recall": subres,
           "gate_rows": len(g), "infectious": int((g.y == 1).sum()),
           "non_infectious": int((g.y == 0).sum()), "test_rows": len(idx_te)},
          open(os.path.join(OUT, "gate_results_clean.json"), "w"), indent=2)
print("=" * 66)
