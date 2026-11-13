import os, json, time, traceback
from collections import Counter
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn as nn

MF   = r"E:\SkinShield\manifest.csv"
OUT  = r"E:\SkinShield\stage2"
QOUT = r"E:\SkinShield\queue"
os.makedirs(QOUT, exist_ok=True)
TOK  = os.environ.get("HF_TOKEN")
NCLASS, CAP = 50, 2000

def log(m):
    print(m, flush=True)
    with open(os.path.join(QOUT, "queue_log.txt"), "a", encoding="utf-8") as f:
        f.write(m + "\n")

results = {}
def save():
    json.dump(results, open(os.path.join(QOUT, "queue_results.json"), "w"), indent=2)

# ============================== COMMON DATA ==============================
log("=" * 72)
log("  RESEARCH QUEUE  —  steps 2,3,4  (sequential, unattended)")
log("=" * 72)

df = pd.read_csv(MF, low_memory=False)
H = df["hierarchical_disease_label"].astype(str)
df["condition"] = H.str.split(",").str[-1].str.strip()
df["top_cat"]   = H.str.split(",").str[0].str.strip()
df["is_infectious"] = H.str.contains("infectious") & ~H.str.contains("non-infectious")
d = df[(~df["is_infectious"]) & (df["condition"] != "no definitive diagnosis")].copy()
top = list(d["condition"].value_counts().head(NCLASS).index)
d = d[d["condition"].isin(top)].copy()
d = d.groupby("condition", group_keys=False).head(CAP).reset_index(drop=True)
classes = sorted(d["condition"].unique())
d["y"] = d["condition"].map({c: i for i, c in enumerate(classes)})
idx_tr, idx_te = train_test_split(d.index, test_size=0.2, stratify=d["y"], random_state=42)
y = d["y"].values
cats = sorted(d["top_cat"].unique())
d["yc"] = d["top_cat"].map({c: i for i, c in enumerate(cats)})
yc = d["yc"].values
log(f"[data] rows={len(d)} classes={len(classes)} categories={len(cats)}")

img  = np.load(os.path.join(OUT, "img.npy"))
tcln = np.load(os.path.join(OUT, "txt_clean.npy"))
norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
img, tcln = norm(img), norm(tcln)

age = pd.to_numeric(d["age"], errors="coerce").fillna(-1).values.reshape(-1, 1)
g   = pd.get_dummies(d["gender"].astype(str)).values.astype(np.float32)
bl  = d["body_location"].astype(str); tb = bl.value_counts().head(30).index
bl  = bl.where(bl.isin(tb), "other"); blf = pd.get_dummies(bl).values.astype(np.float32)
sc  = d["skin_concept"].astype(str); cnt = Counter()
for v in sc:
    for t in v.split(","):
        if t.strip() and t.strip() != "No visual concepts": cnt[t.strip()] += 1
topc = [t for t, _ in cnt.most_common(60)]; ix = {t: i for i, t in enumerate(topc)}
M = np.zeros((len(d), len(topc)), dtype=np.float32)
for i, v in enumerate(sc):
    for t in v.split(","):
        t = t.strip()
        if t in ix: M[i, ix[t]] = 1
stf = pd.get_dummies(d["source_type"].astype(str)).values.astype(np.float32)
meta = np.hstack([age, g, blf, M, stf]).astype(np.float32)
X = np.hstack([img, tcln, meta])
log(f"[data] feature dims: img 512 + txt 512 + meta {meta.shape[1]} = {X.shape[1]}")

# ============================== STEP 2 : HIERARCHICAL MULTI-TASK ==============================
try:
    t0 = time.time()
    log("\n" + "=" * 72)
    log("  STEP 2 — Ontology-guided hierarchical multi-task (condition + category)")
    log("=" * 72)
    torch.manual_seed(42)
    sc_ = StandardScaler().fit(X[idx_tr])
    Xtr = torch.tensor(sc_.transform(X[idx_tr]), dtype=torch.float32)
    Xte = torch.tensor(sc_.transform(X[idx_te]), dtype=torch.float32)
    ytr = torch.tensor(y[idx_tr]); yctr = torch.tensor(yc[idx_tr])

    trunk = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(1024, 512), nn.ReLU())
    head_c = nn.Linear(512, len(classes))
    head_k = nn.Linear(512, len(cats))
    opt = torch.optim.Adam(list(trunk.parameters()) + list(head_c.parameters()) + list(head_k.parameters()), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    for ep in range(30):
        trunk.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            b = perm[i:i+512]
            opt.zero_grad()
            h = trunk(Xtr[b])
            (lf(head_c(h), ytr[b]) + 0.5 * lf(head_k(h), yctr[b])).backward()
            opt.step()
    trunk.eval()
    with torch.no_grad():
        p = head_c(trunk(Xte)).argmax(1).numpy()
        pk = head_k(trunk(Xte)).argmax(1).numpy()
    r = dict(acc=float(accuracy_score(y[idx_te], p)),
             bal_acc=float(balanced_accuracy_score(y[idx_te], p)),
             macro_f1=float(f1_score(y[idx_te], p, average="macro")),
             cat_acc=float(accuracy_score(yc[idx_te], pk)))
    results["step2_hierarchical"] = r
    log(f"  -> condition macro_f1 = {r['macro_f1']:.4f} | acc = {r['acc']:.4f} | category acc = {r['cat_acc']:.4f}  ({time.time()-t0:.0f}s)")
    save()
    X_nn_te = Xte; X_nn_tr = Xtr   # keep for ensembling
except Exception as e:
    log("STEP 2 FAILED: " + repr(e)); traceback.print_exc(); X_nn_tr = X_nn_te = None

# ============================== STEP 3 : kNN + ENSEMBLE ==============================
try:
    t0 = time.time()
    log("\n" + "=" * 72)
    log("  STEP 3 — kNN / retrieval classifier + soft-voting ensemble")
    log("=" * 72)
    sc2 = StandardScaler().fit(X[idx_tr])
    Xtr_s, Xte_s = sc2.transform(X[idx_tr]), sc2.transform(X[idx_te])

    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine", n_jobs=-1)
    knn.fit(Xtr_s, y[idx_tr])
    pknn = knn.predict(Xte_s)
    rk = dict(acc=float(accuracy_score(y[idx_te], pknn)),
              bal_acc=float(balanced_accuracy_score(y[idx_te], pknn)),
              macro_f1=float(f1_score(y[idx_te], pknn, average="macro")))
    log(f"  kNN (k=15, cosine): macro_f1 = {rk['macro_f1']:.4f}")
    results["step3_knn"] = rk

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(X[idx_tr], y[idx_tr]); plr = lr.predict_proba(X[idx_te])
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=120, early_stopping=True, random_state=42))
    mlp.fit(X[idx_tr], y[idx_tr]); pmlp = mlp.predict_proba(X[idx_te])
    hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, random_state=42)
    hgb.fit(X[idx_tr], y[idx_tr]); phgb = hgb.predict_proba(X[idx_te])
    proba = (plr + pmlp + phgb) / 3.0
    pe = proba.argmax(1)
    re = dict(acc=float(accuracy_score(y[idx_te], pe)),
              bal_acc=float(balanced_accuracy_score(y[idx_te], pe)),
              macro_f1=float(f1_score(y[idx_te], pe, average="macro")))
    log(f"  ENSEMBLE (LR+MLP+HistGBM): macro_f1 = {re['macro_f1']:.4f}  ({time.time()-t0:.0f}s)")
    results["step3_ensemble"] = re
    save()
except Exception as e:
    log("STEP 3 FAILED: " + repr(e)); traceback.print_exc()

# ============================== STEP 4 : TEST-TIME AUGMENTATION ==============================
try:
    t0 = time.time()
    log("\n" + "=" * 72)
    log("  STEP 4 — Test-time augmentation (horizontal flip) + retrain")
    log("=" * 72)
    fp = os.path.join(OUT, "img_flip.npy")
    if os.path.exists(fp):
        imgf = np.load(fp); log("[tta] flip features loaded from cache")
    else:
        model, _, transform = open_clip.create_model_and_transforms("hf-hub:redlessone/DermLIP_ViT-B-16")
        model.eval()
        paths = d["img_path"].tolist()
        imgf = np.zeros((len(paths), 512), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm(range(0, len(paths), 16), desc="flip"):
                b = []
                for pth in paths[i:i+16]:
                    try:
                        im = transform(Image.open(pth).convert("RGB"))
                        b.append(torch.flip(im, dims=[2]))       # horizontal flip
                    except Exception:
                        b.append(torch.zeros(3, 224, 224))
                imgf[i:i+len(b)] = model.encode_image(torch.stack(b)).float().numpy()
        np.save(fp, imgf); log(f"[tta] flip features extracted in {time.time()-t0:.0f}s")
    imgf = norm(imgf)
    img_tta = norm(img + imgf)                    # averaged embedding
    X_tta = np.hstack([img_tta, tcln, meta])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(X_tta[idx_tr], y[idx_tr]); p = lr.predict(X_tta[idx_te])
    rt = dict(acc=float(accuracy_score(y[idx_te], p)),
              bal_acc=float(balanced_accuracy_score(y[idx_te], p)),
              macro_f1=float(f1_score(y[idx_te], p, average="macro")))
    log(f"  TTA tri-modal: macro_f1 = {rt['macro_f1']:.4f}  (total {time.time()-t0:.0f}s)")
    results["step4_tta_trimodal"] = rt
    save()
except Exception as e:
    log("STEP 4 FAILED: " + repr(e)); traceback.print_exc()

# ============================== FINAL SUMMARY ==============================
log("\n" + "=" * 72)
log("  FINAL QUEUE SUMMARY  (50-class test set, macro_F1)")
log("=" * 72)
for k, v in results.items():
    log(f"  {k:26s} macro_f1={v.get('macro_f1', float('nan')):.4f}  acc={v.get('acc', float('nan')):.4f}")
log("\nALL DONE -> " + os.path.join(QOUT, "queue_results.json"))
log("=" * 72)
