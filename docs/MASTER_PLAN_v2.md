# SkinShield v2 — Full-Scope Master Plan
**Scope change:** 12 classes → whole Derm1M ontology, two-stage safety-first design
**Date:** 2026-09-29 · **Status:** planning · **Compute:** college PC

---

## 1. The new product goal

A two-stage, safety-first dermatology assistant over the **entire** Derm1M ontology:

```
                 ┌──────────────────────────┐
   image + text  │  STAGE 1: CONTAGION GATE │
   ─────────────▶│  infectious vs not       │
                 └───────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
      INFECTIOUS (≈contagious)      NON-INFECTIOUS
              │                             │
              ▼                             ▼
   ┌─────────────────────┐      ┌──────────────────────────┐
   │ "Likely infectious │      │ STAGE 2: DISEASE ID      │
   │  — see a doctor,   │      │ multi-class over          │
   │  avoid contact"    │      │ well-populated conditions │
   └─────────────────────┘      └──────────────────────────┘
                                          │
                                     (long tail → zero-shot)
```

**Safety principle (kept from v1):** a contagious/infectious result never implies "safe" — it routes to *seek care*.

---

## 2. Why this is now possible (the data gives us the labels)

Derm1M's `hierarchical_disease_label` is a structured ontology. **Verified on the real pretrain CSV:**

| Axis | Values | Rows |
|---|---|---|
| Infection | `infectious` | **27,964** |
| | `non-infectious` | **109,824** |
| Infectious subtype | bacterial / fungal / viral / parasitic | 10,679 / 7,267 / 6,869 / 3,035 |
| Malignancy | `malignant` / `benign` | **21,413 / 42,825** |
| Top categories | inflammatory / proliferations / reaction patterns / hereditary / exogenous / nevus / keratosis / nail / hair / vascular / acne / systemic / scleroderma | — |

**Consequence:** the contagion gate's labels come **directly from the dataset** — no hand-labelling of ~390 conditions. That is the single biggest de-risking fact in this plan.

> ⚠️ **One nuance to decide:** `infectious` (caused by a pathogen) ≠ `contagious` (transmissible person-to-person). E.g. *cellulitis/folliculitis* are infectious but not very contagious; *tinea, impetigo, scabies, molluscum, herpes* are contagious. Plan: use `infectious` as the gate, then derive a **professor-reviewed "clearly contagious" sub-flag** from the bacterial/fungal/viral/parasitic leaves. Two safety tiers, not one.

---

## 3. The long-tail reality (what we can and cannot train)

| Conditions with ≥ | Count |
|---|---|
| 10,000 | 6 |
| 1,000 | 40 |
| 500 | 80 |
| 300 | 114 |
| 100 | 213 |
| 50 | 344 |
| 20 | 685 |
| < 5 pairs | 17,027 |

**Decision:** train on the **head**, zero-shot the **tail**.
- **Stage 1 (contagion):** strong — ~138k usable rows.
- **Stage 2 (disease ID):** train over **top-N** (recommend **N = 213 at ≥100 pairs**, or 80 at ≥500 for cleaner metrics).
- **Tail (<100 pairs):** **zero-shot** via DermLIP text prompts (no training possible). Reported as "covered, lower confidence."

**No fake numbers:** we do **not** claim a 390-way trained classifier. We claim: binary gate + N-way trained head + zero-shot breadth, each measured on its supported set.

---

## 4. Architecture (carried over from v1)

- **Backbone:** DermLIP ViT-B-16 (**frozen**) — already pretrained on all 403,563 Derm1M pairs, so it already "knows" every condition.
- **Features:** image (512-d) + text (512-d) → fused (2,048-d).
- **Stage 1 head:** MLP, binary (infectious / non-infectious).
- **Stage 2 head:** MLP, N-way over non-infectious conditions.
- **Zero-shot path:** cosine similarity to condition prompts for the tail.
- **Everything reuses the v1 code** (`train_dermlip.py`, `export_onnx.py`) with the class list swapped.

---

## 5. Phases

| # | Phase | Output | Gate to pass |
|---|---|---|---|
| 0 | **College PC probe + env setup** | specs report; Python + CUDA torch + libs installed | `torch.cuda.is_available() == True`, DermLIP loads |
| 1 | **Full dataset download** | all Derm1M zips (~34.5 GB) + extract | all splits present, image count verified |
| 2 | **Curation v2 (ontology labels)** | per-row labels: infectious/malignant/condition + per-class counts | label counts reconcile with ontology |
| 3 | **Stage 1 — contagion gate** | trained binary model + metrics | balanced accuracy + per-subtype recall |
| 4 | **Stage 2 — disease ID** | N-way model + metrics | macro-F1 on supported set |
| 5 | **Zero-shot breadth** | all-conditions zero-shot report | coverage % of ontology reported |
| 6 | **Evaluation + report** | ablation, confusion, per-class, professor report | fused ≥ best single modality |
| 7 | **Deploy** | ONNX → Jetson TensorRT, latency | median < 100 ms |

---

## 6. Decisions needed (blockers in bold)

1. **College PC specs** — run the probe script. GPU/VRAM decides batch size and whether we fine-tune the backbone. **← BLOCKER**
2. **Stage-2 breadth `N`** — 80 (≥500 pairs, clean) vs 213 (≥100 pairs, broader). Recommend **213**.
3. **Contagion definition** — accept `infectious` as the gate + add a "clearly contagious" sub-tier? (recommended)
4. **Professor sign-off** — the 12-class deliverable becomes the "well-populated core" of this; confirm the committee accepts the expanded scope.

---

## 7. What carries over (v1 is not wasted)

- `train_dermlip.py`, `export_onnx.py`, probe/ablation pipeline — **reused as-is**
- The 12 classes become a subset of the new ontology (all 5 common + 7 rare are in the ontology)
- Curation code (`curate_derm1m.py`) — extended, not rewritten
- The frozen-DermLIP + probe architecture is unchanged

---

## 8. Immediate next step

**Run the college-PC probe** (script already on your clipboard / at `dist\college_pc_remote_setup.ps1`), then send the output. Everything else keys off that report.
