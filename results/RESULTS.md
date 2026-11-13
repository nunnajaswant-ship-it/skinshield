# Results

All figures below come from actual runs on the project workstation (CPU-only, Intel i9-13900). Test splits are stratified 80/20 with a fixed seed (42), so the numbers are reproducible.

---

## Stage 1 — Contagion gate (infectious vs non-infectious)

**Data:** 137,788 cases — 27,964 infectious, 109,824 non-infectious. Train 110,230 / test 27,558.

| Model | Accuracy | Balanced accuracy | F1 |
|---|---|---|---|
| image_only | 0.7613 | 0.7590 | 0.5622 |
| text_only | 0.9292 | 0.9312 | 0.8428 |
| **fused** | **0.9320** | **0.9317** | **0.8475** |

**Confusion matrix (fused)**

```
                    non-infectious   infectious
non-infectious           20,474          1,491
infectious                  384          5,209
```

| Metric | Value |
|---|---|
| Infection sensitivity | **93.1%** (5,209 / 5,593) |
| Specificity | 93.2% |
| Missed infections | 384 |

**Per-pathogen recall (fused)**

| Pathogen | Test n | Recall |
|---|---|---|
| bacterial | 2,135 | 0.939 |
| fungal | 1,423 | 0.915 |
| viral | 1,413 | 0.946 |
| parasitic | 605 | 0.917 |

---

## Stage 2 — Disease identification (50 conditions)

**Data:** 69,879 cases across 50 non-contagious conditions, capped at 2,000 per class.

| Configuration | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| image_only | 0.4404 | 0.4658 | 0.4311 |
| text_deleaked | 0.6879 | 0.7046 | 0.6814 |
| fused (img + txt) | 0.7095 | 0.7115 | 0.7045 |
| meta_only | 0.2966 | 0.3031 | 0.2800 |
| tri-modal (img + txt + metadata) | 0.7483 | 0.7500 | 0.7461 |
| tri-modal + MLP | 0.8092 | 0.8043 | 0.8083 |
| **tri-modal + HistGBM** | **0.8090** | 0.7943 | **0.8102** |

**Hierarchical / ensemble**

| Model | Macro-F1 |
|---|---|
| hierarchical multi-task (category + condition) | 0.8337 |
| ensemble (LR + MLP + HistGBM) | **0.8338** |
| kNN (k=15, cosine) | 0.7415 |

---

## Caption-leakage investigation

**53.0% of clinical captions contained the diagnosis** (e.g. *"this image shows melanoma"*). Measured per condition:

| Condition | Captions naming the diagnosis |
|---|---|
| melanoma | 92% |
| tinea | 92% |
| basal cell carcinoma | 91% |
| urticaria | 91% |
| eczema | 90% |
| allergic contact dermatitis | 86% |
| acne | 83% |
| squamous cell carcinoma | 75% |
| nevus | 72% |
| psoriasis | 56% |

**Leakage-controlled re-evaluation**

| Configuration | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| image_only | 0.4404 | 0.4658 | 0.4311 |
| text_raw (leaky) | 0.8734 | 0.8785 | 0.8736 |
| text_deleaked | 0.6879 | 0.7046 | 0.6814 |
| fused_raw (leaky) | 0.8503 | 0.8551 | 0.8553 |
| **fused_deleaked (honest)** | **0.7095** | **0.7115** | **0.7045** |

The leak was worth roughly **0.19 macro-F1**. With the leak removed, **fusion beats both single modalities** — the core scientific claim of the project.

---

## Zero-shot breadth

**150 conditions, 6,000 images, no training.**

| Metric | Value |
|---|---|
| Top-1 accuracy | **0.2198** |
| Top-5 accuracy | **0.4447** |
| Random chance (1/150) | 0.0067 |

**Best conditions (top-1):** pityriasis rosea 0.78, cheilitis 0.68, condyloma 0.65, limbal dermoid 0.62, alopecia 0.60.

**Hardest:** dermatitis, contact dermatitis, keratosis, deep fungal infection — broad categories that are inherently ambiguous as text prompts.

---

## Evaluation protocol

- **Split:** stratified 80/20 with `random_state=42`.
- **Metrics:** balanced accuracy and macro-F1 (class-balanced), plus raw accuracy.
- **Leakage control:** each caption's own diagnosis name removed before text encoding.
- **Reproducibility:** all scripts in `codes/` regenerate these numbers from the curated dataset.
