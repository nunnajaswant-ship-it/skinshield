# SkinShield — Project Progress Report
**Multimodal (image + text) skin disease classification for edge deployment**
Report date: 2026-10-05

---

## 1. Objective

Build a **safety-first, multimodal dermatology classifier** that takes a **skin image + clinical text** and:
1. **Stage 1 — Contagion gate:** decides whether the condition is *infectious/potentially contagious* → routes to *"seek medical attention"*.
2. **Stage 2 — Disease identification:** if non-contagious, names the specific skin disease.
3. Target deployment: **NVIDIA Jetson Orin Nano** (edge), TensorRT.

Originally a 12-class task; **expanded to the full Derm1M ontology** (all ~390 conditions).

---

## 2. Why image + text (and not sensors)

The original concept used **sensor** input. No public sensor dataset exists for skin disease, so the project **pivoted to image + text** — a decision that defines the whole architecture. Datasets without paired text (HAM10000, ISIC, Fitzpatrick17k) are therefore structurally unusable.

---

## 3. Dataset — Derm1M v2

| Property | Value |
|---|---|
| Source | `redlessone/Derm1M` (HuggingFace, gated) |
| License | CC BY-NC 4.0 (non-commercial academic research) |
| Size downloaded | **34.5 GB** (8 source zips + CSVs) |
| Total datapoints | **423,016 image–text pairs** (413,210 pretrain + 9,806 validation) |
| Distinct conditions | ~390 clean ontology conditions |
| Images extracted | **423,016 — 100% match, zero missing** |
| Sources | YouTube, PubMed, IIYI (Chinese clinical), textbooks, public sets, Reddit, Twitter |

**A key discovery:** Derm1M's `hierarchical_disease_label` is a structured ontology that **already encodes**:
- `infectious` vs `non-infectious` (27,964 vs 109,824 rows)
- `malignant` vs `benign` (21,413 vs 42,825 rows)
- pathogen subtype (bacterial / fungal / viral / parasitic)

This means the **contagion labels come free from the data** — no manual labelling of 390 diseases.

---

## 4. Model architecture

- **Backbone:** **DermLIP ViT-B-16** (dermatology CLIP-style model, pretrained on 403,563 Derm1M pairs; arXiv 2503.14911 / Nature Medicine 2025). Weights open (CC BY-NC-ND).
- **Method:** frozen encoder → extract 512-d image features + 512-d text features → fused 1024-d → train lightweight classifier heads. (CPU-feasible; heavy work is one-time feature extraction.)
- **Evaluation design:** every stage is an **ablation** — image-only vs text-only vs fused.

---

## 5. Results so far

### Stage 1 — Contagion gate (binary: infectious vs non-infectious)
Test set: 27,558 rows (of 137,788). Train 110,230.

| Model | Accuracy | Balanced acc | F1 |
|---|---|---|---|
| image_only | 0.7613 | **0.7590** | 0.5622 |
| text_only | 0.9292 | 0.9312 | 0.8428 |
| **fused** | 0.9320 | **0.9317** | **0.8475** |

Confusion matrix (fused):
```
                  non-inf   infectious
non-infectious    20474       1491
infectious          384       5209     ← only 384 infections missed
```
➡️ **Infection sensitivity = 93.1%** (the safety-critical metric).

Per-pathogen recall (fused):
| Pathogen | n | Recall |
|---|---|---|
| bacterial | 2135 | 0.939 |
| fungal | 1423 | 0.915 |
| viral | 1413 | 0.946 |
| parasitic | 605 | 0.917 |

### Stage 2 — Disease identification (50 classes)
Rows 60,079 (train 48,048 / test 12,079), 50 non-contagious conditions incl. melanoma, BCC, SCC, nevus, psoriasis, eczema, lupus, pemphigus, CTCL.

| Model | Accuracy | Balanced acc | Macro-F1 |
|---|---|---|---|
| image_only | 0.4404 | 0.4658 | **0.4311** |
| text_only | 0.8734 | 0.8785 | 0.8736 |
| fused | 0.8503 | 0.8551 | 0.8553 |

**Per-class (fused):** easiest = Kaposi sarcoma 0.985, seborrheic keratoses 0.976, skin tag 0.973; hardest = dermatitis 0.629, eczema 0.658, psoriasis 0.760, CTCL 0.761 — i.e. the model struggles exactly where clinicians do (overlapping inflammatory conditions).

### Critical methodological finding — caption leakage
**53% of captions literally name the diagnosis** (melanoma 92%, tinea 92%, BCC 91%, eczema 90%).
➡️ So `text_only` performance is inflated; **`image_only` is the deployment-realistic number**. A de-leak re-evaluation (strip the diagnosis from text) is in progress.

---

## 6. Infrastructure

| | Details |
|---|---|
| Machine | College PC: Intel i9-13900 (24 threads), 32 GB RAM, **4 TB storage**, **CPU-only** (no NVIDIA GPU) |
| Env | Python 3.12, CPU PyTorch, open_clip + huggingface_hub + sklearn |
| Data | `E:\SkinShield\` (raw + manifest), `E:\imgs\` (423k images) |
| Access | AnyDesk remote access configured (unattended), PC set to never sleep |
| Artifacts | `manifest.csv` (labelled index), `gate_results_clean.json`, `stage2_results.json` |

---

## 7. Documentation produced
- Dataset curation & class-selection rationale (professor-facing HTML)
- Master plan v2 (full-scope two-stage design)
- Remote-access playbooks

---

## 8. Status & next steps

| Phase | Status |
|---|---|
| Download full Derm1M (34.5 GB) | ✅ Done |
| Extract images + build labelled manifest | ✅ Done (423,016 / 100%) |
| Stage 1 — contagion gate | ✅ Done (93.1% sensitivity) |
| Stage 2 — disease ID (50 classes) | ✅ Done (image-only macro-F1 0.431) |
| De-leak honest re-evaluation | ⏳ In progress |
| Zero-shot breadth across full ontology | ⬜ Next |
| ONNX export → Jetson TensorRT deploy | ⬜ Pending |
| Final professor report | ⬜ Pending |

---

## 9. Honest limitations
- **CPU-only training** — slower; fine for frozen-encoder + probe, would need GPU for full fine-tuning.
- **Caption leakage** inflates text-based numbers; being reported transparently.
- **Long tail:** 17,027 conditions have <5 pairs — not trainable; handled by zero-shot, not a trained head.
- Not yet validated on an external dataset (planned: SCIN).
