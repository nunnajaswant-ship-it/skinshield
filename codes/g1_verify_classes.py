"""
G1 Gate — Derm1M class-count verification for the SkinShield class set.

WHAT THIS DOES
    1. Downloads ONLY the small metadata files from Derm1M (CSVs + ontology.json).
       NO images are downloaded (34.5 GB total repo; metadata is a few MB).
    2. Loads the pretrain CSV (the validation CSV has NO label column - skip it).
    3. Cleans messy labels: excludes "no definitive diagnosis", splits
       comma-separated multi-labels, lowercases.
    4. Counts image-text pairs per target class using an alias map.
    5. Applies the G1 gate: every class must have >= 500 pairs.
    6. Writes a report (console + JSON) and the full condition list for manual review.

REALITY CHECK (verified 2026-09-20 on Derm1M v2 pretrain CSV):
    - 413,210 rows (not 1M - the HF v2 release is smaller than the paper)
    - 22,708 unique raw labels (messy: multi-label commas, parenthetical notes)
    - 102,380 rows (24.8%) are "no definitive diagnosis" -> excluded
    - After cleaning: ~310K rows, ~20K unique label tokens

USAGE
    pip install huggingface_hub pandas
    python g1_verify_classes.py --token hf_xxxx
    # or: set HF_TOKEN=hf_xxxx in your environment, then:
    python g1_verify_classes.py

OUTPUT
    data/derm1m_meta/          downloaded metadata
    data/g1_report.json        machine-readable report
    data/g1_conditions_all.csv full condition list (for alias-map review)
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "redlessone/Derm1M"
META_FILES = [
    "Derm1M_v2_pretrain.csv",
    "ontology.json",
]
MIN_PAIRS_PER_CLASS = 500

# ---------------------------------------------------------------------------
# Target classes + alias patterns (case-insensitive substring match on the
# cleaned label tokens). LOCKED 12-class set (user decision 2026-09-20):
# 5 common + 7 rare (professor's full rare-disease mandate).
# ---------------------------------------------------------------------------
TARGET_CLASSES = {
    "melanoma": ["melanoma"],
    "basal_cell_carcinoma": ["basal cell carcinoma"],
    "squamous_cell_carcinoma": ["squamous cell carcinoma"],
    "atopic_dermatitis": ["atopic dermatitis", "atopic eczema"],
    "psoriasis": ["psoriasis"],
    "hidradenitis_suppurativa": ["hidradenitis suppurativa"],
    "cutaneous_lupus": ["lupus erythematosus", "cutaneous lupus", "discoid lupus"],
    "pemphigus_vulgaris": ["pemphigus vulgaris", "pemphigus"],
    "bullous_pemphigoid": ["bullous pemphigoid"],
    "epidermolysis_bullosa": ["epidermolysis bullosa"],
    "ichthyosis": ["ichthyosis"],
    "cutaneous_t_cell_lymphoma": ["cutaneous t cell lymphoma", "cutaneous t-cell lymphoma", "mycosis fungoides"],
}

# Aliases that must NOT match (e.g., "herpes" would catch herpes zoster).
EXCLUDE_PATTERNS = {
    "pemphigus_vulgaris": ["foliaceus", "vegetans", "paraneoplastic", "drug-induced", "endemic"],
    "cutaneous_lupus": ["lupus pernio", "lupus vulgaris", "neonatal lupus"],
}

NOISE_PATTERNS = ["no definitive diagnosis", "clinical diagnosis: no", "unknown"]


def download_metadata(out_dir: str, token: str) -> dict:
    """Download only the small metadata files. Returns {filename: local_path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for fname in META_FILES:
        print(f"[download] {fname} ...", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            token=token,
            local_dir=out_dir,
        )
        paths[fname] = path
        print(f"  -> {path}")
    return paths


def clean_labels(series: pd.Series) -> list:
    """Split multi-labels on commas, lowercase, drop noise tokens."""
    tokens = []
    for raw in series.dropna().astype(str):
        lowered = raw.lower()
        if any(n in lowered for n in NOISE_PATTERNS):
            continue
        for part in lowered.split(","):
            part = part.strip()
            if part and part != "nan":
                tokens.append(part)
    return tokens


def match_class(token: str, class_map: dict, exclude_map: dict) -> str | None:
    """Map a cleaned label token to a target class, or None."""
    for cls, aliases in class_map.items():
        if not any(a in token for a in aliases):
            continue
        excluded = exclude_map.get(cls, [])
        if any(x in token for x in excluded):
            return None
        return cls
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="G1 gate: verify Derm1M class counts")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF read token")
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: no HF token. Set HF_TOKEN env var or pass --token hf_xxx.")
        print("Get a token at https://huggingface.co/settings/tokens after accepting")
        print("the Derm1M conditions at https://huggingface.co/datasets/redlessone/Derm1M")
        return 1

    meta_dir = os.path.join(args.out, "derm1m_meta")
    paths = download_metadata(meta_dir, args.token)

    # --- Load pretrain CSV ---------------------------------------------------
    df = pd.read_csv(paths["Derm1M_v2_pretrain.csv"])
    print(f"[csv] Derm1M_v2_pretrain.csv: {len(df)} rows")
    print(f"[csv] columns: {list(df.columns)}")

    # --- Clean labels ---------------------------------------------------------
    tokens = clean_labels(df["disease_label"])
    print(f"[clean] {len(tokens)} label tokens after noise removal + multi-label split")
    token_counts = Counter(tokens)

    # --- Full condition list for manual review --------------------------------
    conditions_df = pd.DataFrame(token_counts.items(), columns=["condition", "count"])
    conditions_df = conditions_df.sort_values("count", ascending=False)
    conditions_csv = os.path.join(args.out, "g1_conditions_all.csv")
    conditions_df.to_csv(conditions_csv, index=False)
    print(f"[conditions] {len(token_counts)} unique tokens -> {conditions_csv}")

    # --- Count pairs per target class -----------------------------------------
    def count_class(class_map: dict, exclude_map: dict) -> dict:
        counts = Counter()
        for token in tokens:
            cls = match_class(token, class_map, exclude_map)
            if cls:
                counts[cls] += 1
        return counts

    main_counts = count_class(TARGET_CLASSES, EXCLUDE_PATTERNS)

    print("\n" + "=" * 70)
    print("G1 GATE — PAIR COUNTS PER TARGET CLASS (pretrain CSV)")
    print("LOCKED 12-CLASS SET: 5 common + 7 rare")
    print("=" * 70)
    results = {}
    gate_pass = True
    for cls in TARGET_CLASSES:
        count = main_counts.get(cls, 0)
        status = "PASS" if count >= MIN_PAIRS_PER_CLASS else "FAIL"
        if count < MIN_PAIRS_PER_CLASS:
            gate_pass = False
        results[cls] = {"pairs": count, "min_required": MIN_PAIRS_PER_CLASS, "status": status}
        print(f"  {cls:<26} {count:>8,} pairs   [{status}]")

    print("-" * 70)
    print(f"  GATE RESULT (12-class set): {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 70)

    # --- Report ----------------------------------------------------------------
    report = {
        "gate": "G1",
        "min_pairs_per_class": MIN_PAIRS_PER_CLASS,
        "gate_pass": gate_pass,
        "total_rows": len(df),
        "clean_label_tokens": len(tokens),
        "unique_clean_tokens": len(token_counts),
        "class_counts": results,
    }
    report_path = os.path.join(args.out, "g1_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")

    if not gate_pass:
        print("\nNEXT: review data/g1_conditions_all.csv, refine the alias map in this")
        print("script, and re-run. If a class still has < 500 pairs, swap/merge it")
        print("per the curation plan (Section 3, rule 4).")
        return 2
    print("\nG1 PASSED. Class set is locked. Proceed to image download for these classes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())