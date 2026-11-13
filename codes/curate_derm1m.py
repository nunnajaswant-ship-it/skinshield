"""
SkinShield — Derm1M curation pipeline (12-class locked set).

Runs on Google Colab (recommended: 78GB free disk) or locally.
Downloads ONLY the 8 zips needed for the 12 classes (~30.8 GB), extracts
matched images directly from the zips (no full unzip), splits 70/15/15
stratified by class, and exports a curated folder + zip.

USAGE
    pip install huggingface_hub pandas
    python curate_derm1m.py --token hf_xxx [--out data/curated] [--max-per-class 2679]

OUTPUT (in --out)
    images/train/<class>/<file>     images/val/<class>/<file>     images/test/<class>/<file>
    train.csv  val.csv  test.csv    (image_path, text, label, source, source_type, body_location, symptoms, age, gender)
    curated.zip                     (everything above, for download)

NOTES
    - The validation CSV (Derm1M_v2_validation.csv) has NO labels -> skipped.
    - Filenames are unique in the pretrain CSV -> no dedup needed.
    - "no definitive diagnosis" rows (24.8%) are excluded.
    - Multi-label rows are split on commas; each row maps to ONE class
      (first matching class wins).
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "redlessone/Derm1M"
PRETRAIN_CSV = "Derm1M_v2_pretrain.csv"

# Locked 12-class set (G1-verified 2026-09-20)
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

EXCLUDE_PATTERNS = {
    "pemphigus_vulgaris": ["foliaceus", "vegetans", "paraneoplastic", "drug-induced", "endemic"],
    "cutaneous_lupus": ["lupus pernio", "lupus vulgaris", "neonatal lupus"],
}

NOISE_PATTERNS = ["no definitive diagnosis", "clinical diagnosis: no", "unknown"]

# Zip per filename folder (verified: all CSV folders map to these)
FOLDER_TO_ZIP = {
    "IIYI": "IIYI.zip",
    "edu": "edu.zip",
    "note": "note.zip",
    "public": "public.zip",
    "pubmed": "pubmed.zip",
    "reddit": "reddit.zip",
    "twitter": "twitter.zip",
    "youtube": "youtube.zip",
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def clean_tokens(series: pd.Series) -> list:
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


def clean_single(raw: str) -> list:
    """Clean one label string into tokens (no pandas dependency)."""
    tokens = []
    lowered = str(raw).lower()
    if any(n in lowered for n in NOISE_PATTERNS):
        return tokens
    for part in lowered.split(","):
        part = part.strip()
        if part and part != "nan":
            tokens.append(part)
    return tokens


def match_class(token: str) -> str | None:
    for cls, aliases in TARGET_CLASSES.items():
        if any(a in token for a in aliases):
            if any(x in token for x in EXCLUDE_PATTERNS.get(cls, [])):
                return None
            return cls
    return None


def assign_class(row) -> str | None:
    """First matching class wins for a row's disease_label."""
    for tok in clean_single(row["disease_label"]):
        cls = match_class(tok)
        if cls:
            return cls
    return None


def stratified_split(df: pd.DataFrame, ratios=SPLIT_RATIOS, seed=SEED):
    train, val, test = [], [], []
    for cls, group in df.groupby("label", sort=False):
        g = group.sample(frac=1, random_state=seed)
        n = len(g)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.append(g.iloc[:n_train])
        val.append(g.iloc[n_train:n_train + n_val])
        test.append(g.iloc[n_train + n_val:])
    return pd.concat(train), pd.concat(val), pd.concat(test)


def zip_is_valid(path: str) -> bool:
    """Quick check: can we read the zip central directory (doesn't read contents)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except Exception:
        return False


def extract_from_zips(rows: pd.DataFrame, zip_dir: str, out_dir: str, split: str) -> None:
    """Extract only the needed files from each zip (no full unzip).

    Derm1M zip members are BARE filenames (e.g. '100_1.png') while the CSV
    filenames carry a folder prefix (e.g. 'reddit/100_1.png'). We match by
    basename within each zip (safe: full filenames are unique and each folder
    maps to exactly one zip, so basenames are unique per zip).
    """
    label_by_file = dict(zip(rows["filename"], rows["label"]))
    needed_by_zip = {}
    for fname in rows["filename"]:
        folder = fname.split("/")[0]
        zip_name = FOLDER_TO_ZIP.get(folder)
        if zip_name is None:
            print(f"  [warn] no zip for folder '{folder}' (file {fname})")
            continue
        needed_by_zip.setdefault(zip_name, set()).add(fname)

    total = len(rows)
    done = 0
    for zip_name, needed in needed_by_zip.items():
        zip_path = os.path.join(zip_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"  [skip] {zip_name} not downloaded")
            continue
        # basename -> label for THIS zip's needed files (unique per zip)
        label_by_base = {os.path.basename(f): label_by_file[f] for f in needed}
        print(f"  [extract] {zip_name}: {len(needed)} files needed")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                base = os.path.basename(member)
                if member in needed:
                    cls = label_by_file[member]
                elif base in label_by_base:
                    cls = label_by_base[base]
                else:
                    continue
                dest_dir = os.path.join(out_dir, "images", split, cls)
                os.makedirs(dest_dir, exist_ok=True)
                with zf.open(member) as src, open(os.path.join(dest_dir, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                done += 1
                if done % 2000 == 0:
                    print(f"    ... {done}/{total}")
    print(f"  [extract] done: {done}/{total} files")
    if done < total:
        print(f"  [extract] *** WARNING: {total - done} files could not be extracted ***")
        print(f"  [extract] Check that all zips were downloaded and are not corrupt.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Derm1M curation for SkinShield 12-class set")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF read token")
    parser.add_argument("--out", default="data/curated", help="output directory")
    parser.add_argument("--zip-dir", default="data/zips", help="where to store downloaded zips")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap samples per class (0 = no cap, keep all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="skip zip download (use existing --zip-dir)")
    parser.add_argument("--delete-zips", action="store_true",
                        help="delete downloaded zips after extraction (frees ~31 GB; "
                             "needed when local disk is tight)")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: no HF token. Set HF_TOKEN env var or pass --token hf_xxx.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.zip_dir, exist_ok=True)

    # --- 1. Metadata ---------------------------------------------------------
    csv_path = os.path.join(args.out, PRETRAIN_CSV)
    if not os.path.exists(csv_path):
        print("[1/6] Downloading metadata CSV ...")
        csv_path = hf_hub_download(repo_id=REPO_ID, filename=PRETRAIN_CSV,
                                   repo_type="dataset", token=args.token,
                                   local_dir=args.out)
    df = pd.read_csv(csv_path)
    print(f"      {len(df)} rows")

    # --- 2. Filter 12 classes --------------------------------------------------
    print("[2/6] Filtering 12 classes ...")
    df["label"] = df.apply(assign_class, axis=1)
    matched = df[df["label"].notna()].copy()
    print(f"      matched: {len(matched)} rows")
    print(Counter(matched["label"]).most_common())

    # --- 3. Optional class cap --------------------------------------------------
    if args.max_per_class > 0:
        print(f"[3/6] Capping classes at {args.max_per_class} samples ...")
        capped = []
        for cls, group in matched.groupby("label", sort=False):
            capped.append(group.sample(min(len(group), args.max_per_class), random_state=SEED))
        matched = pd.concat(capped)
        print(f"      after cap: {len(matched)} rows")
        print(Counter(matched["label"]).most_common())

    # --- 4. Stratified split -----------------------------------------------------
    print("[4/6] Stratified 70/15/15 split ...")
    train, val, test = stratified_split(matched)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"      {name}: {len(part)}")

    # --- 5. Download zips ---------------------------------------------------------
    if not args.skip_download:
        needed_zips = sorted({FOLDER_TO_ZIP[f] for f in matched["filename"].str.split("/").str[0]})
        print(f"[5/6] Downloading {len(needed_zips)} zips (~30.8 GB total) ...")
        for zip_name in needed_zips:
            dest = os.path.join(args.zip_dir, zip_name)
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000 and zip_is_valid(dest):
                print(f"      {zip_name}: already present ({os.path.getsize(dest)/1e9:.2f} GB), skipping")
                continue
            if os.path.exists(dest) and not zip_is_valid(dest):
                print(f"      {zip_name}: existing file is corrupt — re-downloading")
            # retry up to 3 times (HF can rate-limit or drop connections)
            for attempt in range(1, 4):
                try:
                    print(f"      {zip_name}: downloading (attempt {attempt}) ...")
                    hf_hub_download(repo_id=REPO_ID, filename=zip_name, repo_type="dataset",
                                    token=args.token, local_dir=args.zip_dir)
                    size_gb = os.path.getsize(dest) / 1e9
                    print(f"      {zip_name}: done ({size_gb:.2f} GB)")
                    break
                except Exception as e:
                    print(f"      {zip_name}: attempt {attempt} failed: {e}")
                    if attempt == 3:
                        print(f"      {zip_name}: giving up after 3 attempts")
                    else:
                        print(f"      {zip_name}: retrying ...")

    # --- 6. Extract + export --------------------------------------------------------
    print("[6/6] Extracting matched images and exporting CSVs ...")
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        extract_from_zips(part, args.zip_dir, args.out, split_name)
        out_csv = os.path.join(args.out, f"{split_name}.csv")
        part[["filename", "truncated_caption", "label", "source", "source_type",
              "body_location", "symptoms", "age", "gender"]].to_csv(out_csv, index=False)
        print(f"      wrote {out_csv}")

    # --- Verify images were actually extracted --------------------------------------
    print("\n[verify] counting extracted images ...")
    total_expected = len(train) + len(val) + len(test)
    total_images = 0
    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(args.out, "images", split_name)
        n = sum(len(files) for _, _, files in os.walk(img_dir)) if os.path.isdir(img_dir) else 0
        total_images += n
        print(f"      {split_name}: {n} images (expected {len({'train': train, 'val': val, 'test': test}[split_name])})")
    if total_images < total_expected:
        print(f"\n[verify] *** FAILURE: only {total_images}/{total_expected} images extracted. ***")
        print("[verify] Do NOT use this dataset. Check zip downloads and re-run.")
        return 1
    print(f"[verify] OK: {total_images}/{total_expected} images present")

    # --- Free disk before zipping (optional) ------------------------------------------
    if args.delete_zips:
        import glob as _glob
        freed = 0
        for z in _glob.glob(os.path.join(args.zip_dir, "*.zip")):
            freed += os.path.getsize(z)
            os.remove(z)
        print(f"[disk] deleted zips, freed {freed/1e9:.1f} GB")

    # --- Zip it up -------------------------------------------------------------------
    print("[zip] Creating curated.zip ...")
    zip_path = os.path.join(args.out, "curated.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(os.path.join(args.out, "images")):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, args.out))
        for csv_name in ["train.csv", "val.csv", "test.csv"]:
            zf.write(os.path.join(args.out, csv_name), csv_name)
    size_gb = os.path.getsize(zip_path) / 1024**3
    print(f"      curated.zip: {size_gb:.2f} GB")

    print("\nDONE. Curated dataset ready at:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())