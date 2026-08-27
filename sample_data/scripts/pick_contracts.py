"""
Extracts the 20 best contracts from your dataset.

Your files:
  complete.json            — full contract texts (file_name + full_text)
  all_reshaped_clauses.csv — clause_type per contract (filename + clause_type + clause_text)

Strategy:
  1. Score each contract by how many TARGET clause types it has
     (using all_reshaped_clauses.csv — fast, no need to parse 71MB JSON first)
  2. Take top 20 by score
  3. Pull their full_text from complete.json
  4. Write to sample_data/contracts/base/

Run from project root:
  python sample_data/scripts/pick_contracts.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.parent
RAW_DIR       = PROJECT_ROOT / "sample_data" / "raw"
OUTPUT_DIR    = PROJECT_ROOT / "sample_data" / "contracts" / "base"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLAUSES_CSV   = RAW_DIR / "all_reshaped_clauses.csv"
COMPLETE_JSON = RAW_DIR / "complete.json"

MAX_CONTRACTS = 20
MAX_CHARS     = 60_000   # truncate very long contracts

# Clause types that give us the best contradiction demos
# Must match values in the clause_type column (case-insensitive partial match)
TARGET_TYPES = {
    "payment terms",
    "expiration date",
    "renewal term",
    "termination for convenience",
    "cap on liability",
    "governing laws",
    "license grant",
    "warranty",
    "ip ownership",
    "non-compete",
}


def safe_stem(name: str, maxlen: int = 60) -> str:
    """Filesystem-safe filename stem from a contract filename."""
    # Remove .pdf / .txt extension if present
    stem = Path(name).stem
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    return safe.strip()[:maxlen]


# ── Step 1: Score contracts by target clause coverage ──────────────────────
def score_contracts() -> dict[str, int]:
    """
    Returns {filename: score} where score = number of distinct
    target clause types found for that contract.
    """
    print(f"Scoring contracts from: {CLAUSES_CSV}")
    scores: dict[str, set] = defaultdict(set)

    with open(CLAUSES_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname  = row.get("filename", "").strip()
            ctype  = row.get("clause_type", "").strip().lower()
            ctext  = row.get("clause_text", "").strip()

            if not fname or not ctext:
                continue

            # Check if this clause type is one we care about
            for target in TARGET_TYPES:
                if target in ctype:
                    scores[fname].add(ctype)
                    break

    # Convert set sizes to counts
    result = {fname: len(ctypes) for fname, ctypes in scores.items()}
    print(f"  Scored {len(result)} unique contracts")
    return result


# ── Step 2: Pick top N contracts ───────────────────────────────────────────
def pick_top(scores: dict[str, int], n: int) -> list[str]:
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    print(f"\nTop {len(top)} contracts by target clause coverage:")
    for fname, score in top:
        print(f"  {score} target clauses  |  {fname[:70]}")
    return [fname for fname, _ in top]


# ── Step 3: Extract full texts from complete.json ──────────────────────────
def extract_full_texts(target_filenames: list[str]) -> dict[str, str]:
    """
    Matches CSV filenames to JSON file_names by stem (no extension),
    case-insensitive. This handles the .pdf vs .txt mismatch.
    """
    print(f"\nLoading full texts from: {COMPLETE_JSON}")
    print(f"  File size: {COMPLETE_JSON.stat().st_size // (1024*1024)} MB — loading...")

    with open(COMPLETE_JSON, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    print(f"  JSON loaded. Type: {type(data).__name__}, "
          f"length: {len(data) if isinstance(data, list) else 'dict'}")

    # Build lookup: stem_lower → full_text
    # stem = filename without extension, lowercased
    stem_to_text: dict[str, str] = {}

    if isinstance(data, list):
        for item in data:
            fname = item.get("file_name", "").strip()
            text  = item.get("full_text", "").strip()
            if fname and text:
                stem = Path(fname).stem.lower()
                stem_to_text[stem] = text

    elif isinstance(data, dict):
        for fname, text in data.items():
            if isinstance(text, str) and text:
                stem = Path(fname).stem.lower()
                stem_to_text[stem] = text

    print(f"  Built stem index: {len(stem_to_text)} entries")

    # Match target filenames by stem
    found: dict[str, str] = {}
    for target_fname in target_filenames:
        target_stem = Path(target_fname).stem.lower()
        if target_stem in stem_to_text:
            found[target_fname] = stem_to_text[target_stem]
        else:
            # Try partial match — first 60 chars of stem
            # (CSV truncates long filenames at 60 chars in the top-20 display)
            short_stem = target_stem[:60]
            matches = [
                (s, t) for s, t in stem_to_text.items()
                if s.startswith(short_stem) or short_stem in s
            ]
            if matches:
                found[target_fname] = matches[0][1]
                print(f"  ~ fuzzy match: {target_stem[:50]} → {matches[0][0][:50]}")

    print(f"  Found full text for {len(found)} / {len(target_filenames)} contracts")

    missing = [f for f in target_filenames if f not in found]
    if missing:
        print(f"  Missing ({len(missing)}):")
        for m in missing[:5]:
            print(f"    {m[:70]}")

    return found
# ── Step 4: Write contracts to base/ ──────────────────────────────────────
def write_contracts(texts: dict[str, str]) -> int:
    written = 0
    for fname, text in texts.items():
        stem = safe_stem(fname)
        dest = OUTPUT_DIR / f"{stem}.txt"

        if dest.exists():
            print(f"  SKIP (exists): {dest.name}")
            continue

        dest.write_text(text[:MAX_CHARS], encoding="utf-8")
        size_kb = len(text[:MAX_CHARS]) // 1024
        written += 1
        print(f"  ✓ {size_kb:4d} KB  {dest.name}")

    return written


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Contract Picker — CUAD / changethan dataset")
    print("=" * 65)
    print(f"Source : {RAW_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print()

    # 1. Score
    scores = score_contracts()
    if not scores:
        print("ERROR: No contracts scored — check CLAUSES_CSV path and columns")
        return

    # 2. Pick top 20
    top_filenames = pick_top(scores, MAX_CONTRACTS)

    # 3. Get full texts
    full_texts = extract_full_texts(top_filenames)
    if not full_texts:
        print("\nERROR: No full texts found in complete.json")
        print("Check that file_name values in JSON match filename values in CSV")
        # Print a sample from each to help debug
        print("\nSample filename from CSV:")
        print(f"  {top_filenames[0]}")
        print("\nSample file_name from JSON (first entry):")
        with open(COMPLETE_JSON, encoding="utf-8") as f:
            sample = json.load(f)
            if isinstance(sample, list) and sample:
                print(f"  {sample[0].get('file_name', 'KEY NOT FOUND')}")
        return

    # 4. Write
    print(f"\nWriting contracts to: {OUTPUT_DIR}")
    written = write_contracts(full_texts)

    # Summary
    print(f"\n{'='*65}")
    print(f"Done. Written {written} contracts to {OUTPUT_DIR}")
    files = list(OUTPUT_DIR.glob("*.txt"))
    total_kb = sum(f.stat().st_size for f in files) // 1024
    print(f"Total: {len(files)} files, {total_kb} KB")
    print()

    # Show clause coverage for written contracts
    print("Clause coverage summary:")
    with open(CLAUSES_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        coverage: dict[str, set] = defaultdict(set)
        for row in reader:
            fname = row.get("filename", "").strip()
            ctype = row.get("clause_type", "").strip()
            if fname in set(full_texts.keys()):
                coverage[fname].add(ctype)

    for fname in top_filenames[:5]:
        if fname in coverage:
            clauses = sorted(coverage[fname])
            stem = safe_stem(fname)
            print(f"  {stem[:40]:40s}  {len(clauses)} clause types")
            for c in clauses[:6]:
                print(f"    - {c}")
            if len(clauses) > 6:
                print(f"    ... and {len(clauses)-6} more")


if __name__ == "__main__":
    main()