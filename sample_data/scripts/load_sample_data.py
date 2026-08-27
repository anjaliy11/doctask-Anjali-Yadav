"""
Loads sample contracts into the pipeline via the REST API.

What this does:
  1. Finds all contract groups under sample_data/contracts/
     - A "group" = base contract + optional amendment + optional invoice
     - Naming convention:
         base:      {stem}.txt
         amendment: {stem}_amendment_1.txt
         invoice:   {stem}_invoice_001.txt
  2. For each group, creates ONE pile (name = contract short name)
     with a standard vendor-contract rule checklist attached.
  3. POSTs documents in order: base → amendment → invoice
     (order matters: reconcile compares amendment against base,
      invoice against both)
  4. Polls until each run finishes or hits human_gate (paused).
  5. Prints a summary table: pile_id, docs ingested, run status,
     findings count, any contradictions detected.

Run from project root (backend venv active):
  python sample_data/scripts/load_sample_data.py

Options (env vars):
  API_BASE   = http://localhost:8000   (default)
  ONLY_STEM  =           load one group only (for testing)
  MAX_GROUPS = 3                       load first N groups (default: all 9)
  DRY_RUN    = 1                       print plan, don't call API

Re-run safe:
  The API deduplicates by content hash, so re-running this script
  does not create duplicate documents or piles. Each run just gets
  "duplicate_skipped" for docs already uploaded.
  However, it WILL create a new pile each time for groups not yet
  loaded. Use ONLY_STEM to reload one group specifically.
"""

import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE   = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
ONLY_STEM  = os.getenv("ONLY_STEM", "").strip()
MAX_GROUPS = int(os.getenv("MAX_GROUPS", "99"))
DRY_RUN    = os.getenv("DRY_RUN", "0").strip() == "1"

POLL_INTERVAL_S  = 3
POLL_MAX_RETRIES = 40   # 40 × 3s = 2 min per document max

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
BASE_DIR      = PROJECT_ROOT / "sample_data" / "contracts" / "base"
AMENDMENTS_DIR= PROJECT_ROOT / "sample_data" / "contracts" / "amendments"
INVOICES_DIR  = PROJECT_ROOT / "sample_data" / "contracts" / "invoices"

# Standard rule checklist attached to every vendor-contract pile.
# These are the rules the examine_rules node checks against extracted facts.
STANDARD_RULES = [
    {
        "rule_id": "payment_terms_present",
        "description": (
            "The pile must contain explicit payment terms (net-days or equivalent). "
            "If an amendment changes them, the amendment's terms supersede the original."
        ),
    },
    {
        "rule_id": "termination_notice_defined",
        "description": (
            "A termination-for-convenience clause must specify a notice period in days. "
            "Ambiguous or missing notice periods are a finding."
        ),
    },
    {
        "rule_id": "governing_law_stated",
        "description": (
            "The agreement must state the governing law jurisdiction. "
            "Conflicting jurisdictions across documents must be surfaced."
        ),
    },
    {
        "rule_id": "liability_cap_present",
        "description": (
            "A cap on liability or limitation-of-liability clause should be present. "
            "Absence is an info-level finding."
        ),
    },
    {
        "rule_id": "amendment_supersedes_original",
        "description": (
            "Where an amendment explicitly supersedes original terms, "
            "the invoice must reference the amended terms, not the original ones. "
            "An invoice citing superseded terms is a critical finding."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ContractGroup(NamedTuple):
    stem: str                  # base filename without .txt
    short_name: str            # human-readable label (first 60 chars of stem)
    base_path: Path
    amendment_path: Path | None
    invoice_path: Path | None


class LoadResult(NamedTuple):
    stem: str
    pile_id: str
    docs_ingested: int
    docs_skipped: int
    run_ids: list[str]
    final_statuses: list[str]
    findings_count: int
    contradiction_count: int
    error: str | None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_groups() -> list[ContractGroup]:
    """
    Walk base/ and pair each contract with its amendment and invoice
    using the known naming convention.
    Returns groups sorted by stem name, filtered by ONLY_STEM / MAX_GROUPS.
    """
    groups: list[ContractGroup] = []

    for base_path in sorted(BASE_DIR.glob("*.txt")):
        stem = base_path.stem  # filename without .txt

        if ONLY_STEM and ONLY_STEM.lower() not in stem.lower():
            continue

        amendment_path = AMENDMENTS_DIR / f"{stem}_amendment_1.txt"
        invoice_path   = INVOICES_DIR   / f"{stem}_invoice_001.txt"

        groups.append(ContractGroup(
            stem           = stem,
            short_name     = stem[:60],
            base_path      = base_path,
            amendment_path = amendment_path if amendment_path.exists() else None,
            invoice_path   = invoice_path   if invoice_path.exists()   else None,
        ))

    return groups[:MAX_GROUPS]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post(client: httpx.Client, path: str, json: dict) -> dict:
    url = f"{API_BASE}{path}"
    resp = client.post(url, json=json, timeout=30)
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"POST {path} → {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()


def _get(client: httpx.Client, path: str) -> dict:
    url = f"{API_BASE}{path}"
    resp = client.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET {path} → {resp.status_code}: {resp.text[:200]}"
        )
    return resp.json()


def create_pile(client: httpx.Client, name: str) -> str:
    """Returns pile_id."""
    data = _post(client, "/piles", {
        "name": name,
        "rules": STANDARD_RULES,
    })
    return data["id"]


def ingest_doc(
    client: httpx.Client,
    pile_id: str,
    path: Path,
    doc_type_label: str,
) -> tuple[str, str, str]:
    """
    Ingest one document. Returns (document_id, run_id, status).
    status is "processing" or "duplicate_skipped".
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    data = _post(client, f"/piles/{pile_id}/documents", {
        "filename": path.name,
        "text_content": text,
    })
    return data["document_id"], data["run_id"], data["status"]


def poll_run(client: httpx.Client, run_id: str) -> dict:
    """
    Poll GET /runs/{run_id} until the run is no longer actively
    processing (done / failed / paused waiting for human approval).
    Returns the final RunStatus dict.
    """
    if not run_id:
        return {"status": "skipped", "stage": "duplicate_skipped"}

    for attempt in range(POLL_MAX_RETRIES):
        try:
            status = _get(client, f"/runs/{run_id}")
        except RuntimeError as e:
            # Run row may not exist yet if background task hasn't started
            if attempt < 3:
                time.sleep(POLL_INTERVAL_S)
                continue
            raise

        run_status = status.get("status", "")
        stage      = status.get("stage", "")

        # Terminal states
        if run_status in ("done", "failed"):
            return status

        # Paused at human_gate — not an error, expected behavior
        if run_status == "paused":
            return status

        # Still running but graph is interrupted waiting for approval
        if status.get("pending_interrupt"):
            return status

        # Still processing — keep polling
        time.sleep(POLL_INTERVAL_S)

    # Timeout — return last known status rather than crashing
    return {"status": "timeout", "stage": "unknown", "run_id": run_id}


def get_findings(client: httpx.Client, pile_id: str) -> list[dict]:
    try:
        return _get(client, f"/findings/pile/{pile_id}")
    except RuntimeError:
        return []


# ---------------------------------------------------------------------------
# Load one group
# ---------------------------------------------------------------------------

def load_group(client: httpx.Client, group: ContractGroup) -> LoadResult:
    docs_ingested = 0
    docs_skipped  = 0
    run_ids: list[str]      = []
    final_statuses: list[str] = []
    error: str | None       = None

    try:
        # 1. Create pile
        pile_id = create_pile(client, group.short_name)
        print(f"  pile_id = {pile_id}")

        # 2. Documents in order: base → amendment → invoice
        docs_to_load = [
            (group.base_path,       "base contract"),
            (group.amendment_path,  "amendment"),
            (group.invoice_path,    "invoice"),
        ]

        for path, label in docs_to_load:
            if path is None:
                print(f"  {label:20s} — not found, skipping")
                continue

            print(f"  {label:20s} ← {path.name[:55]}")
            doc_id, run_id, status = ingest_doc(client, pile_id, path, label)

            if status == "duplicate_skipped":
                docs_skipped += 1
                print(f"                       → duplicate, skipped")
                continue

            docs_ingested += 1
            print(f"                       → doc_id={doc_id[:8]}… run_id={run_id[:8]}…")

            # Poll the run to completion / human gate
            run_status = poll_run(client, run_id)
            final_status = run_status.get("status", "unknown")
            stage        = run_status.get("stage", "unknown")
            run_ids.append(run_id)
            final_statuses.append(final_status)

            interrupt = run_status.get("pending_interrupt")
            if interrupt:
                print(
                    f"                       → PAUSED at human_gate "
                    f"(finding: {interrupt.get('rule_id','?')})"
                )
            else:
                print(f"                       → {final_status} / stage={stage}")

            # Brief pause between documents so the background task
            # pool doesn't get hammered when MAX_GROUPS is large
            time.sleep(1)

        # 3. Collect findings
        findings = get_findings(client, pile_id)
        contradiction_count = sum(
            1 for f in findings if f.get("rule_id") == "contradiction"
        )

    except Exception as e:
        error = str(e)
        pile_id = pile_id if "pile_id" in dir() else "ERROR"
        findings = []
        contradiction_count = 0

    return LoadResult(
        stem               = group.stem[:55],
        pile_id            = pile_id,
        docs_ingested      = docs_ingested,
        docs_skipped       = docs_skipped,
        run_ids            = run_ids,
        final_statuses     = final_statuses,
        findings_count     = len(findings),
        contradiction_count= contradiction_count,
        error              = error,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_api_health(client: httpx.Client) -> None:
    try:
        resp = client.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code != 200:
            print(f"WARNING: /health returned {resp.status_code}")
    except httpx.ConnectError:
        print(
            f"\nERROR: Cannot reach API at {API_BASE}\n"
            f"Start the backend first:\n"
            f"  cd backend && uvicorn app.main:app --reload\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("DocTask — Sample Data Loader")
    print("=" * 70)
    print(f"API:        {API_BASE}")
    print(f"Base dir:   {BASE_DIR}")
    print(f"Max groups: {MAX_GROUPS}")
    print(f"Filter:     {ONLY_STEM or '(none)'}")
    print(f"Dry run:    {DRY_RUN}")
    print()

    # Discover groups
    groups = discover_groups()
    if not groups:
        print("No contract groups found. Check BASE_DIR path and ONLY_STEM filter.")
        sys.exit(1)

    print(f"Found {len(groups)} group(s):")
    for g in groups:
        amend = "✓ amendment" if g.amendment_path else "  no amendment"
        inv   = "✓ invoice"   if g.invoice_path   else "  no invoice"
        print(f"  {g.short_name[:50]:50s}  {amend}  {inv}")
    print()

    if DRY_RUN:
        print("DRY_RUN=1 — stopping here, no API calls made.")
        return

    # Health check
    client = httpx.Client()
    check_api_health(client)

    # Load each group
    results: list[LoadResult] = []
    for i, group in enumerate(groups):
        print(f"[{i+1}/{len(groups)}] {group.short_name[:60]}")
        result = load_group(client, group)
        results.append(result)
        if result.error:
            print(f"  ERROR: {result.error}")
        print()

    client.close()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Stem':40s}  {'Pile':8s}  "
        f"{'In':>3}  {'Skip':>4}  "
        f"{'Findings':>8}  {'Contrad':>7}  {'Status'}"
    )
    print("-" * 70)

    total_contradictions = 0
    total_findings       = 0
    total_errors         = 0

    for r in results:
        statuses = ", ".join(set(r.final_statuses)) if r.final_statuses else "—"
        err_flag = " ← ERROR" if r.error else ""
        print(
            f"{r.stem[:40]:40s}  {r.pile_id[:8]}  "
            f"{r.docs_ingested:>3}  {r.docs_skipped:>4}  "
            f"{r.findings_count:>8}  {r.contradiction_count:>7}  "
            f"{statuses}{err_flag}"
        )
        total_contradictions += r.contradiction_count
        total_findings       += r.findings_count
        if r.error:
            total_errors += 1

    print("-" * 70)
    print(
        f"{'TOTAL':40s}  {'':8s}  "
        f"{sum(r.docs_ingested for r in results):>3}  "
        f"{sum(r.docs_skipped  for r in results):>4}  "
        f"{total_findings:>8}  {total_contradictions:>7}"
    )
    print()

    if total_contradictions > 0:
        print(
            f"✓ {total_contradictions} contradiction(s) detected — "
            f"pending human review at GET /findings/pile/{{pile_id}}"
        )
    else:
        print(
            "⚠  No contradictions detected. "
            "Check that amendments were ingested and LLM keys are set."
        )

    if total_errors:
        print(f"✗ {total_errors} group(s) failed — see ERROR lines above")
        sys.exit(1)

    print()
    print(
        "Next: open http://localhost:5173 to review findings, "
        "or use the MCP tools to approve/reject programmatically."
    )


if __name__ == "__main__":
    main()