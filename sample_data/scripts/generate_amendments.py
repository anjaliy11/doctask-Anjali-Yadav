"""
Generates amendments + invoices for each base contract.

For each contract:
  - Amendment No. 1: changes key clauses (deliberate contradictions)
  - Invoice: references the amended terms

Contradictions introduced:
  - Payment terms: net-30 → net-60 (or vice versa)
  - Termination notice: 90 days → 30 days
  - Renewal: auto-renew → written notice required
  - Liability cap: increase or decrease the amount
  - Governing law: State X → State Y

Run from project root:
  python sample_data/scripts/generate_amendments.py

Needs:
  GEMINI_API_KEY in backend/.env

Re-run safe:
  skips already-generated files.
"""

import os
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


# ============================================================================
# PROJECT PATHS
# ============================================================================

# generate_amendments.py
#   ↓
# sample_data/
#   ↓
# project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT / "sample_data" / "contracts" / "base"
AMENDMENTS_DIR = PROJECT_ROOT / "sample_data" / "contracts" / "amendments"
INVOICES_DIR = PROJECT_ROOT / "sample_data" / "contracts" / "invoices"

ENV_PATH = PROJECT_ROOT / "backend" / ".env"


# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

print(f"Looking for .env at: {ENV_PATH}")
print(f".env exists: {ENV_PATH.exists()}")

if not ENV_PATH.exists():
    raise SystemExit(
        f"\nERROR: .env file not found at:\n{ENV_PATH}\n"
    )

load_dotenv(ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

print(f"GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)}")

if not GEMINI_API_KEY:
    raise SystemExit(
        "\nERROR: GEMINI_API_KEY not set.\n"
        f"Add it to:\n{ENV_PATH}\n\n"
        "Expected format:\n"
        "GEMINI_API_KEY=your_key_here\n"
    )


# ============================================================================
# GEMINI SDK
# ============================================================================

# IMPORTANT:
# This uses the NEW google-genai package.
#
# Install with:
#   python -m pip install -U google-genai
#
# Do NOT use:
#   import google.generativeai as genai

try:
    from google import genai
except ImportError as e:
    raise SystemExit(
        "\nERROR: The new Google GenAI SDK is not installed.\n\n"
        "Run:\n"
        "  python -m pip install -U google-genai\n\n"
        "Then run this script again.\n"
    ) from e


try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise SystemExit(
        f"\nERROR: Could not initialize Gemini client:\n{e}\n"
    ) from e


# Model can be overridden through backend/.env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

print(f"Gemini model: {GEMINI_MODEL}")


# ============================================================================
# DIRECTORIES
# ============================================================================

AMENDMENTS_DIR.mkdir(parents=True, exist_ok=True)
INVOICES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONTRACT_CHARS = 4000
MAX_AMENDMENT_CHARS = 1500

# Seconds between Gemini calls.
# Free-tier APIs can be rate limited.
RATE_LIMIT_SLEEP = 4


# ============================================================================
# PROMPTS
# ============================================================================

AMENDMENT_PROMPT = """
You are a legal document drafter.

Read this contract excerpt and write a formal AMENDMENT NO. 1 that
deliberately contradicts at least one key clause.

CONTRADICTION RULES — apply whichever fits this contract best:

- If payment is net-30 → change to net-60 (or vice versa)
- If termination notice is 90 days → change to 30 days (or vice versa)
- If renewal is automatic → change to require written notice
- If there is a liability cap amount → change the amount
- If governing law is State X → change to State Y

Pick the 2-3 most impactful contradictions you can find.

OUTPUT FORMAT (use exactly these headings):

AMENDMENT NO. 1

Reference Contract: [extract the contract name or ID from the text]

Effective Date: [pick a date 6-12 months after any date in the contract]

Parties: [extract party names from the contract]


AMENDED CLAUSES:

[For each changed clause:

1. Cite the original section number.
2. State the original language.
3. State the new language.

Make the contradiction explicit.]


CONTRADICTIONS INTRODUCED:

[Bullet list of exactly what changed.
Be specific with numbers, dates, and terms.

Example:
- Payment terms changed from Net-30 to Net-60.
- Termination notice changed from 90 days to 30 days.
- Renewal changed from automatic renewal to written notice.]

This is important for testing automated conflict detection.


All other terms and conditions of the Agreement remain unchanged.


CONTRACT EXCERPT:

{contract_text}
"""


INVOICE_PROMPT = """
Write a realistic invoice referencing this contract and its amendment.

IMPORTANT:
The invoice MUST use the AMENDED payment terms, not the original terms.

Use realistic dollar amounts for the type of services described.

OUTPUT FORMAT (use exactly these headings):

INVOICE

Invoice Number: INV-{year}-{invoice_num}

Invoice Date: {invoice_date}

Due Date: [based on amended payment terms]

Reference: [contract name/ID from the contract text]


FROM:

[Vendor/service provider name and address — extract from contract]


TO:

[Client/buyer name and address — extract from contract]


SERVICES RENDERED:

[2-4 line items matching the contract's service type,
with hours/units and rates]


SUBTOTAL: $[amount]

TAX: $0.00 (B2B services)

TOTAL DUE: $[amount]


PAYMENT TERMS:

[State the AMENDED terms explicitly.

Example:
"Net-60 per Amendment No. 1 effective [date].
Original Net-30 terms superseded."]


NOTES:

[1-2 sentences referencing the amendment and confirming
that the original payment terms no longer apply.]


CONTRACT EXCERPT:

{contract_text}


AMENDMENT EXCERPT:

{amendment_text}
"""


# ============================================================================
# HELPERS
# ============================================================================

def truncate(text: str, chars: int) -> str:
    """Return at most `chars` characters."""
    return text[:chars]


def call_gemini(prompt: str, retries: int = 3) -> str:
    """Call Gemini using the Interactions API."""

    for attempt in range(retries):
        try:
            interaction = client.interactions.create(
                model=GEMINI_MODEL,
                input=prompt,
            )

            if not interaction.output_text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return interaction.output_text.strip()

        except Exception as e:
            msg = str(e).lower()

            # Permanent model/configuration errors
            if (
                "404" in msg
                or "not_found" in msg
                or "not found" in msg
                or "invalid_argument" in msg
            ):
                raise RuntimeError(
                    f"Gemini model/API configuration error.\n"
                    f"Model: {GEMINI_MODEL}\n"
                    f"Error: {e}"
                ) from e

            # Rate limit
            if (
                "429" in msg
                or "quota" in msg
                or "rate limit" in msg
                or "resource exhausted" in msg
            ):
                wait = RATE_LIMIT_SLEEP * (attempt + 2)

                print(
                    f"    Rate limit hit. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            if attempt < retries - 1:
                print(f"    Gemini error: {e}")
                print(
                    f"    Retrying in {RATE_LIMIT_SLEEP}s..."
                )

                time.sleep(RATE_LIMIT_SLEEP)
                continue

            raise

    raise RuntimeError(
        "Gemini request failed after retries."
    )

def extract_year(text: str) -> str:
    """
    Extract a four-digit year correctly.

    Avoids the old bug:
        r'\\b(19|20)\\d{2}\\b'

    which returns only '19' or '20'.
    """

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        text,
    )

    return years[0] if years else str(date.today().year)


# ============================================================================
# MAIN
# ============================================================================

def main():

    contracts = sorted(BASE_DIR.glob("*.txt"))

    if not contracts:
        print(
            f"\nERROR: No contracts found in:\n{BASE_DIR}"
        )

        print(
            "\nRun pick_contracts.py first."
        )

        return

    print()
    print("=" * 70)
    print(f"Found {len(contracts)} base contracts")
    print(f"Amendments → {AMENDMENTS_DIR}")
    print(f"Invoices   → {INVOICES_DIR}")
    print("=" * 70)
    print()

    # Track invoice numbering
    invoice_counter = 1

    processed = 0
    skipped = 0

    for i, contract_path in enumerate(contracts):

        stem = contract_path.stem

        amendment_path = (
            AMENDMENTS_DIR
            / f"{stem}_amendment_1.txt"
        )

        invoice_path = (
            INVOICES_DIR
            / f"{stem}_invoice_001.txt"
        )

        print(
            f"[{i + 1}/{len(contracts)}] "
            f"{stem[:55]}"
        )

        # ====================================================================
        # READ CONTRACT
        # ====================================================================

        try:
            contract_text = contract_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception as e:
            print(
                f"  ✗ Could not read contract: {e}"
            )

            skipped += 1
            print()
            continue

        # ====================================================================
        # AMENDMENT
        # ====================================================================

        if amendment_path.exists():

            print(
                "  SKIP amendment (exists)"
            )

            amendment_text = amendment_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        else:

            prompt = AMENDMENT_PROMPT.format(
                contract_text=truncate(
                    contract_text,
                    MAX_CONTRACT_CHARS,
                )
            )

            try:

                amendment_text = call_gemini(
                    prompt
                )

                amendment_path.write_text(
                    amendment_text,
                    encoding="utf-8",
                )

                print(
                    f"  ✓ Amendment written "
                    f"({len(amendment_text) // 1024}KB)"
                )

                processed += 1

            except Exception as e:

                print(
                    f"  ✗ Amendment FAILED: {e}"
                )

                skipped += 1

                time.sleep(
                    RATE_LIMIT_SLEEP
                )

                continue

            # Give the API a little breathing room
            time.sleep(
                RATE_LIMIT_SLEEP
            )

        # ====================================================================
        # INVOICE
        # ====================================================================

        if invoice_path.exists():

            print(
                "  SKIP invoice (exists)"
            )

        else:

            year = extract_year(
                amendment_text
            )

            invoice_date = date.today().strftime(
                "%B %d, %Y"
            )

            prompt = INVOICE_PROMPT.format(
                contract_text=truncate(
                    contract_text,
                    MAX_CONTRACT_CHARS,
                ),
                amendment_text=truncate(
                    amendment_text,
                    MAX_AMENDMENT_CHARS,
                ),
                year=year,
                invoice_num=str(
                    invoice_counter
                ).zfill(4),
                invoice_date=invoice_date,
            )

            try:

                invoice_text = call_gemini(
                    prompt
                )

                invoice_path.write_text(
                    invoice_text,
                    encoding="utf-8",
                )

                print(
                    f"  ✓ Invoice written "
                    f"({len(invoice_text) // 1024}KB)"
                )

                invoice_counter += 1

            except Exception as e:

                print(
                    f"  ✗ Invoice FAILED: {e}"
                )

                skipped += 1

            time.sleep(
                RATE_LIMIT_SLEEP
            )

        print()

    # ========================================================================
    # SUMMARY
    # ========================================================================

    amendments = list(
        AMENDMENTS_DIR.glob("*.txt")
    )

    invoices = list(
        INVOICES_DIR.glob("*.txt")
    )

    print("=" * 70)
    print("Done.")
    print(
        f"  Amendments : {len(amendments)} files "
        f"in {AMENDMENTS_DIR.name}/"
    )

    print(
        f"  Invoices   : {len(invoices)} files "
        f"in {INVOICES_DIR.name}/"
    )

    print(
        f"  Processed  : {processed} new files generated"
    )

    print(
        f"  Skipped    : {skipped} failures"
    )

    print()

    # ========================================================================
    # SAMPLE AMENDMENT
    # ========================================================================

    print(
        "Sample amendment (first 500 chars):"
    )

    if amendments:

        print("-" * 70)

        try:
            sample = amendments[0].read_text(
                encoding="utf-8",
                errors="ignore",
            )

            print(sample[:500])

        except Exception as e:

            print(
                f"Could not read sample amendment: {e}"
            )

        print("-" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()