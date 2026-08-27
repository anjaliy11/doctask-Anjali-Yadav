import re
from typing import TypedDict


class PileState(TypedDict):
    pile_id: str
    run_id: str
    document_ids: list[str]  # documents new to THIS run
    stage: str


# Heuristic, intentionally conservative (a few false positives are fine —
# they become a "suspicious content" finding for a human to dismiss, not a
# silent failure). Checked against BOTH raw source text and the model's own
# extracted claim/evidence text, since a successful injection could show up
# in either place.
_INJECTION_PATTERNS = [
    r"\bignore (all|any|the)?\s*(previous|prior|above)\s*instructions?\b",
    r"\bdisregard\s+(the|all|your)\s+(previous|prior|above|system)\b",
    r"\byou (are|must|should) now\s+(act|behave|respond|ignore)\b",
    r"\bnew instructions?:\s",
    r"\bsystem\s*:\s*.{0,20}(override|ignore|forget)\b",
    r"\bmark (all|this)\s+(findings?|items?)\s+as\s+(passed|approved|clean)\b",
    r"\bthis (document|section) (is|has been)\s+(pre[- ]?approved|verified|compliant)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injected_instruction(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))
