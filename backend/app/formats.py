"""
Single source of truth for which document formats this system accepts.

Centralizing this fixes a real bug: format detection was previously
duplicated verbatim in routes/ingest.py, watcher.py, and mcp_server.py,
and NONE of them actually rejected an unsupported format. A PDF would
sail through with format="pdf" (or, in ingest.py's case, format was
hardcoded to "txt" regardless of the real extension), get written to
raw/{id}.txt as raw undecoded bytes, and _read_document_text() would
then decode those bytes as UTF-8 (garbage in, garbage out) instead of
failing loudly. Every ingestion entry point below now calls
is_supported() before persisting anything, so "second run, different
documents" fails clearly on an unsupported format instead of silently
producing an empty or corrupted extraction.
"""
from pathlib import Path

# Formats classify_and_chunk / _read_document_text can actually read
# today. PDF/DOCX parsing is architecturally wired (see the single seam
# at app/graph/nodes.py::_read_document_text) but not implemented — see
# README "Known simplifications: PDF/DOCX parsing not implemented."
# Accepting them here would silently corrupt extraction; rejecting them
# loudly, with a clear error, is the honest failure mode the task brief
# asks for ("it never bluffs").
SUPPORTED_FORMATS = {"txt"}

_EXTENSION_MAP = {
    ".txt": "txt",
    ".md": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
}


def detect_format(filename: str) -> str:
    """
    Maps a filename's extension to a normalized format string.
    Unknown extensions map to 'unknown', NOT to 'txt' — guessing txt
    for an unrecognized extension is exactly how unsupported content
    used to sneak past every check silently.
    """
    suffix = Path(filename).suffix.lower()
    return _EXTENSION_MAP.get(suffix, "unknown")


def is_supported(filename: str) -> bool:
    return detect_format(filename) in SUPPORTED_FORMATS


def unsupported_format_message(filename: str) -> str:
    fmt = detect_format(filename)
    if fmt == "unknown":
        return (
            f"'{filename}' has an unrecognized extension. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}."
        )
    return (
        f"'{filename}' is a {fmt.upper()} file. {fmt.upper()} parsing is not "
        f"implemented in this build — only plain text (.txt, .md) is "
        f"currently supported. See README 'Known simplifications' for the "
        f"planned extension point (app/graph/nodes.py::_read_document_text)."
    )
