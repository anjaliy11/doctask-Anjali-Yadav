"""
Watches app.config.settings.inbox_dir. Layout convention: one subfolder per
pile, named after the pile —

    sample_data/inbox/
      vendor-acme/
        contract.txt          <- picked up on startup AND on later arrival
        amendment-1.txt        <- arriving later triggers a TARGETED update,
                                   not a full re-run, via the same graph path
                                   ingest.py uses (see nodes.py's touched-
                                   slots scoping in reconcile/examine)

Run as its own process: `python -m app.watcher`. Independent of the FastAPI
process — the API can be down and files dropped in the inbox will still
queue up; they're picked up on the next watcher pass (or the watcher's own
startup scan) rather than lost.
"""
import logging
import os
import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.config import get_settings
from app.db.models import Pile
from app.db.session import SessionLocal
from app.routes.ingest import _run_graph_background, ingest_text_document_sync_part

logger = logging.getLogger("doctask.watcher")
logging.basicConfig(level=logging.INFO)

_SEEN_LOCK = threading.Lock()
_SEEN: set[str] = set()  # absolute file paths already queued THIS process run;
# durable de-dup (across restarts) is the content_hash unique constraint in
# ingest_text_document_sync_part — this in-memory set is only to avoid
# double-triggering on the same event firing twice within one process.


def _get_or_create_pile(session, name: str) -> str:
    pile = session.query(Pile).filter(Pile.name == name).first()
    if pile is not None:
        return pile.id
    pile = Pile(name=name, rules_json={"rules": []})
    session.add(pile)
    try:
        session.commit()
    except Exception:
        # Race with another watcher/process creating the same-named pile
        # concurrently — not fatal, just re-fetch.
        session.rollback()
        pile = session.query(Pile).filter(Pile.name == name).first()
        if pile is None:
            raise
    return pile.id


def _handle_file(path: str) -> None:
    with _SEEN_LOCK:
        if path in _SEEN:
            return
        _SEEN.add(path)

    if not os.path.isfile(path):
        return
    pile_name = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text_content = f.read()
    except OSError as e:
        logger.error("Could not read %s: %s — skipping, will retry on next watcher restart.", path, e)
        with _SEEN_LOCK:
            _SEEN.discard(path)  # allow retry — this failure is likely transient (file still being written)
        return

    if not text_content.strip():
        logger.info("Skipping empty file %s", path)
        return

    session = SessionLocal()
    try:
        pile_id = _get_or_create_pile(session, pile_name)
        result = ingest_text_document_sync_part(
            session, pile_id=pile_id, filename=filename, text_content=text_content
        )
    except Exception:
        logger.exception("Failed to ingest %s", path)
        with _SEEN_LOCK:
            _SEEN.discard(path)
        return
    finally:
        session.close()

    if result.status == "duplicate_skipped":
        logger.info("%s: byte-identical content already processed for pile %s, skipping.", path, pile_name)
        return

    logger.info("%s: queued as run %s on pile %s (%s)", path, result.run_id, pile_name, pile_id)
    # No FastAPI BackgroundTasks here — the watcher isn't inside a request
    # cycle. A plain thread is the equivalent: the graph run happens
    # off the watcher's event-handling thread so a slow LLM call doesn't
    # stall detection of the NEXT file arriving.
    threading.Thread(
        target=_run_graph_background,
        kwargs={"pile_id": pile_id, "document_id": result.document_id, "run_id": result.run_id},
        daemon=True,
    ).start()


class _InboxHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            _handle_file(event.src_path)

    def on_moved(self, event):
        # Covers the common "write to a temp name, then rename into place"
        # pattern many tools use to avoid partial-file reads — on_created
        # alone would miss this.
        if not event.is_directory:
            _handle_file(event.dest_path)


def _scan_existing(inbox_dir: str) -> None:
    """Startup scan: pick up files that were already sitting in the inbox
    before the watcher started, not just ones that arrive after."""
    if not os.path.isdir(inbox_dir):
        os.makedirs(inbox_dir, exist_ok=True)
        return
    for pile_dir in os.listdir(inbox_dir):
        full_pile_dir = os.path.join(inbox_dir, pile_dir)
        if not os.path.isdir(full_pile_dir):
            continue
        for fname in os.listdir(full_pile_dir):
            _handle_file(os.path.join(full_pile_dir, fname))


def main() -> None:
    settings = get_settings()
    inbox_dir = settings.inbox_dir
    logger.info("Scanning existing files in %s", inbox_dir)
    _scan_existing(inbox_dir)

    observer = Observer()
    observer.schedule(_InboxHandler(), inbox_dir, recursive=True)
    observer.start()
    logger.info("Watching %s for new documents (per-pile subfolders)", inbox_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
