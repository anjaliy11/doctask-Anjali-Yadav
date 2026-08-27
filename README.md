# DocTask — Agentic Document Pile Analyst

**Author:** Anjali Yadav  
**Domain:** Vendor contracts · amendments · invoices  
**Stack:** Python 3.11 · FastAPI · LangGraph · PostgreSQL + pgvector · React · MCP

---

## What This System Does 

Organizations run on piles of documents that describe the same reality and never quite agree.
This system owns such a pile end to end — across three movements:

**Understand** — Ingests mixed-format documents, classifies and chunks them, extracts
structured facts where every claim is grounded to an exact source quote, detects
disagreements between documents, and produces a deliverable register where every
section cites the facts it was rendered from.

**Examine** — Takes a user-supplied rule checklist and audits the extracted facts against
it stage by stage, producing findings that each point to an exact source location.
A clean corpus produces the rarest output in this industry: an honest report of no findings.

**Stay alive** — Watches a folder for new documents. Each arrival triggers a targeted
update — only sections whose source facts were touched by the new document are
recomputed. Untouched sections stay byte-identical, provably so via content hashes.
Contradictions from new sources are surfaced, never silently resolved.

A human gates all of it. Conflicts, findings, and updates are approved or rejected
item-by-item before they commit. The same gate is exposed as an MCP tool so a
machine can drive the entire flow programmatically.

---

## The Ten Behaviors — How Each Is Implemented

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | **Visible stages** | `Run.stage` updated at every node; `GET /runs/{id}` shows current stage and what the graph is waiting on |
| 2 | **Survives being stopped** | `PostgresSaver` checkpoints after every node; `POST /runs/{id}/resume` re-attaches with the same `thread_id`; no finished work is lost |
| 3 | **Human holds the gate** | `interrupt()` per finding in `human_gate` node; approve/reject are independent operations; rejecting finding N never touches N+1 |
| 4 | **Machine can drive it** | Full MCP server at `/mcp` mirrors every REST operation including approve/reject; `whoami` tool for connectivity check |
| 5 | **Never bluffs** | `_quote_appears_in()` grounding check on every extracted fact; ungrounded facts stored with `grounded=False`, excluded from deliverable; fabricated quotes get confidence capped at 0.2 |
| 6 | **Stranger can run it** | `docker compose up` starts DB + backend + frontend; schema created automatically; see Quick Start below |
| 7 | **Real tests, no live key** | 46 tests across 6 files; LLM mocked at `call_structured()` boundary; pure-unit tests never need a DB; DB tests skip cleanly if DB unreachable |
| 8 | **No prompt injection** | `looks_like_injected_instruction()` checked on every chunk and every extracted fact; injected text becomes a `critical` finding, never a command |
| 9 | **Concurrent runs stay isolated** | `pg_advisory_xact_lock(hashtext(pile_id))` serializes writes per pile; two runs on different piles never block each other |
| 10 | **Knows what it cost** | Every LLM call writes a `RunCost` row (stage, provider, tokens\_in, tokens\_out, latency\_ms); `GET /runs/{id}/cost` returns a stage-by-stage breakdown |

---

## Architecture

┌─────────────────────────────────────────────┐ │ Watched inbox (filesystem / watchdog) │ └──────────────────┬──────────────────────────┘ │ new file event ▼ ┌─────────────────────────────────────────────────────────────────┐ │ FastAPI (REST) + MCP server (same process, mounted at /mcp) │ │ │ │ POST /piles GET /runs/{id} │ │ POST /piles/{id}/documents POST /runs/{id}/resume │ │ GET /piles/{id}/deliverable POST /findings/{id}/approve │ │ GET /piles/{id}/deliverable/export │ │ GET /runs/{id}/cost POST /findings/{id}/reject │ └──────────────────┬──────────────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────┐ │ LangGraph StateGraph + PostgresSaver checkpointer │ │ │ │ classify_and_chunk │ │ → extract_facts │ │ → reconcile (scoped to touched claim_slots only) │ │ → examine_rules │ │ → human_gate (interrupt() per finding, resumable) │ │ → compose_deliverable │ │ → writes section_sources (dependency graph) │ │ → byte-identical check before any write │ └──────────────────┬──────────────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────┐ │ PostgreSQL 16 + pgvector │ │ │ │ piles documents chunks (+ embeddings) │ │ facts findings deliverable_sections │ │ section_sources runs run_costs │ │ approvals audit_log │ └─────────────────────────────────────────────────────────────────┘ ▼ ┌─────────────────────────────────────────────────────────────────┐ │ React review UI (Vite + TypeScript) │ │ Reads/writes the same REST endpoints as the MCP server │ │ Human and machine paths are provably identical │ └─────────────────────────────────────────────────────────────────┘

### Key design decisions

**Why LangGraph's `PostgresSaver` for resumability**  
Building crash-safe resumability from scratch is weeks of work and
easy to get subtly wrong. `PostgresSaver` checkpoints the full graph
state after every node. Kill the process mid-extraction, restart,
call `/runs/{id}/resume` with the same `thread_id` — the graph
continues at the exact node it was on. This turns behavior #2 from
a feature to write into a feature to configure.

**Why `interrupt()` per finding, not one big approval gate**  
The task requires rejecting finding N without discarding N+1 and N+2.
`interrupt()` is called once per pending finding inside a deterministic
`ORDER BY id` loop. Each interrupt has its own checkpoint. Pre-recorded
decisions (humans acting out of order) are consumed without blocking.
This makes the gate genuinely order-independent.

**Why `pg_advisory_xact_lock` for concurrency**  
Application-level locks race. The advisory lock is transaction-scoped:
acquired at the start of a write transaction, released automatically
at commit or rollback. Two concurrent requests on the same pile cannot
both pass through — the second blocks until the first commits. Two
requests on different piles take different lock keys and never touch
each other.

**Why structured output at the schema level, not prompt-level grounding**  
The deliverable is rendered from `Fact` rows, not from LLM-generated
prose. `ExtractionResult` requires an `evidence_quote` field. If the
quote cannot be found verbatim in the source chunk, the fact is marked
`grounded=False` at write time and excluded from the deliverable at
render time. Fabrication cannot reach the output — it is blocked at
the schema boundary, not caught after the fact.

**Why scoped reconciliation**  
A pile with 500 existing facts across 40 claim slots, fed one new
invoice that touches 2 slots, runs 2 LLM reconciliation calls, not 40.
`reconcile` queries which slots were touched by `document_ids` in the
current run's state, then restricts all subsequent queries to those
slots. The `section_sources` join table extends this to the deliverable:
only sections whose backing facts were touched get rewritten.

**Why `run_id == thread_id`**  
LangGraph's checkpoint is keyed by `thread_id`. The `Run` row is keyed
by `run_id`. Keeping them equal means a single value unlocks both the
DB row (for status queries) and the checkpoint (for resume). Diverging
them, as the original code accidentally did, breaks resume silently.

---

## Quick Start — Fresh Clone to Running in Minutes

### With Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/doctask-anjali-yadav
cd doctask-anjali-yadav

cp backend/.env.example backend/.env
# Edit backend/.env — add at least one of GEMINI_API_KEY or GROQ_API_KEY

docker compose up
That starts:

PostgreSQL 16 with pgvector on port 5432
FastAPI backend on port 8000 (schema created automatically)
React frontend on port 5173
Open http://localhost:5173 for the review UI.
API docs: http://localhost:8000/docs
MCP endpoint: http://localhost:8000/mcp

Without Docker (local dev)
Prerequisites: Python 3.11+, PostgreSQL 16 with pgvector, Node 20+

# 1. Database
createdb doctask
psql doctask -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add GEMINI_API_KEY and/or GROQ_API_KEY

python init_db.py               # creates all tables

uvicorn app.main:app --reload   # starts on port 8000

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev                     # starts on port 5173
Load the sample dataset
# From project root, backend venv active
python sample_data/scripts/load_sample_data.py
This loads 9 contract groups (base contract + amendment + invoice each) into 9 separate piles with a standard rule checklist. Each amendment deliberately contradicts its base contract on payment terms, termination notice, and governing law — guaranteed contradiction findings on every pile.

To load one specific group:

ONLY_STEM=Aimmune python sample_data/scripts/load_sample_data.py
To preview without calling the API:

DRY_RUN=1 python sample_data/scripts/load_sample_data.py
Running Tests
Tests run without a live LLM key. Pure unit tests run without a DB.

cd backend

# All tests (DB tests skip if DB unreachable)
python -m pytest tests/ -v

# With DB (point at your running instance)
$env:TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/doctask"
python -m pytest tests/ -v

# Unit tests only (never need DB or LLM key)
python -m pytest tests/ -v -k "not (db_engine or session)"

# One behavior at a time
python -m pytest tests/test_behaviour_8_injection.py -v
What the tests prove
File	Behaviors tested	DB needed
test_behaviour_2_resume.py	Checkpoint survives kill; no duplicate facts on resume	Yes
test_behaviour_3_human_gate.py	Reject one finding, others stay pending; idempotent decisions	Yes
test_behaviour_5_no_bluffing.py	Grounding check unit tests; ungrounded facts excluded from deliverable	Partial
test_behaviour_8_injection.py	All injection patterns detected; clean text not flagged; injected doc → finding not obedience	Partial
test_behaviour_9_concurrency.py	Advisory lock contract; concurrent attempts serialized; different piles don't interfere	Partial
test_cost.py	Every LLM call logged; cost aggregated by stage; runs isolated	Yes
test_idempotency.py	Duplicate doc rejected; chunk/fact/section writes idempotent	Yes
test_incremental_update.py	Untouched sections byte-identical; contradictions surfaced not resolved	Yes
test_api.py	Every endpoint happy path and error path; structured errors; 409 on double-decide	Yes
Inbox Watcher — Stay Alive Demo
Drop any .txt file into sample_data/inbox/ while the backend is running:

cp sample_data/contracts/base/VitalibisInc_*.txt sample_data/inbox/
The watcher detects the file within 1 second, creates a document row, and triggers a graph run. Check the run status:

curl http://localhost:8000/piles   # find the inbox pile_id
curl http://localhost:8000/findings/pile/{pile_id}
To route a file to a specific pile instead of the default inbox pile, create a pile_id.txt sidecar file in the inbox directory:

echo "your-pile-id-here" > sample_data/inbox/pile_id.txt
MCP — Machine Interface
Every operation is available as an MCP tool at http://localhost:8000/mcp.

# Example: full automated flow with no human clicking
tools = [
    "create_pile",
    "ingest_document",   # blocks until done/paused
    "list_pending_findings",
    "approve_finding",   # or reject_finding
    "get_run_cost",
]
Available tools:

Tool	What it does
whoami	Connectivity check; returns pile and run counts
create_pile	Create a pile with optional rule checklist
list_piles	List all piles
ingest_document	Upload doc + run graph synchronously; returns terminal status
get_run_status	Stage, status, and current pending interrupt
resume_run	Re-attach after server restart (behavior #2)
list_pending_findings	All findings with status for a pile
approve_finding	Approve; resumes graph if it was waiting on this finding
reject_finding	Reject; same resumption semantics
get_run_cost	Token/latency/cost breakdown by stage and provider
Supported Formats and Domains
Document formats accepted: .txt, .md (plain text).
PDF and DOCX ingestion is architecturally wired (_read_document_text in nodes.py is the single seam) but the format-specific parsers are not yet implemented. All 27 sample documents are stored as plain text.

Domain: Vendor contracts with amendments and invoices.
The system is domain-agnostic at the architecture level — claim slots (payment_terms, governing_law, termination_notice, etc.) are extracted by the LLM, not hardcoded. A second domain requires only different sample documents and a different rule checklist. No code changes.

A second run with different documents means: any .txt files representing contracts, amendments, or invoices not in the sample set. The system declares no domain-specific assumptions in code.

LLM Providers
The system uses two free-tier providers with automatic fallback:

Provider	Model	Free tier
Google Gemini	gemini-2.5-flash	Yes — aistudio.google.com
Groq	llama-3.3-70b-versatile	Yes — console.groq.com
At least one key is required. Both is recommended — the system falls back automatically on rate-limit or outage with exponential backoff (up to 3 retries). A hard ceiling of 60 LLM calls per run prevents runaway loops from exhausting daily quotas.

Embeddings use sentence-transformers/all-MiniLM-L6-v2 (local, ~90MB download on first run, no API key needed).

Project Structure
doctask-anjali-yadav/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, lifespan, CORS, MCP mount
│   │   ├── config.py            # Settings from env, psycopg conninfo
│   │   ├── locking.py           # Advisory lock + active-run check
│   │   ├── mcp_server.py        # MCP tools mirroring all REST routes
│   │   ├── watcher.py           # Watchdog inbox watcher
│   │   ├── db/
│   │   │   ├── models.py        # All SQLAlchemy models
│   │   │   └── session.py       # Engine, SessionLocal, get_session
│   │   ├── graph/
│   │   │   ├── graph.py         # StateGraph + PostgresSaver wiring
│   │   │   ├── nodes.py         # All 6 node functions
│   │   │   ├── llm.py           # Two-provider LLM caller + cost logging
│   │   │   ├── schemas.py       # Structured output schemas
│   │   │   └── state.py         # PileState + injection pattern detection
│   │   └── routes/
│   │       ├── piles.py         # CRUD for piles
│   │       ├── ingest.py        # Document upload + background run trigger
│   │       ├── runs.py          # Run status + resume
│   │       ├── findings.py      # Approve/reject with order-independence
│   │       ├── deliverable.py   # JSON + markdown export
│   │       └── cost.py          # Stage-by-stage cost report
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_behaviour_2_resume.py
│   │   ├── test_behaviour_3_human_gate.py
│   │   ├── test_behaviour_5_no_bluffing.py
│   │   ├── test_behaviour_8_injection.py
│   │   ├── test_behaviour_9_concurrency.py
│   │   ├── test_cost.py
│   │   ├── test_idempotency.py
│   │   ├── test_incremental_update.py
│   │   └── test_api.py
│   ├── alembic/                 # Migration environment
│   ├── init_db.py               # Schema bootstrap (pgvector + metadata)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   └── App.tsx              # Review UI (approve/reject findings)
│   ├── Dockerfile
│   └── package.json
├── sample_data/
│   ├── contracts/
│   │   ├── base/                # 19 CUAD contracts (plain text)
│   │   ├── amendments/          # 9 LLM-generated amendments (deliberate contradictions)
│   │   └── invoices/            # 9 LLM-generated invoices (reference amended terms)
│   ├── inbox/                   # Watched folder — drop files here
│   └── scripts/
│       ├── pick_contracts.py    # Selects 20 best contracts from CUAD dataset
│       ├── generate_amendments.py # Generates amendments + invoices via Gemini
│       └── load_sample_data.py  # Loads all groups into the API
├── docker-compose.yml
└── README.md
Honest Limitations
What works reliably

Full pipeline on plain-text documents end to end
Crash-safe resumability via PostgresSaver checkpoints
Order-independent human gate with pre-recorded decisions
Prompt injection detection on all 8 tested patterns
Byte-identical untouched section proof via content hashes
Scoped reconciliation (only touched slots re-checked)
Cost logging on every LLM call, by stage and provider
MCP server exposing all operations including approval gate
Known simplifications (documented, not hidden)

PDF/DOCX parsing not implemented. _read_document_text() in nodes.py is the single seam where format-specific parsers plug in. All sample documents are stored as plain text. Real PDF extraction (pdfplumber, pymupdf) is the next addition.
Paragraph splitter is naive. Documents are split on \n\n. A structure-aware splitter that respects headings and section numbers would improve extraction quality on long contracts.
LLM call budget is process-local. _call_counts in llm.py is a dict in process memory. A Redis counter would survive restarts. Safe for this scale; noted for production hardening.
Inbox pile assignment is simple. The watcher assigns new files to the first existing pile unless a pile_id.txt sidecar is present. Production would use subdirectory-per-pile or a metadata file convention.
Frontend is functional, not polished. The review UI covers the approve/reject flow. Design polish is deferred — the API and agent behaviors are the submission's primary surface.
No authentication. The API has no auth layer. Adding OAuth2 or API key middleware to FastAPI is one decorator; deferred for clarity.
What the tests do not cover

The full LangGraph checkpoint + resume cycle requires a live DB. Tests that need it are skipped cleanly when the DB is unreachable — they do not silently pass.
The MCP server's HTTP transport is not integration-tested. The tool functions themselves are tested via the shared route functions they call.
Assumptions Logged During Build
run_id == thread_id — LangGraph checkpoints by thread_id. Keeping it equal to run_id means one value unlocks both the DB row and the checkpoint. Diverging them breaks resume silently.

Advisory lock released before graph starts — The pile lock is held only for the duration of the "is there an active run?" check, then released. The graph runs its own per-node sessions. Holding the lock for a full multi-minute LLM run would starve other pile operations.

Contradiction finding = claim_slot scoped — Two facts contradict each other within a slot, not globally. A payment_terms fact and a governing_law fact cannot contradict each other by definition.

approved contradiction stays visible — Approving a contradiction finding means "yes, this conflict is real." The deliverable keeps the sources-disagree banner. Only a rejected contradiction finding dismisses the banner. This matches the task's "surfaced, not silently resolved" requirement.

Watcher creates a default pile if none exists — The first pile in the DB is used. If none exists, one named "inbox" is created with an empty rule checklist. This is the right default for the watcher's "stay alive" role — it should never drop a file because no pile was manually created first.

Data Sources
Base contracts are drawn from the CUAD dataset (public domain legal contracts). Amendments and invoices are synthetically generated using Gemini 2.0 Flash with deliberate contradictions introduced for demonstration purposes. No real confidential or NDA-bound material is included.

Built for the SuperDocs Engineering Task
This repository is the Task 1 submission for the SuperDocs Full-Stack AI Engineer round. The system is built entirely by Anjali Yadav using Claude as the primary coding assistant (~85% AI-assisted, fully directed and reviewed).

Credit line: built for the SuperDocs engineering task, superdocs.app.



