# DocTask: Agentic Document Pile Analyst

**Author:** Anjali Yadav
**Domain:** Vendor contracts · amendments · invoices
**Stack:** Python 3.11 · FastAPI · LangGraph · PostgreSQL + pgvector · React · MCP

Built for the SuperDocs Full-Stack AI Engineer task.

---

## Contents

- [What This System Does](#what-this-system-does)
- [The Ten Required Behaviors](#the-ten-required-behaviors)
- [Architecture](#architecture)
- [Key Design Decisions](#key-design-decisions)
- [Quick Start](#quick-start)
- [Sample Dataset](#sample-dataset)
- [Inbox Watcher — "Stay Alive" Demo](#inbox-watcher--stay-alive-demo)
- [MCP Interface](#mcp-interface)
- [REST API](#rest-api)
- [Testing](#testing)
- [Supported Formats & Domain](#supported-formats--domain)
- [LLM Providers](#llm-providers)
- [Project Structure](#project-structure)
- [Security & Configuration](#security--configuration)
- [Honest Limitations](#honest-limitations)
- [Operational Assumptions](#operational-assumptions)
- [Data Sources](#data-sources)

---

### What This System Does

DocTask owns a pile of related documents end to end, documents that describe the same business relationship but don't always agree with each other.

**Understand**
Ingests mixed-format documents, classifies and chunks them, extracts structured facts, and grounds every fact to an exact source quote. Contradictions between documents are detected and the resulting deliverable is fully traceable to its sources.

**Examine**
Takes a user-supplied rule checklist and audits extracted facts against it, producing findings that stay pending until a human approves or rejects them. A clean corpus produces an honest report of zero findings.

**Stay alive**
Watches an inbox directory for new documents. Each arrival triggers a targeted update, only the sections whose source facts were actually affected are recomputed. Untouched sections stay byte-identical. New contradictions are surfaced, never silently resolved.

A human gates all of it, and the same operations are exposed through MCP so a machine can drive the entire workflow without a person clicking through the UI.

---

### The Ten Required Behaviors

| # | Requirement | Implementation |
|---|---|---|
| 1 | Visible stages | `Run.stage` updates at every graph node; `GET /runs/{id}` exposes current stage and pending work |
| 2 | Survives being stopped | LangGraph `PostgresSaver` checkpoints after every node; `POST /runs/{id}/resume` continues from the same `thread_id` |
| 3 | Human holds the gate | `interrupt()` fires once per finding in `human_gate`; approve/reject are independent operations |
| 4 | Machine can drive it | MCP server mounted at `/mcp` mirrors every REST operation, including approve/reject |
| 5 | Never bluffs | Every extracted fact is checked against its source quote; ungrounded facts are excluded from the deliverable, fabricated evidence is confidence-capped |
| 6 | Stranger can run it | `docker compose up` starts DB, backend, and frontend with automatic schema init |
| 7 | Real tests, no live key | LLM boundary is mocked; pure unit tests need no DB; DB-dependent tests skip cleanly when unreachable |
| 8 | No prompt injection | Chunks and facts are checked for injection patterns; detected instructions become findings, never commands |
| 9 | Concurrent runs stay isolated | PostgreSQL advisory locks serialize writes per pile; independent piles never block each other |
| 10 | Knows what it cost | Every LLM call logs stage, provider, tokens, and latency; `GET /runs/{id}/cost` gives a stage-by-stage breakdown |

---

## Architecture

```mermaid
flowchart TB
    inbox["Watched inbox"] --> watcher["Filesystem watcher"]
    frontend["React review UI\n(Intake · Review · Deliverable · Cost)"] -->|"REST API"| api

    subgraph api["FastAPI application"]
        rest["REST API"]
        mcp["MCP server"]
    end

    watcher --> rest

    subgraph workflow["LangGraph workflow — checkpointed with PostgresSaver"]
        direction LR
        classify["classify\n& chunk"] --> extract["extract\nfacts"] --> reconcile["reconcile\nclaim slots"] --> rules["examine\nrules"] --> gate["human\ngate"] --> compose["compose\ndeliverable"] --> track["track section\nsources"]
    end

    rest --> workflow
    mcp --> workflow

    db[("PostgreSQL + pgvector\n(piles · documents · chunks · facts\nfindings · sections · runs · costs · audit)")]

    api <--> db
    workflow <--> db

    classDef input fill:#e8f1ff,stroke:#2563eb,color:#111827
    classDef service fill:#ecfdf5,stroke:#059669,color:#111827
    classDef process fill:#fff7ed,stroke:#ea580c,color:#111827
    classDef data fill:#f5f3ff,stroke:#7c3aed,color:#111827
    classDef ui fill:#fdf2f8,stroke:#db2777,color:#111827

    class inbox,watcher input
    class rest,mcp service
    class classify,extract,reconcile,rules,gate,compose,track process
    class db data
    class frontend ui
```

---

### Key Design Decisions

**LangGraph `PostgresSaver` for resumability**
Checkpoints graph state after every node. `run_id == thread_id`, so the database run record and the LangGraph checkpoint are addressable through one identifier. If the backend stops mid-run, `POST /runs/{id}/resume` continues from the last checkpoint instead of restarting completed work.

**One human interrupt per finding**
`human_gate` processes pending findings in a deterministic, ordered loop, and each finding gets its own `interrupt()` checkpoint. Approving or rejecting one finding never touches the others, and pre-recorded decisions are consumed cleanly when the graph reaches them.

**PostgreSQL advisory locks for concurrency**
Pile-level writes take a transaction-scoped advisory lock keyed on the pile ID: serialized within a pile, no blocking across piles, automatic release on commit or rollback.

**Schema-level evidence grounding**
Facts are structured rows, not generated prose. Every extraction carries an `evidence_quote`, verified against the source chunk before the fact is marked `grounded=True`. Ungrounded facts never reach the deliverable.

**Scoped reconciliation**
A new document only triggers reconciliation for the claim slots it actually touches, and `section_sources` extends that dependency tracking to the deliverable — only affected sections are recomputed.

---

## Quick Start

### Prerequisites

- **Docker:** Docker Desktop, Git
- **Local dev:** Python 3.11+, PostgreSQL 16 with pgvector, Node.js 20+, npm
- At least one LLM provider key (Gemini or Groq) for real processing

### Docker (recommended)

```bash
git clone https://github.com/anjaliy11/doctask-Anjali-Yadav.git
cd doctask-Anjali-Yadav

cp backend/.env.example backend/.env   # Windows: Copy-Item backend/.env.example backend/.env
# edit backend/.env — set GEMINI_API_KEY and/or GROQ_API_KEY

docker compose up
```

| Service | URL |
|---|---|
| Backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MCP endpoint | http://localhost:8000/mcp |
| Frontend | http://localhost:5173 |
| PostgreSQL | localhost:5432 |

### Local development

```bash
# 1. Database
createdb doctask
psql doctask -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # Windows: Copy-Item .env.example .env
# edit .env with your LLM key(s)

python init_db.py
uvicorn app.main:app --reload

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

### Sample Dataset

The repo ships sample vendor-contract data base contracts, amendments, and invoices, where each amendment deliberately contradicts its base contract, guaranteeing contradiction findings out of the box.

```bash
python sample_data/scripts/load_sample_data.py         # load everything
ONLY_STEM=Aimmune python sample_data/scripts/load_sample_data.py   # one group only
DRY_RUN=1 python sample_data/scripts/load_sample_data.py           # preview, no API calls
```

### Inbox Watcher — "Stay Alive" Demo

With the backend running, drop a document into the watched inbox:

```bash
cp sample_data/contracts/base/VitalibisInc_*.txt sample_data/inbox/
```

The watcher creates the document record and triggers processing automatically.

```bash
curl http://localhost:8000/piles
curl http://localhost:8000/findings/pile/{pile_id}
```

To route a file to a specific pile, add a sidecar file:

```bash
echo "your-pile-id-here" > sample_data/inbox/pile_id.txt
```

---

### MCP Interface

Available at `http://localhost:8000/mcp`, mirroring the full REST workflow.

| Tool | Purpose |
|---|---|
| `whoami` | Connectivity check, basic counts |
| `create_pile` | Create a pile with an optional rule checklist |
| `list_piles` | List existing piles |
| `ingest_document` | Upload a document and run the graph |
| `get_run_status` | Stage, status, pending interrupt |
| `resume_run` | Resume an interrupted or stopped run |
| `list_pending_findings` | Findings and their statuses |
| `approve_finding` / `reject_finding` | Decide a finding |
| `get_run_cost` | Token, latency, and cost breakdown |

Typical machine-driven flow:

```
create_pile → ingest_document → get_run_status → list_pending_findings
→ approve_finding / reject_finding → resume_run → get_run_cost
```

### REST API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/piles` | Create a pile |
| GET | `/piles` | List piles |
| POST | `/piles/{id}/documents` | Ingest a document |
| GET | `/runs/{id}` | Get run status |
| POST | `/runs/{id}/resume` | Resume a run |
| GET | `/findings/pile/{pile_id}` | List findings |
| POST | `/findings/{id}/approve` | Approve a finding |
| POST | `/findings/{id}/reject` | Reject a finding |
| GET | `/piles/{id}/deliverable` | Get the deliverable |
| GET | `/piles/{id}/deliverable/export` | Export as markdown |
| GET | `/runs/{id}/cost` | Get run cost breakdown |

Interactive docs: `http://localhost:8000/docs`

---

### Testing

No live LLM key is required to run the suite.

```bash
cd backend
python -m pytest tests/ -v
```

DB-dependent tests need a reachable PostgreSQL instance and skip cleanly if one isn't found.

```bash
# Windows PowerShell, explicit test DB
$env:TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/doctask"
python -m pytest tests/ -v

# Unit tests only, no DB required
python -m pytest tests/ -v -k "not (db_engine or session)"

# One behavior at a time
python -m pytest tests/test_behaviour_8_injection.py -v
```

| Test file | Coverage |
|---|---|
| `test_behaviour_2_resume.py` | Checkpoint/resume, no duplicate work |
| `test_behaviour_3_human_gate.py` | Independent approve/reject, idempotency |
| `test_behaviour_5_no_bluffing.py` | Evidence grounding, deliverable exclusion |
| `test_behaviour_8_injection.py` | Prompt-injection detection and safe handling |
| `test_behaviour_9_concurrency.py` | Advisory-lock behavior, run isolation |
| `test_cost.py` | LLM call logging, stage-level cost aggregation |
| `test_idempotency.py` | Duplicate document and write prevention |
| `test_incremental_update.py` | Incremental recomputation, byte-identical sections |
| `test_api.py` | REST endpoint success and error paths |

---

### Supported Formats & Domain

**Formats:** `.txt`, `.md`. PDF and DOCX are not yet implemented — `backend/app/formats.py` is the single seam where format-specific parsers plug in.

**Domain:** vendor contracts, amendments, and invoices. Claim slots (`payment_terms`, `governing_law`, `termination_notice`, etc.) are extracted by the LLM schema rather than hardcoded, so a different document domain needs only different documents and a different rule checklist — no code changes.

### LLM Providers

| Provider | Model | Env var |
|---|---|---|
| Google Gemini | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |

At least one key is required; with both configured, the app falls back between providers on failure or rate limit with retry/backoff. A hard per-run LLM call ceiling prevents runaway processing. Embeddings use `sentence-transformers/all-MiniLM-L6-v2` locally, no extra API key needed.

---

## Project Structure

```
doctask-Anjali-Yadav/
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, formats.py, locking.py
│   │   ├── mcp_server.py, watcher.py
│   │   ├── db/            (models.py, session.py)
│   │   ├── graph/         (graph.py, llm.py, nodes.py, schemas.py, state.py)
│   │   └── routes/        (piles.py, ingest.py, runs.py, findings.py, deliverable.py, cost.py)
│   ├── tests/
│   ├── alembic/
│   ├── init_db.py, requirements.txt, Dockerfile, .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx, api/client.js
│   │   └── components/    (IntakeTab, ReviewTab, DeliverableTab, CostTab, Sidebar)
├── sample_data/
│   ├── contracts/         (base, amendments, invoices)
│   ├── inbox/
│   └── scripts/           (pick_contracts.py, generate_amendments.py, load_sample_data.py)
├── docker-compose.yml
└── README.md
```

---

## Security & Configuration

Secrets stay out of version control. `backend/.env.example` documents required variables; `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `node_modules/`, `dist/`, `.pytest_cache/`, `.mypy_cache/`, and `sample_data/raw/`. Never commit real API keys, database passwords, or other credentials.

---

## Honest Limitations

**What works:** end-to-end pipeline on plain-text documents, crash-safe resumability, independent human approval/rejection, prompt-injection detection, evidence grounding, byte-identical preservation of untouched sections, scoped reconciliation, per-call cost/latency logging, MCP interface covering the full workflow, REST API, and a functional React review UI.

**Known simplifications:**
- **PDF/DOCX parsing** isn't implemented — `formats.py` is the seam for adding it.
- **Paragraph splitting** is a naive `\n\n` split; a structure-aware splitter would help on long, heavily-sectioned contracts.
- **LLM call budget** is tracked in process memory; a distributed deployment would need a shared counter (e.g. Redis).
- **Inbox pile assignment** defaults to a single pile unless a `pile_id.txt` sidecar is present; production would want subdirectory- or metadata-based routing.
- **No authentication layer** — would be added at the FastAPI middleware boundary for production.
- **Test boundaries:** the full checkpoint/resume path needs a live PostgreSQL instance and skips (not silently passes) when one isn't reachable; the MCP HTTP transport itself isn't integration-tested, only the shared operations it calls.

## Operational Assumptions

- **`run_id == thread_id`** — keeps the database run record and the LangGraph checkpoint addressable through one identifier.
- **Advisory lock scope** — the pile lock protects the active-run/write check only, not the full multi-minute LLM workflow.
- **Contradictions are claim-slot scoped** — `payment_terms ↔ payment_terms` can conflict; `payment_terms ↔ governing_law` cannot.
- **Approved contradictions stay visible** — approving a contradiction finding confirms the conflict; the deliverable keeps showing it. Only rejecting the finding clears it.
- **Default inbox pile** — an unassigned document is never silently dropped; it goes to a default pile if none is specified.

## Data Sources

Kaggle CUAD dataset used.
Base contracts are drawn from the CUAD dataset. Amendments and invoices are synthetically generated for demonstration, with contradictions deliberately introduced. No confidential or NDA-bound material is included.

---

*Built for the SuperDocs Full-Stack AI Engineer task — LangGraph orchestration, persistent checkpoints, human-in-the-loop approval, evidence-grounded extraction, prompt-injection handling, incremental updates, concurrency control, cost tracking, MCP, REST, and a React review interface.*
