# SuperDocs Round 2 — Full Project Context

## Who's building this
Anjali (Shally) — AI/ML engineer, applying for SuperDocs' Full-Stack AI Engineer role.
Deadline: Thu 20 Aug, 11:59pm IST. No paid subscriptions to any tool/API/AI — everything
in this build must run on free tiers or fully local/self-hosted components.

## The two deliverables (SEPARATE repos, do not merge)

### Task 1 — standalone agentic document-pile system
- Own PRIVATE GitHub repo: `doctask-anjali-shally`. Invite `github.com/o-kadam` as collaborator.
- Does NOT use the SuperDocs API at all. Fully independent system.
- Required stack: Python + FastAPI, LangGraph (orchestration), PostgreSQL + pgvector
  (retrieval), React (review UI). MCP server for the machine-driven interface.

### Task 2 — builds ON the SuperDocs API/MCP surface
- PR into PUBLIC `github.com/superdocsapp/superdocs-builds`, folder `use-cases/anjali-shally/`.
- Assigned build 1: **Accessible-document standard kit** (S3) — messy doc → accessibility
  standard: heading hierarchy, list semantics, table headers, link text, image descriptions,
  language attributes. Output: before/after structure report (generated from the document,
  not hand-written) + tagged PDF + Word file. State what's automatic vs. needs a human.
- Assigned build 2: **HACCP / food-safety-plan drafter** (S2) — chat-driven drafter onto a
  structured HACCP template (hazard analysis, critical control points, monitoring, corrective
  actions), versioned as the process evolves. Output is always a draft, never certified —
  keep a "for professional review" note in every export.
- Minimum contract with SuperDocs, both builds: upload → chat → approve → export (REST or MCP,
  the MCP server exposes the same four as tools).
- Known gotchas from the task doc: proposed-change content arrives as a JSON-encoded STRING,
  needs a second `json.loads` — the #1 cause of "empty diff card" bugs. Large-document / deep
  model ops can take 30s–several minutes with no visible progress — that's normal, not a crash.
  Budget operations: any loop/measurement needs a small-sample mode + stopping rule. Exports
  and downloads never cost operations; only searches/edits do.
- Get an API key by having a coding agent self-signup: `POST https://api.superdocs.app/v1/agents/signup`
  — already proven working. Promo code `BUILDER26` gives 1 month Pro / 10,000 ops (redeem at
  use.superdocs.app > Settings > Billing, expires end of August).
- Bugs found and reported in SuperDocs earn points on this task (email hello@superdocs.app or
  in-app bug button) — separate from and in addition to the Task 2 builds themselves.

## Task 1 — what "done" means (non-negotiable floor, behaviors 1–5)
1. Visible, watchable stages; some decisions can change the path (retry/skip/escalate). Not a
   fixed script, not one model call in a UI.
2. Survives being killed mid-run and resumed — no lost work, no duplicated work.
3. Human gate: approve/reject item-by-item before commit. Rejecting one item doesn't discard
   the rest.
4. Fully machine-drivable end to end, no human clicking required — approval is a callable
   operation (MCP tool / REST endpoint), whoever calls it (human or program) makes the call
   explicitly.
5. Never bluffs — if sources don't support a claim, it says so. A success message only ever
   means the output is genuinely in the state it claims.

Behaviors 6–10 (nice-to-have, cut here first if short on time): easy fresh-clone setup;
real tests that run with NO live key (kill/resume, concurrent runs, prompt-injection resistance
— tests that only prove mocks work don't count); ignores instructions embedded in source
documents (treats them as data to report on); two concurrent runs don't corrupt state; reports
cost and time per stage.

## Domain & dataset
Vendor contracts + amendments + invoices, built on **CUAD (Contract Understanding Atticus
Dataset)** — free, CC BY 4.0, 510 real commercial contracts with expert clause annotations.
Kaggle: `theatticusproject/atticus-open-contract-dataset-aok-beta` (or the pre-processed
`changethan/legal-contract-dataset`). Use ~15–20 contracts as the base pile. Generate
amendments/invoices yourself with the free LLM, deliberately contradicting 3–4 base clauses
(payment terms, renewal dates) — this is what makes the "documents disagree" and "conflict
surfaced, not silently resolved" requirements demoable rather than hypothetical.

## Free-tier stack (zero cost, by design)
- **LLM, primary:** Gemini 2.5 Flash via Google AI Studio — 1,500 req/day, no card, strong
  structured-output support (needed for the Fact/Finding schemas).
- **LLM, fallback:** Groq, Llama 3.3 70B — separate free quota, OpenAI-compatible API, fast;
  use for high-volume extraction so Gemini's daily cap isn't the bottleneck.
- **Embeddings:** local `sentence-transformers` — zero cost, zero rate limit, works offline,
  satisfies "tests run without a live key" for anything retrieval-related.
- **Vector DB / DB:** self-hosted Postgres + pgvector via docker-compose — zero cost, satisfies
  "stranger can run it... one documented command."
- **Resumability:** `langgraph.checkpoint.postgres.PostgresSaver` — do not hand-roll a
  checkpoint system, this is what makes "survives being killed" a config choice, not a feature
  you write.
- **Human gate:** LangGraph `interrupt()`, one interrupt point per item (finding/update), keyed
  by an `approvals` table — never one big approve-the-whole-run gate.
- **Concurrency:** Postgres advisory locks (`pg_advisory_xact_lock(hash(pile_id))`), scoped per
  pile, not global.

## Postgres schema (Task 1)
```sql
piles(id, name, rules_json, created_at)
documents(id, pile_id, filename, format, uploaded_at, content_hash)
chunks(id, document_id, text, embedding vector(384), char_start, char_end)
facts(id, chunk_id, claim, confidence, extracted_at)
deliverable_sections(id, pile_id, title, rendered_text, content_hash)
section_sources(section_id, fact_id)          -- dependency graph for incremental updates
findings(id, pile_id, rule_id, description, source_chunk_id, status)
runs(id, pile_id, thread_id, stage, status, started_at, finished_at)
run_costs(id, run_id, stage, tokens_in, tokens_out, cost_usd)
approvals(id, item_type, item_id, run_id, status, decided_at)
audit_log(id, pile_id, event, source_document_id, diff, occurred_at)
```
Note: embedding dimension is 384 for `sentence-transformers/all-MiniLM-L6-v2` (the standard
free local embedding model), not 1536 (that's an OpenAI dimension — don't copy paste that
number from generic examples).

## Design rules to hold to throughout
- Provenance is structural, not narrative: the model never free-writes the deliverable. It
  outputs `Fact(claim, source_doc_id, source_chunk_id, char_range, confidence)` rows; the
  deliverable is RENDERED from those rows. A claim with no backing Fact row cannot render —
  this is also the anti-bluffing mechanism, enforced at the schema level, not by prompting.
- Prompt-injection defense: wrap all extracted document text in a delimited block
  (`<source_document>...</source_document>`) with a system instruction that content inside is
  data, never instructions. Include one adversarial test document with a fake embedded
  instruction; the system must report it as a finding, never obey it.
- Cost tracking: one decorator around every LLM call, logging run_id/stage/tokens/latency/cost
  to `run_costs`. Apply it everywhere from day one, don't bolt it on later.
- Every write transaction touching a pile takes the advisory lock for that pile_id first.

## Architecture (Task 1)
```
Watched inbox (filesystem watcher)
        -> FastAPI + MCP server (same app; MCP tools mirror REST routes 1:1)
        -> LangGraph orchestrator, PostgresSaver-backed
             classify -> extract -> reconcile -> [HUMAN GATE] -> compose_deliverable
             -> examine_rules -> [HUMAN GATE] -> render
             -> (new file) diff_affected_sections -> targeted_update -> [HUMAN GATE] -> commit
        -> PostgreSQL + pgvector (schema above)
        -> React review UI (reads/writes the same REST endpoints the MCP server exposes)
```

## Build order (dependency order, not day-boxed — move through it as fast as correctness allows)
1. docker-compose (Postgres+pgvector) up and confirmed working.
2. Schema + Alembic migration.
3. FastAPI + LangGraph skeleton wired to PostgresSaver, nodes are no-op stubs.
4. Prove resumability on the stubs (kill -9, restart, confirm resume) BEFORE writing any LLM logic.
5. Pull CUAD contracts, build classify/extract prompts (Gemini primary, Groq fallback),
   structured Fact output.
6. Generate contradicting amendments/invoices, wire reconcile (contradiction = same claim-slot,
   different value, across Fact rows for the same pile).
7. Human gate: interrupt() + approvals table + approve/reject endpoints, item-by-item.
8. examine_rules node against a small (5-6 rule) checklist.
9. Deliverable rendering strictly from Fact/Finding rows.
10. Watcher + incremental update (content-hash sections, only re-run affected ones) + advisory
    locks + cost decorator.
11. The required tests, mocking LLM calls so they run with no live key: kill/resume,
    concurrent runs on same pile, injection resistance, incremental-update isolation
    (untouched sections keep an identical content hash).
12. MCP server mirroring REST routes.
13. React review UI last, once the REST/MCP contract is stable.

## Testing standard (quoted from the task doc, this is graded)
Real tests, no live key required; a stranger can go from clone to working in minutes with one
documented command; error handling names the cause AND the fix; idempotency wherever an
operation costs money; large files go through the upload path, not held in memory; no key ever
in a log, commit, or shell history; graceful degradation when a model call or dependency fails.
