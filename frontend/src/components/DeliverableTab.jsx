import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";

/**
 * The backend renders each section as markdown text with a fixed shape
 * (see app/graph/nodes.py::_render_section):
 *
 *   ## Section Title
 *
 *   **⚠ Sources disagree on this point — human review required:**
 *
 *   - {claim} (source: chunk `{chunk_id}`, confidence {conf})
 *
 * Parsing it here (rather than asking the backend for structured JSON)
 * keeps the deliverable endpoint a plain read of what compose_deliverable
 * already wrote, while letting the UI give every claim its own exhibit
 * tag instead of one generic label bolted onto the whole section — which
 * is what this used to do.
 */
const FACT_LINE = /^-\s+(.*?)\s+\(source: chunk `([^`]+)`, confidence ([\d.]+)\)\s*$/;

function parseSection(renderedText) {
  const body = renderedText.replace(/^##.*\n\n?/, "");
  const lines = body.split("\n").map((l) => l.trim()).filter(Boolean);

  const hasConflict = lines.some((l) => l.startsWith("**⚠"));
  const facts = [];

  for (const line of lines) {
    const match = line.match(FACT_LINE);
    if (match) {
      facts.push({ claim: match[1], chunkId: match[2], confidence: match[3] });
    }
  }
  return { hasConflict, facts };
}

export function DeliverableTab({ pileId, refreshKey }) {
  const [sections, setSections] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.getDeliverable(pileId)
      .then((data) => !cancelled && setSections(data.sections || []))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Couldn't load the deliverable."))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [pileId, refreshKey]);

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>Deliverable</h2>
        <a href={api.exportDeliverableUrl(pileId)} target="_blank" rel="noreferrer">
          <button className="secondary small">Export</button>
        </a>
      </div>
      <div className="card-sub">{sections.length} section{sections.length === 1 ? "" : "s"} · every line traces to its source</div>
      {error && <div className="error-banner">{error}</div>}
      {loading ? (
        <div className="empty-state"><span className="spinner" />composing…</div>
      ) : sections.length === 0 ? (
        <div className="empty-state">Nothing composed yet — file a document to get started.</div>
      ) : (
        sections.map((s) => {
          const { hasConflict, facts } = parseSection(s.rendered_text);
          return (
            <div className="deliverable-section" key={s.title}>
              <div className="ds-head">
                <h3>{s.title.replace(/_/g, " ")}</h3>
              </div>

              {hasConflict && (
                <div className="conflict-banner">⚠ Sources disagree here — human review required</div>
              )}

              {facts.length > 0 ? (
                facts.map((f, i) => (
                  <div className="fact-line" key={i}>
                    <span className="fact-claim">{f.claim}</span>
                    <span className="exhibit-tag">
                      chunk {f.chunkId.slice(0, 8)} · conf {f.confidence}
                    </span>
                  </div>
                ))
              ) : (
                <div className="fact-line"><span className="fact-claim">{s.rendered_text.replace(/^##.*\n\n?/, "")}</span></div>
              )}

              <div className="hash">sha256:{s.content_hash.slice(0, 16)}… · updated {new Date(s.updated_at).toLocaleString()}</div>
            </div>
          );
        })
      )}
    </div>
  );
}
