import { useState } from "react";
import { api, ApiError } from "../api/client";

/**
 * Handles the ingest response defensively: some backends return
 * {run_id, status:"processing"} to poll (see RunTracker in ReviewTab),
 * others may return a finished result synchronously per the MCP tool
 * table ("returns terminal status"). Either shape is handled here.
 */
export function IntakeTab({ pileId, onRunStarted }) {
  const [filename, setFilename] = useState("");
  const [text, setText] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setNotice(null);
    if (!filename.trim()) { setError("Give it a filename."); return; }
    if (!text.trim()) { setError("Paste the document text."); return; }

    setSubmitting(true);
    try {
      const result = await api.ingestDocument(pileId, filename.trim(), text);

      if (result.status === "duplicate_skipped") {
        setNotice("Byte-identical to a document already in this pile — nothing new to process.");
      } else if (result.run_id && (result.status === "processing" || !result.status)) {
        setText("");
        setFilename("");
        onRunStarted(result.run_id);
      } else {
        setText("");
        setFilename("");
        onRunStarted(result.run_id || null, result);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Intake failed — check the API is reachable.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card">
      <h2>File a document</h2>
      <div className="card-sub">Contracts, amendments, invoices — anything belonging to this pile.</div>
      {error && <div className="error-banner">{error}</div>}
      {notice && <div className="info-banner">{notice}</div>}
      <form onSubmit={handleSubmit}>
        <div className="field">
          <label>Filename</label>
          <input value={filename} onChange={(e) => setFilename(e.target.value)} placeholder="amendment-2026-03.txt" />
          <div className="field-hint">Plain text only for now: .txt or .md</div>
        </div>
        <div className="field">
          <label>Document text</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the document's contents…" />
        </div>
        <button type="submit" disabled={submitting}>{submitting ? "Filing…" : "File into pile"}</button>
      </form>
    </div>
  );
}