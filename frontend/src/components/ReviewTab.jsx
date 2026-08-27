import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";

const POLL_MS = 1500;

function severityClass(interrupt) {
  if (interrupt.rule_id === "contradiction") return "contradiction";
  return interrupt.severity || "";
}

function RunTracker({ runId, onSettled }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [deciding, setDeciding] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function poll() {
      try {
        const s = await api.getRun(runId);
        if (cancelled) return;
        setStatus(s);
        setError(null);
        if (s.status === "done" || s.status === "failed") {
          onSettled?.(runId, s.status);
          return;
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "Lost contact with the API.");
      }
      timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [runId, onSettled]);

  async function decide(decision) {
    if (!status?.pending_interrupt) return;
    setDeciding(true);
    setError(null);
    try {
      const fn = decision === "approved" ? api.approveFinding : api.rejectFinding;
      await fn(status.pending_interrupt.item_id);
      const fresh = await api.getRun(runId);
      setStatus(fresh);
      if (fresh.status === "done" || fresh.status === "failed") onSettled?.(runId, fresh.status);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Decision didn't go through — refreshing…");
      try { setStatus(await api.getRun(runId)); } catch { /* keep prior status */ }
    } finally {
      setDeciding(false);
    }
  }

  if (!status) {
    return <div className="card"><span className="spinner" />connecting to run <span className="watermark-id">{runId.slice(0, 8)}</span>…</div>;
  }

  const interrupt = status.pending_interrupt;

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}>Awaiting your review</h2>
        <span className={`pill ${status.status}`}>{status.stage?.replace(/_/g, " ")}</span>
      </div>
      {error && <div className="error-banner">{error}</div>}

      {interrupt ? (
        <div className={`decision-card ${severityClass(interrupt)}`}>
          <div className="dc-top">
            {/* Exhibit tag — the signature element. Shows exactly what
                this finding is and where it came from, the same way a
                paper exhibit sticker would in a real case file. */}
            <span className="exhibit-tag">
              {interrupt.rule_id ? interrupt.rule_id.replace(/_/g, " ") : interrupt.item_type}
              {" · "}
              {interrupt.item_id.slice(0, 8)}
            </span>
            {interrupt.severity && <span className={`pill ${interrupt.severity}`}>{interrupt.severity}</span>}
          </div>
          <p className="dc-text">{interrupt.description}</p>
          <div className="dc-actions">
            <button className="stamp-btn reject" disabled={deciding} onClick={() => decide("rejected")}>
              Reject
            </button>
            <button className="stamp-btn approve" disabled={deciding} onClick={() => decide("approved")}>
              Approve
            </button>
          </div>
          <div className="dc-hint">
            {deciding ? "recording…" : "nothing downstream proceeds until this is decided"}
          </div>
        </div>
      ) : status.status === "done" ? (
        <div className="empty-state">Run complete.</div>
      ) : status.status === "failed" ? (
        <div className="error-banner">This run failed — check the backend log.</div>
      ) : (
        <div className="empty-state"><span className="spinner" />{status.stage?.replace(/_/g, " ")}…</div>
      )}
    </div>
  );
}

export function ReviewTab({ pileId, activeRunIds, onRunSettled, refreshKey }) {
  const [findings, setFindings] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.listFindings(pileId)
      .then((f) => !cancelled && setFindings(f))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Couldn't load findings."))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [pileId, refreshKey, activeRunIds.length]);

  const decided = findings.filter((f) => f.status !== "pending").length;
  const total = findings.length;

  return (
    <>
      {activeRunIds.map((runId) => (
        <RunTracker key={runId} runId={runId} onSettled={onRunSettled} />
      ))}

      <div className="card">
        <h2>Docket</h2>
        {total > 0 && (
          <div className="queue-progress">
            <span>Reviewed</span>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${(decided / total) * 100}%` }} /></div>
            <span className="count">{decided} / {total}</span>
          </div>
        )}
        {error && <div className="error-banner">{error}</div>}
        {loading ? (
          <div className="empty-state"><span className="spinner" />loading…</div>
        ) : findings.length === 0 ? (
          <div className="empty-state">No findings yet for this file.</div>
        ) : (
          findings.map((f) => (
            <div className="ledger-row" key={f.id}>
              <span className={`pill ${f.severity}`}>{f.severity}</span>
              <span className="ledger-desc">{f.description}</span>
              {f.claim_slot && <span className="exhibit-tag">{f.claim_slot.replace(/_/g, " ")}</span>}
              <span className={`pill ${f.status}`}>{f.status}</span>
            </div>
          ))
        )}
      </div>
    </>
  );
}
