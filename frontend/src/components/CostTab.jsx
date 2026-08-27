import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";

export function CostTab({ runId }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    api.getRunCost(runId)
      .then((r) => !cancelled && setReport(r))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Couldn't load cost."));
    return () => { cancelled = true; };
  }, [runId]);

  if (!runId) {
    return <div className="card"><div className="empty-state">Run something first — cost is reported per run.</div></div>;
  }

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>Cost — run <span className="watermark-id">{runId.slice(0, 8)}</span></h2>
        {report && <span className="pill info">${report.total_cost_usd.toFixed(4)} · {report.total_tokens} tok</span>}
      </div>
      {error && <div className="error-banner">{error}</div>}
      {!report ? (
        <div className="empty-state"><span className="spinner" />tallying…</div>
      ) : report.by_stage.length === 0 ? (
        <div className="empty-state">No billable calls made for this run yet.</div>
      ) : (
        <table className="cost-table">
          <thead>
            <tr><th>Stage</th><th>Provider</th><th>Calls</th><th>Tok in</th><th>Tok out</th><th>Latency</th><th>Cost</th></tr>
          </thead>
          <tbody>
            {report.by_stage.map((s, i) => (
              <tr key={i}>
                <td>{s.stage}</td><td>{s.provider}</td><td>{s.calls}</td>
                <td>{s.tokens_in}</td><td>{s.tokens_out}</td><td>{s.total_latency_ms}ms</td>
                <td>${s.cost_usd.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr><td colSpan={6}>Total</td><td>${report.total_cost_usd.toFixed(4)}</td></tr>
          </tfoot>
        </table>
      )}
    </div>
  );
}