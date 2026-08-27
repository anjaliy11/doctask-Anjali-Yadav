import { useState } from "react";
import { api, ApiError } from "../api/client";

export function Sidebar({ piles, selectedPileId, onSelect, onCreated, apiUp, pilesLoading, pendingByPile }) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [rules, setRules] = useState([{ rule_id: "", description: "" }]);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  function updateRule(idx, field, value) {
    setRules((prev) => prev.map((r, i) => (i === idx ? { ...r, [field]: value } : r)));
  }
  function addRule() { setRules((prev) => [...prev, { rule_id: "", description: "" }]); }
  function removeRule(idx) { setRules((prev) => prev.filter((_, i) => i !== idx)); }

  async function handleCreate(e) {
    e.preventDefault();
    setError(null);
    if (!name.trim()) { setError("Name required."); return; }
    const cleanRules = rules
      .filter((r) => r.rule_id.trim() && r.description.trim())
      .map((r) => ({ rule_id: r.rule_id.trim(), description: r.description.trim() }));
    setSubmitting(true);
    try {
      const pile = await api.createPile(name.trim(), cleanRules);
      onCreated(pile);
      setName("");
      setRules([{ rule_id: "", description: "" }]);
      setCreating(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the API.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="mark">DT</div>
        <div className="name">
          Doctask
          <span className="dim">case file agent</span>
        </div>
      </div>

      <div className="section-label">Open files</div>
      {pilesLoading ? (
        <div className="empty-state" style={{ padding: 12 }}>opening the drawer…</div>
      ) : piles.length === 0 ? (
        <div className="empty-state" style={{ padding: 12 }}>no files yet</div>
      ) : (
        piles.map((p) => (
          <div
            key={p.id}
            className={`pile-nav-item ${p.id === selectedPileId ? "active" : ""}`}
            onClick={() => onSelect(p.id)}
          >
            <span>{p.name}</span>
            {pendingByPile?.[p.id] > 0 && <span className="pending-dot" title={`${pendingByPile[p.id]} pending`} />}
          </div>
        ))
      )}

      {!creating ? (
        <button className="secondary small" style={{ marginTop: 12 }} onClick={() => setCreating(true)}>
          + Open new file
        </button>
      ) : (
        <form onSubmit={handleCreate} style={{ marginTop: 12 }}>
          {error && <div className="error-banner">{error}</div>}
          <div className="field">
            <label>File name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="acme-vendor" autoFocus />
          </div>
          <div className="field">
            <label>Checklist (optional)</label>
            {rules.map((r, i) => (
              <div className="row" key={i} style={{ marginBottom: 6 }}>
                <input placeholder="id" value={r.rule_id} onChange={(e) => updateRule(i, "rule_id", e.target.value)} style={{ flex: "0 0 60px" }} />
                <input placeholder="checks for…" value={r.description} onChange={(e) => updateRule(i, "description", e.target.value)} />
                <button type="button" className="ghost" onClick={() => removeRule(i)}>×</button>
              </div>
            ))}
            <button type="button" className="secondary small" onClick={addRule}>+ rule</button>
          </div>
          <div className="row">
            <button type="submit" className="small" disabled={submitting}>{submitting ? "Opening…" : "Open file"}</button>
            <button type="button" className="secondary small" onClick={() => setCreating(false)}>Cancel</button>
          </div>
        </form>
      )}

      <div className="sidebar-footer">
        <span className={apiUp === true ? "up" : apiUp === false ? "down" : ""}>
          {apiUp === null ? "checking wire…" : apiUp ? "● wire connected" : "● wire down"}
        </span>
      </div>
    </aside>
  );
}
