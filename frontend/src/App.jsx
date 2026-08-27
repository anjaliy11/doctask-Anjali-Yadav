import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { Sidebar } from "./components/Sidebar";
import { IntakeTab } from "./components/IntakeTab";
import { ReviewTab } from "./components/ReviewTab";
import { DeliverableTab } from "./components/DeliverableTab";
import { CostTab } from "./components/CostTab";

const TABS = ["Intake", "Review", "Deliverable", "Cost"];

export default function App() {
  const [piles, setPiles] = useState([]);
  const [pilesLoading, setPilesLoading] = useState(true);
  const [selectedPileId, setSelectedPileId] = useState(null);
  const [apiUp, setApiUp] = useState(null);
  const [activeTab, setActiveTab] = useState("Intake");

  const [activeRunIds, setActiveRunIds] = useState([]);
  const [lastSettledRunId, setLastSettledRunId] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [pendingByPile] = useState({});

  const refreshPiles = useCallback(async () => {
    setPilesLoading(true);
    try {
      const list = await api.listPiles();
      setPiles(list);
      setApiUp(true);
      setSelectedPileId((prev) => (prev && list.some((pile) => pile.id === prev) ? prev : null));
    } catch (err) {
      console.error("Could not reach the API:", err);
      setApiUp(false);
    } finally {
      setPilesLoading(false);
    }
  }, []);

  useEffect(() => { refreshPiles(); }, [refreshPiles]);

  function handlePileCreated(pile) {
    setPiles((prev) => [pile, ...prev]);
    setSelectedPileId(pile.id);
    setActiveTab("Intake");
  }

  function handleRunStarted(runId, syncResult) {
    if (runId) {
      setActiveRunIds((prev) => (prev.includes(runId) ? prev : [...prev, runId]));
    }
    setActiveTab("Review");
    if (syncResult) {
      setLastSettledRunId(runId || lastSettledRunId);
      setRefreshKey((k) => k + 1);
    }
  }

  const handleRunSettled = useCallback((runId) => {
    setActiveRunIds((prev) => prev.filter((id) => id !== runId));
    setLastSettledRunId(runId);
    setRefreshKey((k) => k + 1);
  }, []);

  const selectedPile = piles.find((p) => p.id === selectedPileId) || null;

  function WelcomeView() {
    const [creating, setCreating] = useState(false);
    const [name, setName] = useState("");
    const [error, setError] = useState(null);
    const [submitting, setSubmitting] = useState(false);

    async function handleCreate(e) {
      e.preventDefault();
      if (!name.trim()) {
        setError("Give this pile a name first.");
        return;
      }
      setError(null);
      setSubmitting(true);
      try {
        const pile = await api.createPile(name.trim(), []);
        handlePileCreated(pile);
      } catch (err) {
        setError(err?.message || "Could not create the pile. Check the API connection.");
      } finally {
        setSubmitting(false);
      }
    }

    return (
      <div className="welcome-view">
        <div className="welcome-kicker"><span className="kicker-line" /> AGENTIC DOCUMENT PILE ANALYST</div>
        <div className="welcome-grid">
          <div className="welcome-copy">
            <p className="welcome-index">01 / THE INTAKE ROOM</p>
            <h1>The analyst<br /><em>that never sleeps.</em></h1>
            <p className="welcome-subhead">Every claim, traced to its source. Every conflict, surfaced — never resolved for you.</p>
            {!creating ? (
              <button className="welcome-cta" onClick={() => setCreating(true)}>
                <span>Open your first pile</span><span className="cta-arrow">-&gt;</span>
              </button>
            ) : (
              <form className="welcome-form" onSubmit={handleCreate}>
                <label htmlFor="welcome-pile-name">Name this document pile</label>
                <div className="welcome-form-row">
                  <input id="welcome-pile-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Acme vendor contracts" autoFocus />
                  <button type="submit" disabled={submitting}>{submitting ? "Opening..." : "Create pile"}</button>
                </div>
                {error && <div className="error-banner">{error}</div>}
              </form>
            )}
            <div className="welcome-note"><span className="status-mark">{apiUp === false ? "!" : "✓"}</span>{apiUp === false ? "API connection needs attention" : "Local-first analysis, human-approved output"}</div>
          </div>

          <div className="welcome-figure" aria-label="The analyst workflow">
            <div className="figure-orbit orbit-one" />
            <div className="figure-orbit orbit-two" />
            <div className="figure-card figure-card-main">
              <div className="figure-card-top"><span className="mini-dot" /> LIVE CASE FILE <span className="figure-status">READY</span></div>
              <div className="figure-file-title">Vendor agreement<br /><span>+ 3 related documents</span></div>
              <div className="figure-rule" />
              <div className="figure-fact"><span className="fact-number">01</span><span>Payment terms match base agreement</span><span className="exhibit-pin">EX-01</span></div>
              <div className="figure-fact conflict"><span className="fact-number">02</span><span>Renewal date needs review</span><span className="conflict-pin">!</span></div>
              <div className="figure-footer">SOURCE GROUNDED <span>HUMAN GATE PENDING</span></div>
            </div>
            <div className="figure-tag tag-source">SOURCE<br /><strong>TRACEABLE</strong></div>
            <div className="figure-tag tag-human">HUMAN<br /><strong>IN THE LOOP</strong></div>
          </div>
        </div>

        <div className="welcome-process">
          <div className="process-intro"><span>HOW IT WORKS</span><p>A document pile is a living record. Doctask keeps its reasoning visible as the record changes.</p></div>
          <div className="process-step"><span className="step-no">01</span><strong>Understand</strong><small>Extract facts from mixed documents</small></div>
          <div className="process-step"><span className="step-no">02</span><strong>Reconcile</strong><small>Surface contradictions with evidence</small></div>
          <div className="process-step"><span className="step-no">03</span><strong>Approve</strong><small>Keep a human at every gate</small></div>
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <Sidebar
        piles={piles}
        selectedPileId={selectedPileId}
        onSelect={(id) => { setSelectedPileId(id); setActiveTab("Intake"); }}
        onCreated={handlePileCreated}
        apiUp={apiUp}
        pilesLoading={pilesLoading}
        pendingByPile={pendingByPile}
      />

      <div className="workspace">
        {!selectedPile ? (
          pilesLoading ? <div className="workspace-body"><div className="empty-state">Opening the evidence room...</div></div> : <WelcomeView />
        ) : (
          <>
            <div className="topbar">
              <div>
                <h1>{selectedPile.name}</h1>
                <div className="meta">
                  {selectedPile.rules.length > 0
                    ? `${selectedPile.rules.length} rule${selectedPile.rules.length === 1 ? "" : "s"} · `
                    : ""}
                  <code>{selectedPile.id}</code>
                </div>
              </div>
            </div>

            <div className="tab-strip">
              {TABS.map((t) => (
                <button
                  key={t}
                  className={`tab-btn ${activeTab === t ? "active" : ""}`}
                  onClick={() => setActiveTab(t)}
                >
                  {t}
                  {t === "Review" && activeRunIds.length > 0 && <span className="tab-count">{activeRunIds.length}</span>}
                </button>
              ))}
            </div>

            <div className="workspace-body">
              {activeTab === "Intake" && <IntakeTab pileId={selectedPile.id} onRunStarted={handleRunStarted} />}
              {activeTab === "Review" && (
                <ReviewTab
                  pileId={selectedPile.id}
                  activeRunIds={activeRunIds}
                  onRunSettled={handleRunSettled}
                  refreshKey={refreshKey}
                />
              )}
              {activeTab === "Deliverable" && <DeliverableTab pileId={selectedPile.id} refreshKey={refreshKey} />}
              {activeTab === "Cost" && <CostTab runId={lastSettledRunId} />}
            </div>
          </>
        )}
      </div>
    </div>
  );
}