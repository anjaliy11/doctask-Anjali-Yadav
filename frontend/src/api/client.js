// Use Vite's same-origin proxy in local Docker development. This keeps browser
// requests off the CORS path while the API remains directly reachable for MCP
// and other machine clients.
const BASE_URL = import.meta.env.VITE_API_BASE || "/api";

export class ApiError extends Error {
  constructor(status, body) {
    const detail =
      body?.detail || body?.cause || (body?.fix ? `${body?.error}: ${body.fix}` : null);
    super(detail || `Request failed with status ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON response — body stays null
  }
  if (!res.ok) throw new ApiError(res.status, body);
  return body;
}

export const api = {
  listPiles: () => request("/piles"),
  createPile: (name, rules = []) =>
    request("/piles", { method: "POST", body: JSON.stringify({ name, rules }) }),
  getPile: (pileId) => request(`/piles/${pileId}`),

  ingestDocument: (pileId, filename, textContent) =>
    request(`/piles/${pileId}/documents`, {
      method: "POST",
      body: JSON.stringify({ filename, text_content: textContent }),
    }),

  getRun: (runId) => request(`/runs/${runId}`),
  resumeRun: (runId) => request(`/runs/${runId}/resume`, { method: "POST" }),
  getRunCost: (runId) => request(`/runs/${runId}/cost`),

  listFindings: (pileId) => request(`/findings/pile/${pileId}`),
  approveFinding: (findingId) => request(`/findings/${findingId}/approve`, { method: "POST" }),
  rejectFinding: (findingId) => request(`/findings/${findingId}/reject`, { method: "POST" }),

  getDeliverable: (pileId) => request(`/piles/${pileId}/deliverable`),

  exportDeliverableUrl: (pileId) => `${BASE_URL}/piles/${pileId}/deliverable/export`,
};