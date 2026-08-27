"""
Wires the nodes into a StateGraph with Postgres-backed checkpointing.

Session lifecycle per node (this is the part that makes crash-safety and
interrupt() actually work together, not just each in isolation):
- A fresh SQLAlchemy session is opened at the START of every node
  invocation — including replays after an interrupt() or a crash.
- On a normal return: commit, then close.
- On interrupt() firing (a GraphInterrupt exception unwinding through the
  node): the `finally` closes the session WITHOUT an explicit commit.
  Whatever was flush()-ed but not committed is rolled back. This is
  intentional, not a gap — see the long comment in nodes.py::human_gate.
  LangGraph's own checkpointer independently remembers each interrupt's
  resume value, so replay re-derives and re-applies the same writes
  idempotently. The two systems (our Postgres tables, LangGraph's
  checkpoint store) are NOT in one distributed transaction; that's a
  known, accepted trade-off for this timeline, safe specifically because
  every write in nodes.py is upsert-safe.
- On any OTHER exception: same as above, close without commit, then
  re-raise so LangGraph/FastAPI surfaces a real error instead of a swallowed
  failure.
"""
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.db.session import SessionLocal
from app.graph.nodes import (
    classify_and_chunk,
    compose_deliverable,
    examine_rules,
    extract_facts,
    human_gate,
    reconcile,
)
from app.graph.state import PileState


def _with_session(node_fn):
    def wrapped(state: PileState) -> PileState:
        session = SessionLocal()
        try:
            result = node_fn(state, session)
            session.commit()
            return result
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    wrapped.__name__ = node_fn.__name__
    return wrapped


def build_graph():
    graph = StateGraph(PileState)
    graph.add_node("classify_and_chunk", _with_session(classify_and_chunk))
    graph.add_node("extract_facts", _with_session(extract_facts))
    graph.add_node("reconcile", _with_session(reconcile))
    graph.add_node("examine_rules", _with_session(examine_rules))
    graph.add_node("human_gate", _with_session(human_gate))
    graph.add_node("compose_deliverable", _with_session(compose_deliverable))

    graph.set_entry_point("classify_and_chunk")
    graph.add_edge("classify_and_chunk", "extract_facts")
    graph.add_edge("extract_facts", "reconcile")
    graph.add_edge("reconcile", "examine_rules")
    graph.add_edge("examine_rules", "human_gate")
    graph.add_edge("human_gate", "compose_deliverable")
    graph.add_edge("compose_deliverable", END)

    return graph


_compiled = None
_checkpointer_cm = None  # MUST be kept alive at module scope — see note below


def get_compiled_graph():
    """Lazily compiles and caches the graph with its checkpointer. The
    PostgresSaver connection is opened once per process, not per call.

    `PostgresSaver.from_conn_string()` returns a context manager whose
    generator-based cleanup closes the connection once the context manager
    object itself is garbage collected — not only on explicit __exit__(). A
    local variable holding just the context manager, discarded when this
    function returns, gets GC'd almost immediately in CPython, silently
    closing the connection out from under the checkpointer that's still
    using it (this broke test_resume_after_kill.py on the first run — the
    error was `psycopg.OperationalError: the connection is closed`). Keeping
    `_checkpointer_cm` alive at module scope for the life of the process
    fixes it.
    """
    global _compiled, _checkpointer_cm
    if _compiled is None:
        settings = get_settings()
        _checkpointer_cm = PostgresSaver.from_conn_string(settings.psycopg_conninfo)
        checkpointer = _checkpointer_cm.__enter__()
        checkpointer.setup()  # creates LangGraph's own checkpoint tables if absent
        _compiled = build_graph().compile(checkpointer=checkpointer)
    return _compiled
