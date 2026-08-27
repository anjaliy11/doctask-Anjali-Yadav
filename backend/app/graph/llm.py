"""
Single entry point for every LLM call in the graph: `call_structured()`.

Guardrails baked in here rather than left to each node:
- Two-provider fallback (Gemini -> Groq) so one provider's daily free-tier
  cap or an outage doesn't stall the whole pile.
- Retries with backoff on transient errors (rate limit, timeout) — NOT on
  validation errors, since retrying a schema mismatch with the same input
  just wastes another call.
- Every call is logged to run_costs (tokens, latency, provider) regardless
  of which provider actually served it.
- A hard per-run call ceiling (settings.max_llm_calls_per_run) raises
  instead of looping forever — protects the free-tier daily quota from a
  runaway extraction loop on a malformed document.
"""
import time
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.db.models import RunCost

T = TypeVar("T", bound=BaseModel)

_settings = get_settings()


class LLMCallBudgetExceeded(Exception):
    pass


class AllProvidersFailed(Exception):
    pass


class RetryableProviderError(Exception):
    """Wraps rate-limit/timeout errors from either provider so tenacity
    only retries THIS class, never a schema validation failure."""


def _call_count_key(run_id: str) -> str:
    return f"llm_call_count:{run_id}"


_call_counts: dict[str, int] = {}  # process-local; fine for this scale, see README


def _check_budget(run_id: str) -> None:
    used = _call_counts.get(run_id, 0)
    if used >= _settings.max_llm_calls_per_run:
        raise LLMCallBudgetExceeded(
            f"Run {run_id} hit the {_settings.max_llm_calls_per_run}-call "
            f"ceiling. Raise MAX_LLM_CALLS_PER_RUN if this pile genuinely "
            f"needs more, or check for a loop bug first."
        )
    _call_counts[run_id] = used + 1


def _get_gemini(schema: type[T]):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=_settings.gemini_model,
        google_api_key=_settings.gemini_api_key,
        temperature=0,
        # Let call_structured() hand off to Groq immediately on quota errors.
        # The SDK default retries 429s internally, making a free-tier quota
        # miss look like a hung run.
        max_retries=0,
    )
    return llm.with_structured_output(schema, include_raw=True)


def _get_groq(schema: type[T]):
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=_settings.groq_model,
        api_key=_settings.groq_api_key,
        temperature=0,
        max_retries=0,
    )
    return llm.with_structured_output(schema, include_raw=True)


@retry(
    retry=retry_if_exception_type(RetryableProviderError),
    # Provider failover is faster and more useful than retrying the same
    # exhausted provider. Transient retries can be added per provider later.
    stop=stop_after_attempt(1),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _invoke_one(bound_llm, system: str, user: str):
    try:
        start = time.monotonic()
        result = bound_llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return result, elapsed_ms
    except Exception as e:  # noqa: BLE001 — provider SDKs raise their own exception types
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg or "timeout" in msg or "503" in msg:
            raise RetryableProviderError(str(e)) from e
        raise


def call_structured(
    *,
    session: Session,
    run_id: str,
    stage: str,
    system: str,
    user: str,
    schema: type[T],
) -> T:
    """
    Returns a validated instance of `schema`. Raises AllProvidersFailed if
    both Gemini and Groq are unavailable/misconfigured/exhausted — callers
    must catch this and degrade gracefully (mark the item as
    needs_manual_review, never silently drop it — see nodes.py).
    """
    _settings.require_llm_keys()
    _check_budget(run_id)

    providers: list[tuple[str, callable]] = []
    if _settings.gemini_api_key:
        providers.append(("gemini", _get_gemini))
    if _settings.groq_api_key:
        providers.append(("groq", _get_groq))

    last_error: Exception | None = None
    for name, factory in providers:
        try:
            bound = factory(schema)
            raw_result, elapsed_ms = _invoke_one(bound, system, user)
            parsed = raw_result["parsed"]
            usage = getattr(raw_result.get("raw"), "usage_metadata", {}) or {}
            _log_cost(
                session=session,
                run_id=run_id,
                stage=stage,
                provider=name,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
                latency_ms=elapsed_ms,
            )
            if parsed is None:
                # Structured output parsing failed even though the call
                # succeeded — treat as this provider failing, try the next.
                last_error = ValueError(f"{name} returned unparseable structured output")
                continue
            return parsed
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise AllProvidersFailed(
        f"All configured LLM providers failed for stage={stage}, run={run_id}. "
        f"Last error: {last_error}"
    )


def _log_cost(
    *, session: Session, run_id: str, stage: str, provider: str,
    tokens_in: int, tokens_out: int, latency_ms: int,
) -> None:
    session.add(
        RunCost(
            run_id=run_id,
            stage=stage,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=0.0,  # both providers are free-tier; flip this the day that changes
        )
    )
    session.flush()
