"""
Every LLM call that produces a claim uses one of these as its
`with_structured_output()` target. The point: the model cannot hand back
free prose for anything that will end up in the deliverable — it must fill
a schema where source_chunk_id and char offsets are REQUIRED fields. A
claim with no evidence pointer is a schema validation failure, not a
silently-accepted fact.
"""
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    claim: str = Field(description="The factual claim, in plain language, one sentence.")
    claim_slot: str = Field(
        description=(
            "A short machine-stable category for what KIND of claim this is, "
            "e.g. 'payment_terms', 'renewal_date', 'termination_clause'. Facts "
            "from different documents in the same slot are compared for "
            "contradictions — use the SAME slot name for the same concept "
            "even across different documents."
        )
    )
    evidence_quote: str = Field(
        description=(
            "The exact substring from the source text that supports this "
            "claim, copied verbatim, under 40 words. Used to verify grounding "
            "— if this quote cannot be found in the source, the claim is "
            "rejected as ungrounded."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    could_not_extract: str | None = Field(
        default=None,
        description=(
            "If this chunk contains no extractable factual claims relevant to "
            "the pile, or the text is too garbled to interpret confidently, "
            "say so here instead of inventing facts. Leave null otherwise."
        ),
    )


class ChunkExtractionResult(BaseModel):
    chunk_id: str = Field(description="The exact chunk id from the input.")
    facts: list[ExtractedFact] = Field(default_factory=list)
    could_not_extract: str | None = None


class BatchExtractionResult(BaseModel):
    chunks: list[ChunkExtractionResult] = Field(default_factory=list)


class Contradiction(BaseModel):
    claim_slot: str
    fact_id_a: str
    fact_id_b: str
    explanation: str = Field(description="Plain-language description of the conflict.")


class ContradictionResult(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


class RuleFinding(BaseModel):
    rule_id: str
    description: str
    source_chunk_id: str | None = Field(
        default=None, description="null only if the finding is about an absence, not a passage"
    )
    severity: str = Field(description="one of: info, warning, critical")


class ExamineResult(BaseModel):
    findings: list[RuleFinding] = Field(default_factory=list)
    clean: bool = Field(
        description="true only if findings is empty AND every rule was actually checked"
    )
