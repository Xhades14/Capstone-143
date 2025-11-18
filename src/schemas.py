from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Shared data models

class Claim(BaseModel):
    claim_id: str
    tokenization: Literal["whitespace"] = "whitespace"
    tokens: List[str]
    token_mask: List[int]
    span_text: str
    normalized_claim: str
    lang: str
    english: str
    confidence: float

class EvidenceItem(BaseModel):
    id: int
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    published_date: Optional[str] = None
    domain: Optional[str] = None
    source_type: Literal["factcheck_api","gemini","kb"]
    language: Optional[str] = None
    score: Optional[float] = None

class ClaimEvidence(BaseModel):
    claim_id: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    note: Optional[str] = None

class VerifierResult(BaseModel):
    claim_id: str
    overall_verdict: Optional[dict] = None  # {label: SUPPORT/REFUTE/NEUTRAL, confidence: 0-1}
    support_confidence: float = 0.0  # Confidence that evidence supports the claim
    refute_confidence: float = 0.0   # Confidence that evidence refutes the claim
    claim_score: Optional[float] = None  # 0.0=refuted, 0.5=neutral, 1.0=supported
    reasoning: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)  # URLs of evidence used
    note: Optional[str] = None

class FinalDecision(BaseModel):
    post_id: str
    label: Literal["high_conf_true","high_conf_fake","send_downstream"]
    post_verification_score: Optional[float] = None
    why: Optional[str] = None
    top_supporting_urls: List[str] = Field(default_factory=list)
    top_refuting_urls: List[str] = Field(default_factory=list)

class ManipulationScore(BaseModel):
    manipulation_score: float
    explanation: Optional[str] = None
