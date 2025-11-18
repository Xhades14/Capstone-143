from __future__ import annotations
from typing import List, Optional
import os
try:
    from .gemini_client import GeminiLLM
except Exception:
    GeminiLLM = None
try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover - available after install
    StateGraph = None  # type: ignore
    END = "__END__"  # type: ignore
from .schemas import Claim, ClaimEvidence, VerifierResult, FinalDecision, ManipulationScore
from .agents import (
    build_claim_detector,
    build_claim_extractor,
    build_evidence_retriever,
    build_claim_verifier,
    build_final_decider,
    compute_manipulation_score,
)

class PipelineState(dict):
    pass

# Build the orchestrated graph per the instructions

def build_graph(gemini_model: str = "gemini-2.5-flash", llm_override: Optional[object] = None):
    # LLM instances (can be swapped). Allow injection for tests.
    if llm_override is not None:
        llm = llm_override
    else:
        # Minimal echo LLM for local runs without API keys or deps.
        class _EchoLLM:
            def invoke(self, prompt):
                from types import SimpleNamespace
                return SimpleNamespace(content="[]")

        llm = None
        # Prefer Gemini if key present
        if os.environ.get("GEMINI_API_KEY") and GeminiLLM is not None:
            try:
                llm = GeminiLLM(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    model=os.environ.get("GEMINI_MODEL", gemini_model),
                )
            except Exception:
                llm = None
        if llm is None:
            llm = _EchoLLM()

    detect_claim = build_claim_detector(llm)
    extract_claims = build_claim_extractor(llm)
    retrieve_ev = build_evidence_retriever()
    verify_claim = build_claim_verifier(llm)
    final_decide = build_final_decider(llm)

    def node_detect(state: PipelineState):
        """Pre-filter: Detect if text contains verifiable claims."""
        detection = detect_claim(state["post_text"])
        state["claim_detection"] = detection
        return state

    def node_extract(state: PipelineState):
        # Skip if no claim detected
        detection = state.get("claim_detection", {})
        if not detection.get("has_claim", True):
            print(f"⏭️  Skipping claim extraction - no verifiable claims detected")
            state["claims"] = []
            return state
        
        claims: List[Claim] = extract_claims(state["post_id"], state["post_text"], state.get("lang_hint", ""))
        state["claims"] = claims
        return state

    def node_retrieve(state: PipelineState):
        evidences: List[ClaimEvidence] = retrieve_ev(state.get("claims", []))
        state["claim_evidence"] = evidences
        return state

    def node_verify(state: PipelineState):
        results: List[VerifierResult] = []
        evidences = {ev.claim_id: ev for ev in state.get("claim_evidence", [])}
        for c in state.get("claims", []):
            res = verify_claim(c, evidences.get(c.claim_id, ClaimEvidence(claim_id=c.claim_id)))
            results.append(res)
        state["verifier_results"] = results
        # coverage: fraction with >=1 evidence
        covered = sum(1 for ev in state.get("claim_evidence", []) if ev.evidence)
        total = max(1, len(state.get("claims", [])))
        state["retrieval_coverage"] = covered/total
        return state

    def node_decide(state: PipelineState):
        # If no claims detected, return early decision
        detection = state.get("claim_detection", {})
        if not detection.get("has_claim", True):
            decision = FinalDecision(
                post_id=state["post_id"],
                label="send_downstream",
                why=f"No verifiable claims detected: {detection.get('reasoning', 'N/A')}",
                post_verification_score=None,
            )
            state["final_decision"] = decision
            state["manipulation"] = ManipulationScore(manipulation_score=0.0, explanation="No claims to analyze")
            return state
        
        manipulation: ManipulationScore = compute_manipulation_score(state.get("post_text", ""))

        # Build enriched claims list for final decision per PRD
        claims_list: List[Claim] = state.get("claims", [])
        claim_conf = {c.claim_id: c.confidence for c in claims_list}
        ev_by_claim = {ev.claim_id: ev.evidence for ev in state.get("claim_evidence", [])}

        enriched_claims = []
        for v in state.get("verifier_results", []):
            evidence_urls = getattr(v, "evidence_urls", []) or []
            enriched_claims.append({
                "claim_id": v.claim_id,
                "claim_score": v.claim_score,
                "support_confidence": v.support_confidence,
                "refute_confidence": v.refute_confidence,
                "evidence_urls": evidence_urls,
                "claim_confidence": claim_conf.get(v.claim_id, 0.0),
            })

        decision: FinalDecision = final_decide(
            state["post_id"],
            state["post_text"],
            enriched_claims,
            manipulation,
            state.get("retrieval_coverage", 0.0),
        )
        state["manipulation"] = manipulation
        state["final_decision"] = decision
        return state

    # Prefer the simple runner by default to avoid requiring LangGraph state annotations.
    # You can opt into LangGraph by setting PIPELINE_SIMPLE_RUNNER=0 in the environment.
    use_simple = (
        llm_override is not None
        or StateGraph is None
        or os.environ.get("PIPELINE_SIMPLE_RUNNER", "1") != "0"
    )
    if use_simple:
        class _SimpleApp:
            def invoke(self, state: PipelineState):
                state = node_detect(state)
                state = node_extract(state)
                state = node_retrieve(state)
                state = node_verify(state)
                state = node_decide(state)
                return state
        return _SimpleApp()
    else:
        graph = StateGraph(PipelineState)
        graph.add_node("detect", node_detect)
        graph.add_node("extract", node_extract)
        graph.add_node("retrieve", node_retrieve)
        graph.add_node("verify", node_verify)
        graph.add_node("decide", node_decide)

        graph.set_entry_point("detect")
        graph.add_edge("detect", "extract")
        graph.add_edge("extract", "retrieve")
        graph.add_edge("retrieve", "verify")
        graph.add_edge("verify", "decide")
        graph.add_edge("decide", END)

        return graph.compile()
