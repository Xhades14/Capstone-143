from __future__ import annotations
from types import SimpleNamespace
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph import build_graph


class FakeLLM:
    """Simple fake LLM that returns queued string responses as .content."""
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, prompt):
        resp = self._responses.pop(0) if self._responses else "[]"
        return SimpleNamespace(content=resp)


def test_pipeline_runs_minimally():
    fake_llm = FakeLLM(responses=[
        "[]",  # claim extractor returns no claims
        "{\n \"claim_id\":\"c1\", \"verdicts\":[], \"supporting_count\":0, \"refuting_count\":0, \"claim_score\": null, \"top_evidence_ids\":[] }",
        "{\n \"post_id\":\"p1\", \"label\":\"send_downstream\" }",
    ])
    app = build_graph(llm_override=fake_llm)
    state = {
        "post_id": "p1",
        "post_text": "I read that mrna vaccines cause cancer !",
        "lang_hint": "en",
    }
    out = app.invoke(state)
    assert "final_decision" in out
