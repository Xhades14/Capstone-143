"""Placeholder Google Fact Check adapter.

This module provides a minimal `query_fact_check(claim)` function that
returns a list of normalized result dicts matching the EvidenceItem fields.
It's intentionally a placeholder: when `GOOGLE_FACT_CHECK_API_KEY` is set you
can replace the body with a real HTTP client to Google's Fact Check tools.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any


def query_fact_check(normalized_claim: str) -> List[Dict[str, Any]]:
    """Query Google Fact Check API for the normalized claim.

    Returns a list of dicts: {url, title, snippet, published_date, domain, language, score}

    Currently this is a placeholder that returns an empty list unless you
    implement the API call. It checks for `GOOGLE_FACT_CHECK_API_KEY` env var
    as a signal for intent.
    """
    api_key = os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
    if not api_key:
        # No key configured: return empty list (caller will fallback to Gemini)
        return []

    # TODO: Implement actual Fact Check API call here using api_key.
    # Example: call Google's fact-check tools or a third-party fact-check index.
    # For now, return empty list to indicate no matches.
    return []
