"""Gemini grounded web-search adapter.

Uses the google-genai client pattern (if installed) to call Gemini with the
`google_search` tool enabled. Extracts groundingMetadata and returns a list
of normalized search result dicts: {url, title, snippet, published_date, domain, language, score}.

If the google genai client isn't available, this module will raise at import
time; caller should handle fallback to other search strategies.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any

try:
    from google import genai
    from google.genai import types
except Exception as e:  # pragma: no cover - optional dependency
    genai = None
    types = None


def gemini_grounded_search(query: str, model: str = "gemini-2.5-flash") -> List[Dict[str, Any]]:
    """Run a grounded search with Gemini's google_search tool and return normalized results.

    Returns a list of dicts with keys: url, title, snippet, published_date, domain, language, score
    """
    if genai is None or types is None:
        raise RuntimeError("google.genai client not available; install google-genai or provide an alternative search client")

    # Configure API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    # Enhanced query to prioritize credible sources
    enhanced_query = (
        f"Search for factual information about: {query}\n\n"
        "IMPORTANT: Only use highly credible and authoritative sources such as:\n"
        "- Government health agencies (CDC, NIH, FDA, WHO)\n"
        "- Major medical journals (NEJM, The Lancet, JAMA, BMJ)\n"
        "- Academic institutions (.edu domains)\n"
        "- Established fact-checking organizations (FactCheck.org, Snopes, PolitiFact)\n"
        "- Reputable news organizations (Reuters, AP, BBC)\n"
        "- Scientific organizations (Nature, Science, AAAS)\n\n"
        "Avoid: blogs, social media, opinion pieces, or unverified sources."
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=enhanced_query,
            config=config,
        )
    except Exception as e:
        return []

    candidates = getattr(response, "candidates", None) or []
    results: List[Dict[str, Any]] = []
    
    # Extract response text first
    response_text = getattr(response, "text", None) or ""
    
    for cand in candidates:
        # The cand may have .content.parts[0].text and .groundingMetadata
        text = None
        grounding = None
        try:
            # Try multiple ways to get content text
            if hasattr(cand, "content"):
                content = cand.content
                if hasattr(content, "parts") and content.parts:
                    if len(content.parts) > 0:
                        text = getattr(content.parts[0], "text", None)
            
            # Try multiple attribute names for grounding metadata
            grounding = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
            
        except Exception:
            text = response_text or "No content"
            grounding = None

        if grounding:
            # Try different attribute names for chunks and supports
            chunks = getattr(grounding, "groundingChunks", []) or getattr(grounding, "grounding_chunks", [])
            supports = getattr(grounding, "groundingSupports", []) or getattr(grounding, "grounding_supports", [])

            # Gather urls and titles from chunks
            for idx, chunk in enumerate(chunks):
                try:
                    # Get web info from chunk with robust attribute access
                    web = getattr(chunk, "web", None)
                    if not web:
                        continue
                    
                    url = getattr(web, "uri", None) or getattr(web, "url", None)
                    title = getattr(web, "title", None)
                    
                    # Find snippet from grounding supports - match chunk index
                    # Note: Multiple sources may reference the same synthesized text segment
                    # This is correct behavior for grounded generation
                    snippet = None
                    for support in supports:
                        chunk_indices = getattr(support, "groundingChunkIndices", []) or []
                        if idx in chunk_indices:
                            segment = getattr(support, "segment", None)
                            if segment:
                                snippet = getattr(segment, "text", None)
                                if snippet:
                                    break
                    
                    # Fallback if no specific snippet found
                    if not snippet:
                        snippet = text[:300] if text else "No snippet available"
                    
                    # Extract domain - use title if URL is proxied
                    domain = "unknown"
                    if url:
                        try:
                            from urllib.parse import urlparse
                            parsed_domain = urlparse(url).netloc
                            # If it's a proxy domain, extract from title
                            if "vertexaisearch" in parsed_domain and title:
                                domain = title.strip()
                            else:
                                domain = parsed_domain or "unknown"
                        except:
                            domain = title.strip() if title else "unknown"
                    elif title:
                        domain = title.strip()

                    results.append({
                        "url": url or "",
                        "title": title or "No title",
                        "snippet": snippet,
                        "domain": domain,
                        "score": 1.0,
                    })
                except Exception:
                    continue
        else:
            # No groundingMetadata: fallback to returning the candidate text
            results.append({
                "url": "",
                "title": "No grounding data",
                "snippet": text or "No content available",
                "domain": "unknown",
                "score": 0.2,
            })

    return results
