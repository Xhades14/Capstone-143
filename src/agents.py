from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from .schemas import Claim, EvidenceItem, ClaimEvidence, VerifierResult, FinalDecision, ManipulationScore
try:
    from .gemini_search import gemini_grounded_search
except Exception:
    gemini_grounded_search = None
try:
    from .google_factcheck import query_fact_check
except Exception:
    query_fact_check = None

# JSON parsing helper (extract strict JSON from model outputs that may contain extra text)
def _try_parse_json(raw: str):
    import json
    import re
    
    # First try to parse as-is
    try:
        return json.loads(raw)
    except Exception:
        pass
    
    # Remove markdown code blocks if present
    clean_raw = raw
    if "```json" in raw or "```" in raw:
        # Extract content between ```json and ``` or between ``` and ```
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if json_match:
            clean_raw = json_match.group(1).strip()
            try:
                return json.loads(clean_raw)
            except Exception:
                pass
    
    # Try object extraction first (priority over arrays)
    try:
        l = clean_raw.find("{")
        r = clean_raw.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(clean_raw[l:r+1])
    except Exception:
        pass
    
    # try array extraction as fallback
    try:
        l = clean_raw.find("[")
        r = clean_raw.rfind("]")
        if l != -1 and r != -1 and r > l:
            return json.loads(clean_raw[l:r+1])
    except Exception:
        pass
    
    # For truncated responses, try to extract meaningful patterns
    # Look for label and confidence patterns specifically
    try:
        result = {}
        
        # Extract label (REFUTE, SUPPORT, INSUFFICIENT)
        label_match = re.search(r'"label":\s*"([^"]*)', raw)
        if label_match:
            label = label_match.group(1)
            # Handle truncated labels
            if label.startswith('R'):
                result['label'] = 'REFUTE'
            elif label.startswith('S'):
                result['label'] = 'SUPPORT'
            elif label.startswith('I'):
                result['label'] = 'INSUFFICIENT'
            else:
                result['label'] = label
        
        # Extract confidence
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', raw)
        if confidence_match:
            result['confidence'] = float(confidence_match.group(1))
        
        # Extract reasoning if present
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)', raw)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1)
        
        # Return partial result if we found something useful
        if result:
            return result
    except Exception:
        pass
    
    return None

# Agent 0: Claim Detection Filter (Pre-processing)

CLAIM_DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a claim detection filter. Output ONLY valid JSON. No explanations, no markdown, no code blocks. Just pure JSON."
    ),
    (
        "human",
        (
            "STRICT INSTRUCTIONS:\n"
            "1. Output ONLY valid JSON in this EXACT format: {{\"has_claim\": true/false, \"reasoning\": \"text\"}}\n"
            "2. DO NOT add markdown code blocks (```json)\n"
            "3. DO NOT add any text before or after the JSON\n"
            "4. Use ONLY lowercase true/false (not True/False)\n"
            "5. Always include both fields: has_claim and reasoning\n\n"
            "Task: Determine if the text contains ANY claims (factual, non-factual, or propaganda) that should be fact-checked.\n\n"
            "Return has_claim=false ONLY for:\n"
            "- Pure personal opinions without factual claims (e.g., 'I hate Mondays', 'This movie is boring')\n"
            "- Personal stories/daily activities (e.g., 'I went shopping today', 'My dog is cute')\n"
            "- Jokes, memes, obvious humor, sarcasm\n"
            "- Simple questions (e.g., 'What time is it?', 'Where is the bathroom?')\n"
            "- Greetings and small talk (e.g., 'Hello!', 'How are you?', 'Have a nice day')\n"
            "- Pure emotional expressions WITHOUT factual claims (e.g., 'I'm so happy!', 'This is frustrating!')\n\n"
            "Return has_claim=true for:\n"
            "- ANY factual assertions about the world (e.g., 'Vaccines cause autism', 'The Earth is flat')\n"
            "- Statistics or data presented as facts (e.g., '90% of people believe X')\n"
            "- Medical/health claims (e.g., 'Vitamin C cures cancer', 'Masks don't work')\n"
            "- Political claims or propaganda (e.g., 'The election was stolen', 'Government is lying')\n"
            "- Historical statements (e.g., 'Napoleon was 7 feet tall')\n"
            "- Scientific claims (e.g., 'Climate change is a hoax')\n"
            "- Conspiracy theories (e.g., '5G causes COVID')\n"
            "- Misinformation or disinformation of any kind\n"
            "- Emotional/manipulative language WITH factual claims (e.g., 'WAKE UP!!! Big Pharma is POISONING you!!!')\n"
            "- Any statement that makes an assertion about reality that can be checked\n\n"
            "IMPORTANT: Ignore emotional language, caps, exclamations. Focus on whether there's a factual claim underneath.\n"
            "Example: 'WAKE UP!!! Big Pharma is POISONING you!!!' → has_claim=true (contains claim 'Big Pharma is poisoning people')\n"
            "Example: 'I'm so ANGRY right now!!!' → has_claim=false (pure emotion, no factual claim)\n\n"
            "Text to analyze: \"{text}\"\n\n"
            "Output ONLY this JSON format (no markdown): {{\"has_claim\": true/false, \"reasoning\": \"brief explanation\"}}"
        ),
    ),
])

def build_claim_detector(llm: BaseLanguageModel):
    """Pre-filter to detect if text contains verifiable claims worth processing."""
    def run(text: str) -> Dict[str, Any]:
        import json
        print(f"🔍 Detecting if text contains verifiable claims...")
        prompt = CLAIM_DETECTION_PROMPT.format(text=text)
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _try_parse_json(raw)
        
        if not isinstance(data, dict):
            # If parsing fails, assume it has claims to be safe
            print(f"   ⚠️  Detection failed, processing as potential claim")
            return {"has_claim": True, "reasoning": "parsing_error", "claim_type": "unknown"}
        
        has_claim = data.get("has_claim", True)
        
        if has_claim:
            print(f"   ✅ Yes")
        else:
            print(f"   ❌ No")
        
        return data
    return run

# Agent 1: Claim Span Identification

CLAIM_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract factual claims from text, including propaganda and conspiracy theories. ALWAYS extract claims even if emotional or manipulative. Output ONLY valid JSON array. DO NOT add markdown or explanations."
    ),
    (
        "human",
        (
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1. Output ONLY a valid JSON array: [{{...}}]\n"
            "2. DO NOT wrap in markdown code blocks (```json)\n"
            "3. DO NOT add any text before or after the JSON\n"
            "4. Use lowercase true/false (not True/False)\n"
            "5. All numbers must be decimal format (0.96 not 96%)\n"
            "6. Ensure token_mask length EXACTLY equals tokens length\n\n"
            "Task: Extract factual claims from the post. ALWAYS extract claims from propaganda, conspiracy theories, and emotional text.\n\n"
            "RULE: If text contains a factual assertion (even if false, propaganda, or conspiracy), EXTRACT IT with HIGH confidence.\n"
            "Examples that MUST be extracted:\n"
            "- 'Big Pharma is poisoning people' → EXTRACT (confidence 0.92)\n"
            "- 'Government is hiding the truth' → EXTRACT (confidence 0.90)\n"
            "- 'Vaccines cause autism' → EXTRACT (confidence 0.95)\n"
            "- 'Climate change is a hoax' → EXTRACT (confidence 0.93)\n\n"
            "Only skip if:\n"
            "- Pure emotion: 'I'm so angry!' → SKIP (confidence 0.10, all zeros)\n"
            "- Personal intention: 'I will never trust them' → SKIP (confidence 0.12, all zeros)\n\n"
            "Output a JSON array of claim objects with these EXACT fields:\n"
            "- claim_id: string (format: \"c1\", \"c2\", etc.)\n"
            "- tokenization: string (always \"whitespace\")\n"
            "- tokens: array of strings\n"
            "- token_mask: array of 0 or 1 (MUST be same length as tokens)\n"
            "- span_text: string (exact substring from tokens, or empty string \"\")\n"
            "- normalized_claim: string (cleaned claim in ORIGINAL LANGUAGE, or empty string \"\")\n"
            "- lang: string (e.g., \"en\", \"hi\", \"es\", etc.)\n"
            "- english: string (English translation of normalized_claim, or same as normalized_claim if already English)\n"
            "- confidence: number between 0.0 and 1.0\n\n"
            "IMPORTANT: Keep normalized_claim in the ORIGINAL language of the post. Only put English translation in 'english' field.\n\n"
            "VALIDATION RULES:\n"
            "- If span_text is empty, token_mask must be all zeros [0,0,0,...]\n"
            "- If span_text is not empty, at least one token_mask value must be 1\n"
            "- confidence should be 0.90-0.99 for clear claims, 0.10-0.30 for no claims\n"
            "- normalized_claim should be empty \"\" if no verifiable claim exists\n\n"
            "Few-shot examples (use exactly this format):\n\n"
            "Example 1 (English with claim):\n"
            "Post: \"I read that mrna vaccines cause cancer !\"\n"
            "Tokens: [\"I\",\"read\",\"that\",\"mrna\",\"vaccines\",\"cause\",\"cancer\",\"!\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"I\",\"read\",\"that\",\"mrna\",\"vaccines\",\"cause\",\"cancer\",\"!\"],\n"
            "  \"token_mask\":[0,0,0,1,1,1,1,0],\n"
            "  \"span_text\":\"mrna vaccines cause cancer\",\n"
            "  \"normalized_claim\":\"mRNA vaccines cause cancer\",\n"
            "  \"lang\":\"en\",\n"
            "  \"english\":\"mRNA vaccines cause cancer\",\n"
            "  \"confidence\":0.96\n"
            "}}]\n\n"
            "Example 2 (English - opinion only, no claim):\n"
            "Post: \"I will never take a covid vaccine . . .\"\n"
            "Tokens: [\"I\",\"will\",\"never\",\"take\",\"a\",\"covid\",\"vaccine\",\".\",\".\",\".\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"I\",\"will\",\"never\",\"take\",\"a\",\"covid\",\"vaccine\",\".\",\".\",\".\"],\n"
            "  \"token_mask\":[0,0,0,0,0,0,0,0,0,0],\n"
            "  \"span_text\":\"\",\n"
            "  \"normalized_claim\":\"\",\n"
            "  \"lang\":\"en\",\n"
            "  \"english\":\"\",\n"
            "  \"confidence\":0.15\n"
            "}}]\n\n"
            "Example 3 (Hindi - factual claim)\n\n"
            "Post: \"मुझे बताया गया कि mRNA वैक्सीन से कैंसर होता है।\"\n"
            "Tokens: [\"मुझे\",\"बताया\",\"गया\",\"कि\",\"mRNA\",\"वैक्सीन\",\"से\",\"कैंसर\",\"होता\",\"है।\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"मुझे\",\"बताया\",\"गया\",\"कि\",\"mRNA\",\"वैक्सीन\",\"से\",\"कैंसर\",\"होता\",\"है।\"],\n"
            "  \"token_mask\":[0,0,0,0,1,1,1,1,1,0],\n"
            "  \"span_text\":\"mRNA वैक्सीन से कैंसर होता है\",\n"
            "  \"normalized_claim\":\"mRNA वैक्सीन से कैंसर होता है\",\n"
            "  \"lang\":\"hi\",\n"
            "  \"english\":\"mRNA vaccines cause cancer\",\n"
            "  \"confidence\":0.94\n"
            "}}]\n\n"
            "Example 4 (Hindi - opinion)\n\n"
            "Post: \"मैं कभी भी कोविड वैक्सीन नहीं लूंगा।\"\n"
            "Tokens: [\"मैं\",\"कभी\",\"भी\",\"कोविड\",\"वैक्सीन\",\"नहीं\",\"लूंगा।\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"मैं\",\"कभी\",\"भी\",\"कोविड\",\"वैक्सीन\",\"नहीं\",\"लूंगा।\"],\n"
            "  \"token_mask\":[0,0,0,0,0,0,0],\n"
            "  \"span_text\":\"\",\n"
            "  \"normalized_claim\":\"\",\n"
            "  \"lang\":\"hi\",\n"
            "  \"english\":\"\",\n"
            "  \"confidence\":0.10\n"
            "}}]\n\n"
            "Example 5 (Propaganda with emotional language - EXTRACT THE CLAIM)\n\n"
            "Post: \"WAKE UP!!! Big Pharma is POISONING you!!!\"\n"
            "Tokens: [\"WAKE\",\"UP!!!\",\"Big\",\"Pharma\",\"is\",\"POISONING\",\"you!!!\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"WAKE\",\"UP!!!\",\"Big\",\"Pharma\",\"is\",\"POISONING\",\"you!!!\"],\n"
            "  \"token_mask\":[0,0,1,1,1,1,1],\n"
            "  \"span_text\":\"Big Pharma is POISONING you\",\n"
            "  \"normalized_claim\":\"Big Pharma is poisoning people\",\n"
            "  \"lang\":\"en\",\n"
            "  \"english\":\"Big Pharma is poisoning people\",\n"
            "  \"confidence\":0.92\n"
            "}}]\n\n"
            "Example 6 (Conspiracy/propaganda - government hiding truth)\n\n"
            "Post: \"The government is HIDING the truth about vaccines!!!\"\n"
            "Tokens: [\"The\",\"government\",\"is\",\"HIDING\",\"the\",\"truth\",\"about\",\"vaccines!!!\"]\n"
            "Output:\n"
            "[{{\n"
            "  \"claim_id\":\"c1\",\n"
            "  \"tokenization\":\"whitespace\",\n"
            "  \"tokens\":[\"The\",\"government\",\"is\",\"HIDING\",\"the\",\"truth\",\"about\",\"vaccines!!!\"],\n"
            "  \"token_mask\":[1,1,1,1,1,1,1,1],\n"
            "  \"span_text\":\"The government is HIDING the truth about vaccines\",\n"
            "  \"normalized_claim\":\"The government is hiding the truth about vaccines\",\n"
            "  \"lang\":\"en\",\n"
            "  \"english\":\"The government is hiding the truth about vaccines\",\n"
            "  \"confidence\":0.90\n"
            "}}]\n\n"
            "REMEMBER: Propaganda text with emotional language SHOULD be extracted with HIGH confidence (0.90+).\n"
            "The confidence reflects whether a claim is present (high), NOT whether the claim is true (verify later).\n\n"
            "Now process the following input. Return JSON array only.\n\n"
            "Post: {post_text}\n"
            "Lang hint: {lang_hint}\n"
            "Tokens (whitespace): {tokens}\n"
        ),
    ),
])

def whitespace_tokenize(text: str) -> List[str]:
    return text.split()

def build_claim_extractor(llm: BaseLanguageModel):
    def run(post_id: str, post_text: str, lang_hint: str = "") -> List[Claim]:
        print(f"🔍 Extracting claims from: '{post_text}'")
        tokens = whitespace_tokenize(post_text)
        print(f"   Tokens: {tokens}")
        prompt = CLAIM_PROMPT.format(post_text=post_text, lang_hint=lang_hint, tokens=str(tokens))
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _try_parse_json(raw) or []
        print(f"   Found {len(data)} potential claims")
        claims: List[Claim] = []
        for i, c in enumerate(data):
            # Basic consistency checks per spec
            tokens = c.get("tokens", [])
            token_mask = c.get("token_mask", [])
            
            if len(tokens) != len(token_mask):
                print(f"   ⚠️  Skipping: token/mask length mismatch")
                continue
            
            # If token_mask is all zeros, no claim was selected
            if not any(token_mask):
                print(f"   ⚠️  Skipping: no tokens selected (all-zero mask)")
                continue
            
            # If span_text is empty but tokens are selected, skip
            if c.get("span_text", "").strip() == "" and any(token_mask):
                print(f"   ⚠️  Skipping: empty span_text with selected tokens")
                continue
            
            # Ensure required fields
            c.setdefault("claim_id", f"c{i+1}")
            c.setdefault("tokenization", "whitespace")
            c.setdefault("lang", lang_hint or "")
            c.setdefault("confidence", 0.0)
            
            # Get normalized_claim and strip whitespace
            normalized = (c.get("normalized_claim") or "").strip()
            c["normalized_claim"] = normalized  # Update with stripped version
            c.setdefault("english", normalized)  # Default to normalized if not provided
            
            # Show what we extracted
            confidence = c.get('confidence', 0)
            
            # Filter: must have normalized claim AND confidence >= 0.3
            if normalized and confidence >= 0.3:
                print(f"   ✅ Claim {c.get('claim_id')}: '{normalized}' (confidence: {confidence:.2f})")
                print(f"      Token mask: {c.get('token_mask', [])}")
                print(f"      Span text: '{c.get('span_text', '')}'")
                # Update with stripped normalized_claim
                c['normalized_claim'] = normalized
                claims.append(Claim(**c))
            else:
                reason = "empty claim" if not normalized else f"low confidence ({confidence:.2f})"
                print(f"   ❌ Filtered out: {reason}")
        return claims
    return run

# Agent 2: Evidence Retrieval (hybrid - placeholder implementations)

def build_evidence_retriever():
    def run(claims: List[Claim]) -> List[ClaimEvidence]:
        results: List[ClaimEvidence] = []
        for claim in claims:
            # Claims are already filtered by confidence >= 0.3 at extraction
            if not claim.normalized_claim:
                results.append(ClaimEvidence(claim_id=claim.claim_id, evidence=[], note="no_claim_text"))
                continue
            # First: query Google Fact Check API (if adapter configured)
            ev_items: List[EvidenceItem] = []
            note = "no_results"
            if query_fact_check is not None:
                try:
                    fc_hits = query_fact_check(claim.normalized_claim)
                    for i, h in enumerate(fc_hits[:5], start=1):
                        ev_items.append(EvidenceItem(
                            id=i,
                            url=h.get("url") or "",
                            title=h.get("title"),
                            snippet=h.get("snippet"),
                            published_date=h.get("published_date"),
                            domain=h.get("domain"),
                            source_type="factcheck_api",
                            language=h.get("language"),
                            score=h.get("score", 1.0),
                        ))
                    if ev_items:
                        note = "factcheck_api_found"
                except Exception:
                    note = "factcheck_error"

            # Always try FactCheck API first
            print(f"🔍 Searching FactCheck API for: '{claim.normalized_claim}'")
            # TODO: Implement real Google Fact Check API here
            # For now, using placeholder that returns empty
            print("   FactCheck API: No results (placeholder implementation)")
            
            # If Fact Check returned nothing, fallback to Gemini grounded search
            if not ev_items and gemini_grounded_search is not None:
                try:
                    print(f"🔍 Searching with Gemini for: '{claim.normalized_claim}'")
                    hits = gemini_grounded_search(claim.normalized_claim)
                    print(f"   Found {len(hits)} results")
                    # Normalize and keep top 5
                    for i, h in enumerate(hits[:5], start=1):
                        ev = EvidenceItem(
                            id=i,
                            url=h.get("url", ""),
                            title=h.get("title", "No title"),
                            snippet=h.get("snippet", "No snippet"),
                            published_date=None,  # Simplified - not needed for verification
                            domain=h.get("domain", "unknown"),
                            source_type="gemini",
                            language=None,  # Simplified - not needed for verification
                            score=h.get("score", 0.0),
                        )
                        ev_items.append(ev)
                    if ev_items:
                        note = "gemini_fallback"
                        print(f"   ✅ Retrieved {len(ev_items)} evidence items")
                except Exception as e:
                    print(f"   ❌ Gemini search error: {e}")
                    note = "gemini_error"

            results.append(ClaimEvidence(claim_id=claim.claim_id, evidence=ev_items, note=note))
        return results
    return run

# Agent 3: Claim Verification (LLM)

VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fact verification assistant. Output ONLY valid JSON. DO NOT add markdown code blocks. DO NOT add explanatory text. Be deterministic and consistent."
    ),
    (
        "human",
        (
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1. Output ONLY valid JSON object: {{...}}\n"
            "2. DO NOT wrap in markdown code blocks (```json)\n"
            "3. DO NOT add any text before or after the JSON\n"
            "4. Use exact field names specified below\n"
            "5. All confidence values must be between 0.0 and 1.0\n\n"
            "Input: Claim (text), Evidence (synthesized information from multiple sources with id, url, snippet, domain).\n\n"
            "Task: Analyze the evidence to determine:\n"
            "1. Does the evidence clearly SUPPORT the claim? (support_confidence: 0.0-1.0)\n"
            "2. Does the evidence clearly REFUTE the claim? (refute_confidence: 0.0-1.0)\n"
            "3. Calculate claim_score based on evidence strength\n\n"
            "DETERMINISTIC SCORING RULES:\n"
            "- If evidence explicitly states claim is FALSE/INCORRECT → refute_confidence=0.95-1.0, support_confidence=0.0, claim_score=0.0-0.1\n"
            "- If evidence explicitly states claim is TRUE/CORRECT → support_confidence=0.95-1.0, refute_confidence=0.0, claim_score=0.9-1.0\n"
            "- If evidence is unclear/mixed → both confidences 0.3-0.5, claim_score=0.4-0.6\n"
            "- If evidence is completely unrelated → both confidences 0.0, claim_score=0.5\n"
            "- claim_score formula: 0.0=strongly refuted, 0.5=neutral/insufficient, 1.0=strongly supported\n\n"
            "REQUIRED OUTPUT FORMAT (must include ALL fields):\n"
            "{{\n"
            "  \"claim_id\": null,\n"
            "  \"overall_verdict\": {{\"label\": \"SUPPORT\" or \"REFUTE\" or \"NEUTRAL\", \"confidence\": 0.0-1.0}},\n"
            "  \"support_confidence\": 0.0-1.0,\n"
            "  \"refute_confidence\": 0.0-1.0,\n"
            "  \"claim_score\": 0.0-1.0,\n"
            "  \"reasoning\": \"brief explanation\"\n"
            "}}\n\n"
            "BE CONSISTENT: Same evidence should always produce same scores.\n\n"
            "Claim: {claim_text}\n"
            "Evidence: {evidence}\n\n"
            "Output ONLY the JSON object (no markdown):"
        ),
    ),
])

def build_claim_verifier(llm: BaseLanguageModel):
    def run(claim: Claim, evidence: ClaimEvidence) -> VerifierResult:
        import json
        if not evidence.evidence:
            return VerifierResult(
                claim_id=claim.claim_id,
                claim_score=None,
                support_confidence=0.0,
                refute_confidence=0.0,
                note="no_evidence"
            )
        
        print(f"🔬 Verifying claim: '{claim.normalized_claim}'")
        print(f"   Evidence items: {len(evidence.evidence)}")
        
        ev_min = [
            {"id": e.id, "snippet": e.snippet[:200], "domain": e.domain}  # Limit snippet length
            for e in evidence.evidence
        ]
        
        # Show what evidence we're sending to the verifier
        print(f"   Sources consulted:")
        for i, e in enumerate(evidence.evidence):
            print(f"     {i+1}. {e.domain} ({e.source_type})")
        
        # Show the synthesized finding (use first snippet as they're all similar in grounded search)
        if evidence.evidence and evidence.evidence[0].snippet:
            print(f"\n   📋 Evidence Summary:")
            print(f"   {evidence.evidence[0].snippet[:300]}...")
        print()
        prompt = VERIFIER_PROMPT.format(claim_text=claim.normalized_claim or claim.span_text, evidence=json.dumps(ev_min, ensure_ascii=False))
        
        print("   🤖 Calling verifier...")
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        
        # Check for errors in response
        if "__gemini_error__" in raw:
            print(f"   ❌ Verifier error: {raw}")
            return VerifierResult(
                claim_id=claim.claim_id,
                claim_score=None,
                support_confidence=0.0,
                refute_confidence=0.0,
                note="llm_error"
            )
        
        data = _try_parse_json(raw)
        if not isinstance(data, dict):
            return VerifierResult(
                claim_id=claim.claim_id,
                claim_score=None,
                support_confidence=0.0,
                refute_confidence=0.0,
                note="verifier_error"
            )

        # Ensure claim_id is set correctly (override whatever LLM returned)
        data["claim_id"] = claim.claim_id
        
        # Set defaults for new confidence-based fields
        data.setdefault("support_confidence", 0.0)
        data.setdefault("refute_confidence", 0.0)
        data.setdefault("claim_score", 0.5)
        data.setdefault("reasoning", "")
        
        # Extract evidence URLs
        evidence_urls = [e.url for e in evidence.evidence if e.url]
        data["evidence_urls"] = evidence_urls[:5]  # Keep top 5
        
        # Calculate claim_score if not provided based on confidences
        if data.get("claim_score") is None:
            support_conf = data.get("support_confidence", 0.0)
            refute_conf = data.get("refute_confidence", 0.0)
            
            if support_conf == 0.0 and refute_conf == 0.0:
                data["claim_score"] = 0.5  # Neutral
            else:
                # Score based on relative confidences: refute=0.0, support=1.0
                total_conf = support_conf + refute_conf
                if total_conf > 0:
                    data["claim_score"] = round(support_conf / total_conf, 2)
                else:
                    data["claim_score"] = 0.5
        
        print(f"   ✅ Verification complete:")
        print(f"      Support confidence: {data.get('support_confidence'):.2f}")
        print(f"      Refute confidence: {data.get('refute_confidence'):.2f}")
        print(f"      Claim score: {data.get('claim_score'):.2f}")
        
        return VerifierResult(**data)
    return run

# Agent 4: Final Decision (LLM + logic)

FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You produce final JSON decisions per deterministic rules. Output ONLY valid JSON. DO NOT add markdown code blocks. DO NOT add explanatory text. Apply rules consistently."
    ),
    (
        "human",
        (
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1. Output ONLY valid JSON object: {{...}}\n"
            "2. DO NOT wrap in markdown code blocks (```json)\n"
            "3. DO NOT add any text before or after the JSON\n"
            "4. Use exact field names and values specified below\n\n"
            "Input: claims (with claim_score, support_confidence, refute_confidence), manipulation_score, retrieval_coverage.\n\n"
            "DETERMINISTIC DECISION RULES (apply in this order):\n"
            "Rule 1: If ANY claim_score is null OR retrieval_coverage < 0.5 → label=\"send_downstream\"\n"
            "Rule 2: If ANY claim has (claim_score <= 0.10 AND refute_confidence >= 0.8) → label=\"high_conf_fake\"\n"
            "Rule 3: If ALL claims have (claim_score >= 0.90 AND support_confidence >= 0.8 AND manipulation_score < 0.6) → label=\"high_conf_true\"\n"
            "Rule 4: If neutral claims (0.3 <= claim_score <= 0.7) AND manipulation_score >= 0.3 → label=\"send_downstream\"\n"
            "Rule 5: If manipulation_score >= 0.6 (regardless of claim scores) → label=\"send_downstream\"\n"
            "Rule 6: Otherwise → label=\"send_downstream\"\n\n"
            "REQUIRED OUTPUT FORMAT (must include ALL fields):\n"
            "{{\n"
            "  \"post_id\": null,\n"
            "  \"label\": \"high_conf_true\" or \"high_conf_fake\" or \"send_downstream\",\n"
            "  \"post_verification_score\": 0.0-1.0 or null,\n"
            "  \"why\": \"explanation of which rule was applied\",\n"
            "  \"top_supporting_urls\": [\"url1\", \"url2\"],\n"
            "  \"top_refuting_urls\": [\"url1\", \"url2\"]\n"
            "}}\n\n"
            "IMPORTANT: Only use URLs provided in the input. DO NOT invent URLs.\n"
            "BE CONSISTENT: Same input should always produce same output.\n\n"
            "Input: {{\n'claims': {claims},\n'manipulation_score': {manipulation_score},\n'retrieval_coverage': {retrieval_coverage}\n}}\n\n"
            "Output ONLY the JSON object (no markdown):"
        ),
    ),
])

def build_final_decider(llm: BaseLanguageModel):
    def run(post_id: str, post_text: str, claims_results: List[Dict[str, Any]], manipulation: ManipulationScore, retrieval_coverage: float) -> FinalDecision:
        import json
        # If any null claim_score or low coverage -> send_downstream
        if retrieval_coverage < 0.5 or any(v.get("claim_score") is None for v in claims_results):
            return FinalDecision(post_id=post_id, label="send_downstream", why="Low coverage or missing scores.")
        # Prepare simplified claim view for prompt
        simplified = []
        for v in claims_results:
            simplified.append({
                "claim_id": v.get("claim_id"),
                "claim_score": v.get("claim_score"),
                "support_confidence": v.get("support_confidence", 0.0),
                "refute_confidence": v.get("refute_confidence", 0.0),
            })
        prompt = FINAL_PROMPT.format(
            claims=json.dumps(simplified),
            manipulation_score=manipulation.manipulation_score,
            retrieval_coverage=retrieval_coverage,
        )
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _try_parse_json(raw)
        if not isinstance(data, dict):
            return FinalDecision(post_id=post_id, label="send_downstream", why="final_decider_error")
        # Always override post_id to ensure it's set correctly (LLM might return null)
        data["post_id"] = post_id
        data.setdefault("label", "send_downstream")
        # Fill top URLs from claims_results if LLM omitted
        sup = data.get("top_supporting_urls")
        ref = data.get("top_refuting_urls")
        if not sup or not isinstance(sup, list):
            sup_urls = []
            for v in claims_results:
                # Use support_confidence to determine if claim is supported
                if (v.get("support_confidence") or 0) >= 0.5:
                    for url in v.get("evidence_urls", [])[:2]:
                        if url:
                            sup_urls.append(url) 
            data["top_supporting_urls"] = sup_urls[:5]
        if not ref or not isinstance(ref, list):
            ref_urls = []
            for v in claims_results:
                # Use refute_confidence to determine if claim is refuted
                if (v.get("refute_confidence") or 0) >= 0.5:
                    for url in v.get("evidence_urls", [])[:2]:
                        if url:
                            ref_urls.append(url) 
            data["top_refuting_urls"] = ref_urls[:5]
        return FinalDecision(**data)
    return run

# Manipulation detector (heuristic)

LOADED_ADJECTIVES = {"poison", "genocide", "evil", "fake", "hoax"}

def compute_manipulation_score(text: str) -> ManipulationScore:
    words = re.findall(r"\w+", text)
    if not words:
        return ManipulationScore(manipulation_score=0.0, explanation="empty")
    caps_frac = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
    punct = sum(text.count(ch) for ch in ["!", "?"])
    loaded = sum(1 for w in words if w.lower() in LOADED_ADJECTIVES)
    repeated = 1 if re.search(r"[!?.]{3,}", text) else 0
    score = min(1.0, 0.4*caps_frac + 0.2*(punct/10) + 0.3*(loaded/5) + 0.1*repeated)
    return ManipulationScore(manipulation_score=round(score, 2), explanation="heuristic mix of caps, punctuation, loaded words, repetition")
