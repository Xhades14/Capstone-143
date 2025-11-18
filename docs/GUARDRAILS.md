# Deterministic Guardrails

This document outlines all the guardrails implemented to ensure deterministic and consistent outputs from the AI agents.

## 1. LLM Configuration (Gemini Client)

**File**: `src/gemini_client.py`

- **temperature**: `0` - Completely deterministic token selection
- **topK**: `1` - Only select the most likely token
- **topP**: `1` - No nucleus sampling
- **maxOutputTokens**: `4096` - Sufficient for complete responses
- **timeout**: `30` seconds - Prevents hanging

## 2. Agent 0: Claim Detection Filter

**Guardrails**:
- ✅ Explicit JSON format specification
- ✅ "DO NOT" instructions for markdown blocks
- ✅ Clear boolean values (lowercase true/false)
- ✅ Comprehensive examples of what to include/exclude
- ✅ Deterministic classification criteria

**Key Instructions**:
```
1. Output ONLY valid JSON in this EXACT format
2. DO NOT add markdown code blocks (```json)
3. DO NOT add any text before or after the JSON
4. Use ONLY lowercase true/false (not True/False)
5. Always include both fields: has_claim and reasoning
```

## 3. Agent 1: Claim Span Identification

**Guardrails**:
- ✅ Strict JSON array format
- ✅ Exact field specifications with data types
- ✅ Validation rules for consistency
- ✅ Few-shot examples with exact formatting
- ✅ Confidence scoring guidelines

**Key Validation Rules**:
```
- If span_text is empty, token_mask must be all zeros
- If span_text is not empty, at least one token_mask value must be 1
- token_mask length EXACTLY equals tokens length
- confidence: 0.90-0.99 for clear claims, 0.10-0.30 for no claims
```

## 4. Agent 3: Evidence Verifier

**Guardrails**:
- ✅ Deterministic scoring rules
- ✅ Explicit confidence ranges
- ✅ Required output format with ALL fields
- ✅ Consistency emphasis
- ✅ Clear verdict categories (SUPPORT/REFUTE/NEUTRAL)

**Deterministic Scoring Rules**:
```
- Evidence explicitly FALSE → refute_confidence=0.95-1.0, claim_score=0.0-0.1
- Evidence explicitly TRUE → support_confidence=0.95-1.0, claim_score=0.9-1.0
- Evidence unclear/mixed → both confidences 0.3-0.5, claim_score=0.4-0.6
- Evidence unrelated → both confidences 0.0, claim_score=0.5
```

**Key Emphasis**:
```
BE CONSISTENT: Same evidence should always produce same scores.
```

## 5. Agent 4: Final Decision

**Guardrails**:
- ✅ Numbered, ordered decision rules
- ✅ Exact thresholds for each decision
- ✅ Required JSON fields specification
- ✅ URL validation (only use provided URLs)
- ✅ Consistency emphasis

**Deterministic Decision Rules (applied in order)**:
```
Rule 1: If ANY claim_score is null OR retrieval_coverage < 0.5 → send_downstream
Rule 2: If ANY claim (score <= 0.10 AND refute >= 0.8) → high_conf_fake
Rule 3: If ALL claims (score >= 0.90 AND support >= 0.8 AND manip < 0.6) → high_conf_true
Rule 4: If neutral claims (0.3 <= score <= 0.7) AND manip >= 0.3 → send_downstream
Rule 5: If manip >= 0.6 (regardless of scores) → send_downstream
Rule 6: Otherwise → send_downstream
```

**Key Emphasis**:
```
BE CONSISTENT: Same input should always produce same output.
```

## 6. JSON Parsing Robustness

**File**: `src/agents.py` - `_try_parse_json()`

**Features**:
- ✅ Removes markdown code blocks before parsing
- ✅ Extracts JSON from wrapped text
- ✅ Handles truncated responses with regex patterns
- ✅ Prioritizes full objects over arrays
- ✅ Graceful fallback for partial responses

## 7. Grounded Search Configuration

**File**: `src/gemini_search.py`

**Credibility Guardrails**:
- ✅ Prioritizes authoritative sources:
  - Government health agencies (CDC, NIH, FDA, WHO)
  - Major medical journals (NEJM, Lancet, JAMA, BMJ)
  - Academic institutions (.edu domains)
  - Fact-checking organizations (FactCheck.org, Snopes)
  - Reputable news (Reuters, AP, BBC)
  - Scientific organizations (Nature, Science)
- ✅ Explicitly avoids blogs, social media, opinion pieces
- ✅ Uses Google's grounding API for verified sources

## Benefits of These Guardrails

1. **Reproducibility**: Same input produces same output across runs
2. **Reliability**: Reduced hallucinations and format errors
3. **Validation**: Strong type checking and field requirements
4. **Transparency**: Clear decision rules that can be audited
5. **Quality**: Authoritative sources only for evidence
6. **Consistency**: Explicit instructions to be deterministic

## Testing Determinism

To verify determinism, run the same input multiple times:

```powershell
# Run 5 times and compare outputs
for ($i=1; $i -le 5; $i++) {
    python src/main.py > "output_$i.txt"
}
```

Outputs should be identical (except for timestamps and API latency).

## Maintenance

When updating prompts:
1. Keep "STRICT OUTPUT REQUIREMENTS" section
2. Maintain "DO NOT" instructions
3. Preserve numerical thresholds
4. Include "BE CONSISTENT" emphasis
5. Test with multiple runs to verify determinism
