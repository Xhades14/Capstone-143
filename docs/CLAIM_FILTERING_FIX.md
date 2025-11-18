# Claim Filtering Fix - October 16, 2025

## Problem Identified

**Contradiction between agents:**
- Input: "trump is one of the top-10 most handsome people in the world"
- Claim Detection Agent: ✅ Yes (contains claims)
- Claim Extraction Agent: Extracted with confidence 0.15, printed "❌ No verifiable claim found"
- But claim was still added to pipeline and processed downstream
- UI showed contradictory information

## Root Cause

**Low-confidence claims were not filtered at extraction:**
1. `build_claim_extractor`: Always added claims to list regardless of confidence
2. `build_evidence_retriever`: Had a check `if claim.confidence < 0.3` but claim already in pipeline
3. Result: Low-confidence opinion statements processed as verifiable claims

## Solution Implemented

### 1. Filter at Extraction Stage (src/agents.py)

**Before:**
```python
if c.get("normalized_claim"):
    print(f"   ✅ Claim...")
else:
    print(f"   ❌ No verifiable claim found...")
claims.append(Claim(**c))  # Always added!
```

**After:**
```python
confidence = c.get('confidence', 0)
if c.get("normalized_claim") and confidence >= 0.3:
    print(f"   ✅ Claim...")
    claims.append(Claim(**c))  # Only add if >= 0.3 confidence
else:
    print(f"   ❌ Filtered out low-confidence/non-verifiable claim...")
    # Not added to claims list
```

### 2. Cleaned Up Evidence Retrieval (src/agents.py)

**Before:**
```python
if not claim.normalized_claim or claim.confidence < 0.3:
    results.append(ClaimEvidence(..., note="no_claim_or_low_confidence"))
    continue
```

**After:**
```python
if not claim.normalized_claim:
    results.append(ClaimEvidence(..., note="no_claim_text"))
    continue
# No confidence check needed - already filtered at extraction
```

### 3. UI Fix (app.py)

**Before:**
- Showed claim detection based on first agent's `has_claim` flag
- Could say "Yes" even when no claims extracted

**After:**
```python
# Show actual result based on what was extracted
claims = results.get("claims", [])
has_claims_extracted = len(claims) > 0
render_claim_detection(has_claims_extracted, ...)

# If no claims extracted, show message and stop
if not claims:
    st.info("💭 This text appears to be an opinion or personal statement...")
    return
```

## Benefits

1. **Pipeline Efficiency**: Low-confidence claims filtered early, don't waste resources
2. **Consistency**: No contradiction between detection and extraction
3. **Cleaner Code**: Single source of truth for filtering (extraction stage)
4. **Better UX**: UI shows accurate "Yes/No" based on actual extracted claims

## Confidence Threshold

**Current threshold: 0.3**
- Claims with confidence >= 0.3: Considered verifiable, processed through pipeline
- Claims with confidence < 0.3: Filtered out as opinion/non-verifiable

This threshold can be adjusted in `src/agents.py` if needed.

## Test Case Results

### Before Fix:
```
Input: "trump is one of the top-10 most handsome people in the world"
Detection: ✅ Yes (contradiction!)
Extraction: confidence 0.15, printed "No verifiable claim found" but still added
UI: Showed "Contains claim: Yes" → confusing
```

### After Fix:
```
Input: "trump is one of the top-10 most handsome people in the world"
Detection: Checks actual claims list
Extraction: confidence 0.15 → filtered out, NOT added to claims list
UI: Shows "Contains claim: No" → correct!
Message: "💭 This text appears to be an opinion or personal statement..."
```

## Files Modified

1. `src/agents.py` - Added confidence filtering in `build_claim_extractor`
2. `src/agents.py` - Removed redundant check in `build_evidence_retriever`
3. `app.py` - UI now checks actual claims list instead of detection flag

## No Breaking Changes

- Graph flow unchanged (src/graph.py) - works correctly with filtered claims
- Schemas unchanged (src/schemas.py)
- API unchanged - filtering is internal pipeline logic
