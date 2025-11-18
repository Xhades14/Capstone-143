# Final Decision Agent Rules

## Overview

The Final Decision Agent produces one of three labels:
- **`high_conf_true`**: High confidence the post is TRUE
- **`high_conf_fake`**: High confidence the post is FAKE
- **`send_downstream`**: Uncertain, needs further analysis (propaganda detection, etc.)

## Decision Rules (Applied in Order)

### Rule 1: Missing Data → Send Downstream
**Condition:** ANY claim_score is null OR retrieval_coverage < 0.5  
**Decision:** `send_downstream`  
**Reason:** Insufficient evidence to make confident decision

**Example:**
```
Input: "Scientists claim the earth is flat"
Evidence: No results found (retrieval_coverage = 0.0)
Decision: send_downstream - "Low coverage or missing scores"
```

### Rule 2: Strong Refutation → High Confidence FAKE
**Condition:** ANY claim has (claim_score <= 0.10 AND refute_confidence >= 0.8)  
**Decision:** `high_conf_fake`  
**Reason:** Evidence strongly refutes at least one claim

**Example:**
```
Input: "mRNA vaccines cause cancer"
Claim Score: 0.05 (strongly refuted)
Refute Confidence: 0.95
Decision: high_conf_fake
```

### Rule 3: Strong Support → High Confidence TRUE
**Condition:** ALL claims have (claim_score >= 0.90 AND support_confidence >= 0.8 AND manipulation_score < 0.6)  
**Decision:** `high_conf_true`  
**Reason:** All claims strongly supported with low manipulation

**Example:**
```
Input: "COVID-19 vaccines are effective at preventing severe illness"
Claim Score: 0.95 (strongly supported)
Support Confidence: 0.92
Manipulation Score: 0.15 (low)
Decision: high_conf_true
```

### Rule 4: Neutral Claims + Manipulation → Send Downstream
**Condition:** Neutral claims (0.3 <= claim_score <= 0.7) AND manipulation_score >= 0.3  
**Decision:** `send_downstream`  
**Reason:** Ambiguous evidence combined with manipulation tactics

**Example:**
```
Input: "The government is HIDING the truth about vaccines!!!"
Claim Score: 0.55 (neutral)
Manipulation Score: 0.65 (high - caps, exclamation, loaded words)
Decision: send_downstream - "Needs propaganda detection"
```

### Rule 5: High Manipulation → Send Downstream
**Condition:** manipulation_score >= 0.6 (regardless of claim scores)  
**Decision:** `send_downstream`  
**Reason:** High manipulation detected, needs specialized analysis

**Example:**
```
Input: "BREAKING!!! THEY DON'T WANT YOU TO KNOW!!! Climate change is a HOAX!!!"
Claim Score: 0.20 (refuted)
Manipulation Score: 0.85 (very high - all caps, multiple exclamations, loaded words)
Decision: send_downstream - "High manipulation, needs propaganda analysis"
```

### Rule 6: Default → Send Downstream
**Condition:** All other cases  
**Decision:** `send_downstream`  
**Reason:** Doesn't meet criteria for high-confidence true/fake

## Manipulation Score Calculation

The manipulation score is a heuristic based on:

1. **CAPS (40% weight)**: Fraction of words in all caps
2. **Punctuation (20% weight)**: Number of `!` and `?` marks
3. **Loaded Words (30% weight)**: Presence of emotional/manipulative words
   - Examples: poison, genocide, evil, fake, hoax
4. **Repetition (10% weight)**: Repeated punctuation like `!!!` or `???`

**Formula:**
```
score = min(1.0, 0.4*caps_frac + 0.2*(punct/10) + 0.3*(loaded/5) + 0.1*repeated)
```

## Examples for Each Category

### Example 1: High Confidence TRUE
```
Input: "The COVID-19 vaccine has been approved by the FDA"
Claims: 1
Claim Score: 0.98 (strongly supported)
Support Confidence: 0.96
Refute Confidence: 0.02
Manipulation Score: 0.10 (low)
Retrieval Coverage: 1.0 (100%)
Decision: high_conf_true
Why: "All claims strongly supported with low manipulation"
```

### Example 2: High Confidence FAKE
```
Input: "I read that mRNA vaccines cause cancer!"
Claims: 1
Claim Score: 0.08 (strongly refuted)
Support Confidence: 0.05
Refute Confidence: 0.92
Manipulation Score: 0.25
Retrieval Coverage: 1.0 (100%)
Decision: high_conf_fake
Why: "At least one claim strongly refuted by evidence"
```

### Example 3: Send Downstream - Low Coverage
```
Input: "A new study shows that drinking coffee prevents Alzheimer's"
Claims: 1
Claim Score: null (no evidence found)
Retrieval Coverage: 0.0 (0%)
Manipulation Score: 0.15
Decision: send_downstream
Why: "Low coverage or missing scores"
```

### Example 4: Send Downstream - High Manipulation
```
Input: "WAKE UP!!! Big Pharma is POISONING you with vaccines!!!"
Claims: 1
Claim Score: 0.15 (refuted)
Support Confidence: 0.10
Refute Confidence: 0.85
Manipulation Score: 0.75 (high - caps, loaded words "POISONING", exclamations)
Retrieval Coverage: 1.0
Decision: send_downstream
Why: "High manipulation detected, needs propaganda analysis"
```

### Example 5: Send Downstream - Neutral + Manipulation
```
Input: "Some experts say climate change might not be real..."
Claims: 1
Claim Score: 0.45 (neutral - mixed evidence)
Support Confidence: 0.40
Refute Confidence: 0.55
Manipulation Score: 0.35 (moderate - hedge words, vague)
Retrieval Coverage: 0.8
Decision: send_downstream
Why: "Neutral claims with moderate manipulation"
```

### Example 6: Send Downstream - Ambiguous
```
Input: "Scientists are divided on the safety of artificial sweeteners"
Claims: 1
Claim Score: 0.50 (neutral)
Support Confidence: 0.45
Refute Confidence: 0.48
Manipulation Score: 0.20 (low)
Retrieval Coverage: 1.0
Decision: send_downstream
Why: "Ambiguous evidence, requires deeper analysis"
```

## Decision Flow Chart

```
Start
  ↓
Is any claim_score NULL or coverage < 0.5?
  YES → send_downstream (Rule 1)
  NO ↓
  
Any claim with score ≤ 0.10 AND refute ≥ 0.8?
  YES → high_conf_fake (Rule 2)
  NO ↓
  
ALL claims with score ≥ 0.90 AND support ≥ 0.8 AND manipulation < 0.6?
  YES → high_conf_true (Rule 3)
  NO ↓
  
Neutral claims (0.3-0.7) AND manipulation ≥ 0.3?
  YES → send_downstream (Rule 4)
  NO ↓
  
Manipulation ≥ 0.6?
  YES → send_downstream (Rule 5)
  NO ↓
  
Otherwise → send_downstream (Rule 6)
```

## Key Insights

1. **Conservative by Design**: System defaults to `send_downstream` in most cases
2. **High Confidence Rare**: Only clear-cut cases get `high_conf_true` or `high_conf_fake`
3. **Manipulation Aware**: High manipulation triggers downstream analysis regardless of evidence
4. **Coverage Matters**: Poor evidence retrieval means uncertain decision
5. **All vs Any Logic**: 
   - `high_conf_true`: ALL claims must be strongly supported
   - `high_conf_fake`: ANY claim strongly refuted is enough
   - This is asymmetric but appropriate for fact-checking

## Downstream Tasks

When `send_downstream` is returned, the post should be sent for:
- **Propaganda Detection**: Identify manipulation techniques
- **Emotional Analysis**: Detect fear-mongering, outrage
- **Source Analysis**: Check credibility of cited sources
- **Context Analysis**: Understand broader narrative
- **Human Review**: Complex cases need expert judgment
