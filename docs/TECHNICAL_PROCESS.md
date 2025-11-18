# Technical Process Specification: Evidence Retrieval & Propaganda Detection Pipeline

## Abstract
We present a three-stage misinformation triage pipeline that combines deterministic retrieval heuristics, instruction-following large language models (LLMs), classical propaganda detectors, affect modeling, and graph neural reasoning. The system ingests a cleaned microblog post (2-4 sentences), extracts verifiable claims, retrieves multi-source evidence, issues high-confidence policy-compliant verdicts, and, when required, escalates posts into a hybrid propaganda detection stack. This document details the full data flow, modeling stack, hyperparameters, and evaluation results so that practitioners can reproduce the pipeline or adapt it to adjacent misinformation workloads.

Overall flow: Input post -> Stage 1 (claim-centric fact verification) -> Stage 2 (propaganda and emotion fusion) -> Stage 3 (graph attention inference) -> Analyst interpretation.

- **Stage 1: Claim Verification.** Detects factual statements, retrieves authoritative evidence, and applies deterministic decision rules to emit high-confidence truth labels when possible.
- **Stage 2: Propaganda + Emotion Fusion.** Scores root text with multilingual propaganda detectors, summarizes emotional dynamics across replies, and fuses these signals via a meta-classifier.
- **Stage 3: Graph Attention Reasoning.** Places escalated posts into a semantic similarity graph of historical conversations, leveraging attention-based message passing to refine predictions and expose neighborhood explanations.



## 2. Data Assets and Preprocessing

### 2.1 Conversation Corpus
- **Collection**: 1,154 conversation roots with complete reply trees, spanning balanced propaganda and non-propaganda annotations.
- **Splits**: Stratified 80/10/10 train/validation/test partitions to ensure label parity.
- **Schema**: Each record stores the root text, structured replies, engagement statistics, and gold labels required for Stages 2 and 3 feature derivation.
- **Synthetic Hindi Threads**: Curated exemplars stress-test multilingual robustness without redistributing sensitive user content.

### 2.2 Meta-Feature Matrix
- **Feature Space**: Combines Stage-2 text logits, propaganda probabilities, GoEmotions-to-Ekman aggregates (mean, variance, entropy), and engagement proxies such as reply volume and depth.
- **Use Cases**: Serves as supervised training data for the meta-classifier and supplies the 1-dim auxiliary signal appended to Stage-3 node embeddings.

### 2.3 Knowledge Sources
- **Fact-Check API**: Queries normalized claims via Google Fact Check endpoints to harvest structured verdicts from professional fact-checkers.
- **Grounded Web Search**: Gemini-powered search retrieves authoritative web snippets when the API lacks coverage, prioritizing public health agencies and reputable newsrooms.
- **Local Knowledge Base**: BM25/dense retrieval over curated encyclopedic and fact-check corpora provides an offline fallback and enables deduplication against previously validated narratives.

### 2.4 Embeddings
- **Sentence Encoder**: The pipeline relies on multilingual MPNet sentence embeddings (768 dimensions) for both Stage-2 feature construction and Stage-3 node representations.
- **Normalization**: Embeddings and meta-classifier scores are standardized independently (per-feature z-score) before graph training to prevent high-dimensional text vectors from overwhelming scalar meta signals.

## 3. Stage 1 - Claim-Centric Verification

Stage 1 is executed as a deterministic agent workflow with the following components:

### 3.1 Claim Detector
- **Inputs**: Raw cleaned post (2-4 sentences) and language hint if known.
- **Model**: Lightweight classifier (logistic) that flags presence of factual claims to avoid unnecessary LLM prompts.
- **Output**: Boolean indicator primarily used for monitoring; the pipeline proceeds regardless to ensure recall.

### 3.2 Claim Span Extraction
- **Prompt Spec**: Few-shot instructions mandate whitespace tokenization, binary masks, normalized English claims, and calibrated confidence scores.
- **Filtering Policy**: Only claims with `confidence >= 0.3` and non-empty normalized text are admitted downstream, preventing opinionated spans from consuming retrieval budget.
- **Multilingual Handling**: Extractor emits `lang` and `english` translation fields. Hindi claims are translated inline to maintain consistent downstream queries.

### 3.3 Evidence Retrieval Cascade
1. **Fact Check API**: Called with the normalized claim; results tagged `source_type="factcheck_api"`.
2. **Gemini Web Search**: Invoked when the API misses; returns structured JSON with URL, snippet, domain, language.
3. **Local Knowledge Base**: FAISS/BM25 over curated corpora provides an offline fallback.

Results are deduplicated by `(domain, normalized_title)` similarity (Jaccard > 0.8) and truncated to top 5 per claim. Timeout budget per claim is 6s; partial retrievals are surfaced with `note="partial_timeout"`.

### 3.4 Claim Verification LLM
- **Model**: Gemini 2.5 Flash operating at temperature 0.
- **Prompt**: Few-shot specification ensures structured JSON verdicts per evidence snippet with labels {SUPPORT, REFUTE, NEUTRAL}, confidences, and `top_evidence_ids`.
- **Aggregation**: Claim-level `claim_score` computed from weighted votes (evidence score * verdict confidence) mapped to [0,1].
- **Fallback**: Deterministic echo LLM provides predictable outputs when remote models are unavailable.

### 3.5 Final Decision Agent
- **Rules**: Policy mandates the following order of evaluation:
  - Missing scores or low coverage -> `send_downstream`.
  - Any claim strongly refuted (`claim_score <= 0.10` and refuting evidence >= 2 domains) -> `high_conf_fake`.
  - All claims strongly supported (`claim_score >= 0.90`, support >= 2 domains, `manipulation_score < 0.6`) -> `high_conf_true`.
  - Manipulation heuristics (caps, punctuation, loaded lexicon, repetition) bias toward downstream routing when high.
- **Outputs**: Claim table, aggregated verdict, manipulation diagnostics, plus provenance (URLs, published dates).

### 3.6 Manipulation / Opinion Detector
- **Heuristic Score**: `score = min(1, 0.4*caps_frac + 0.2*(punct/10) + 0.3*(loaded/5) + 0.1*repeated)`.
- **Routing**: When `score >= 0.6`, pipeline tags post for propaganda analysis even if claims appear true.

## 4. Stage 2 - Propaganda and Emotion Fusion

Stage 2 operates when Stage 1 returns `send_downstream` or when analysts explicitly enable deeper inspection.

### 4.1 Propaganda Classifiers
- **English Backbone**: Transformer sequence classifier fine-tuned on SemEval-style propaganda corpora with SentencePiece tokenization and 0.1 dropout.
- **Hindi Adapter**: LoRA weights trained on synthetic + curated Hindi samples share the same backbone, enabling multilingual inference without duplicating parameters.
- **Scoring**: Root posts and salient replies pass through the appropriate head to produce propaganda probabilities, logits, and attention-span highlights.

### 4.2 Translation Strategy
- Lightweight neural machine translation converts Hindi replies into English before emotion and meta-feature extraction, avoiding the overhead of large offline MT models.

### 4.3 Emotion Aggregation
- **Model**: GoEmotions encoder producing 27-label distributions.
- **Collapse**: Mapped to Ekman families (Anger, Fear, Joy, Sadness, Surprise, Disgust, Neutral), then aggregated per conversation: mean, variance, entropy, and max per family.
- **Rationale**: Captures coordinated emotional tone shifts often associated with propaganda narratives.

### 4.4 Meta-Classifier
- **Model**: Logistic regression trained on the meta-feature matrix described in Section 2.2.
- **Inputs**: Propaganda probabilities, text logits, aggregated emotion statistics, and engagement descriptors.
- **Evaluation**:
  - Accuracy: 0.660 (threshold 0.5)
  - F1: 0.649
  - Best F1: 0.693 at threshold 0.4 (precision/recall ~0.65)
- **Outputs**: Calibrated probability plus feature attribution weights that highlight which signals tipped the decision.

## 5. Stage 3 - Graph Attention Network Reasoning

### 5.1 Graph Construction
- **Nodes**: 1,154 historical conversations (train/val/test) plus synthetic nodes injected at inference.
- **Edges**: 29,406 undirected edges from a cosine similarity graph built with `k=20` neighbors and similarity threshold 0.3.
- **Node Features**: Concatenation of 768-dim multilingual MPNet embeddings with 1-dim meta-classifier score (total 769 dims).
- **Dynamic Insertion**: New samples embed via the same encoder, normalized, and connected to `K=5` nearest historic nodes (cosine similarity), then flagged as synthetic nodes for interpretability.

### 5.2 Model Architecture
- **Layers**: 3 x `torch_geometric.nn.GATConv` with ELU activations.
  - Layer 1: 769 -> 128 dims, 8 heads, dropout 0.2, `add_self_loops=False`.
  - Layer 2: 128 -> 128 dims, 8 heads, dropout 0.2.
  - Layer 3: 128 -> 1 dim, 1 head, dropout 0.2.
- **Loss**: Focal Loss (alpha=0.75, gamma=1.5) to mitigate class imbalance and probability collapse.
- **Optimizer**: Adam (lr=5e-3), gradient clipping `max_norm=5.0`, early stopping on validation AUC (patience 50).
- **Implementation**: Trained with PyTorch Geometric using the configuration above.

### 5.3 Training Dynamics
- **Convergence**: Epoch 199 achieved highest validation AUC (84.5%).
- **Calibration**: Test threshold 0.55 yields balanced precision/recall; probability mean ~0.58 across splits.
- **Attention Insights**: Mean attention weight 0.0392 (std 0.0500); selective focus on high-similarity edges rather than uniform smoothing.

### 5.4 Performance
| Model | Accuracy | F1 | AUC |
| --- | --- | --- | --- |
| GAT (Stage 3) | **81.9%** | **83.2%** | **86.2%** |
| Meta-classifier baseline | 67.0% | 65.0% | ~70.0% |
| Text-only baseline | 54.0% | 40.0% | ~55.0% |

Stage 3 improves accuracy by 22 percentage points and F1 by 28 points over Stage 2 while delivering graph-level interpretability (neighbor table + attention heatmap).

## 6. Evaluation Protocol

- **Stage 1**: Manual acceptance tests on four canonical posts validate claim masks, evidence cascades, and deterministic decision rules.
- **Stage 2**: Holdout evaluation on the meta-feature matrix reports accuracy/F1 plus qualitative audits on curated Hindi and English conversations.
- **Stage 3**: Stratified 80/10/10 split underpins the reported 81.9%/83.2%/86.2% metrics, supplemented with diagnostics such as attention histograms, hub-node analysis, and learned edge inspection.
- **Pipeline QA**: An end-to-end integration test mocks retrieval/LLM endpoints to ensure routing logic and manipulation scores remain stable across releases.

## 7. Limitations and Future Work
- **LLM Dependence**: Claim extractor/verifier quality tied to Gemini availability; offline fallback is deterministic but not semantically accurate.
- **Retrieval Coverage**: Fact Check API coverage skews toward English-language narratives; extending local KB with multilingual sources would improve recall.
- **Propaganda Labels**: Synthetic Hindi data may not capture full linguistic nuance; crowdsourced annotations would enhance adapter robustness.
- **Graph Stationarity**: Static historic graph may drift as narratives evolve; periodic embedding refresh and edge recomputation recommended.
- **Explainability**: Stage 2 meta-classifier exposes feature weights, but Stage 3 attention requires careful interpretation; developing contrastive explanations is future work.