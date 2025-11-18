# Evidence Retrieval & Propaganda Detection Pipeline

This repository hosts a three-stage misinformation analysis stack that combines LLM-based fact verification, classical NLP classifiers, and a graph neural network (GAT) to triage social media conversations. The Streamlit UI in `app_complete.py` stitches these components into an interactive analyst workflow that can operate on both English and synthetic Hindi examples.

---

## 1. System Overview

| Stage | Purpose | Key Assets |
| --- | --- | --- |
| **Stage 1 – Claim Verification** | Detects factual claims, retrieves evidence, and produces a policy-compliant verdict (high_conf_true / high_conf_fake / send_downstream). | `src/graph.py`, `src/agents.py`, `.github/copilot-instructions.md` (prompt spec), Gemini 2.5 Flash (or local echo) |
| **Stage 2 – Propaganda + Emotion** | Scores root text with propaganda detectors (English + Hindi LoRA), summarizes reply emotions via GoEmotions→Ekman collapse, and fuses everything with a meta-classifier. | `eng_prop_model/`, `hprop-lora-adapter/`, `eng_emo_model/`, `models/meta_classifier.joblib`, `scripts/build_meta_dataset.py` |
| **Stage 3 – Graph Neural Network** | Refines the decision by looking at graph structure learned from 1,154 historical conversations; new samples are connected via K-nearest-neighbor edges. | `artifacts/gat_propaganda/`, `prop_datasets/tree_width/`, `SentenceTransformer paraphrase-multilingual-mpnet-base-v2` |

The Streamlit frontend orchestrates all stages, handles retries, visualizes attention weights, and exposes pre-crafted Hindi scenarios that demonstrate the full pipeline.

---

## 2. Pipeline Details

### 2.1 Stage 1 – Evidence Retrieval & Claim Verification
1. **Claim Detector** – Lightweight classifier filters out obvious opinions/intents before spending retrieval budget.
2. **Claim Extractor** – Token-level mask extractor (few-shot prompt) produces normalized English claims, confidences, and translations when necessary.
3. **Evidence Retriever** – Searches authoritative sources in cascading order: Google Fact Check API → Gemini grounded search → local BM25/FAISS knowledge base.
4. **Claim Verifier** – Gemini 2.5 Flash evaluates each claim against retrieved snippets, labeling SUPPORT / REFUTE / NEUTRAL per source.
5. **Final Decider** – Applies strict business rules (see `.github/copilot-instructions.md`) to classify the entire post and decides whether downstream propaganda analysis is required.
6. **Manipulation Heuristic** – Fast lexical scorer highlights emotionally manipulative posts even when claims look true.

Outputs flow into the Streamlit UI as: extracted claims, evidence counts, claim-level metrics, final decision, and manipulation warnings.

### 2.2 Stage 2 – Propaganda + Emotion Fusion
1. **Propaganda Classifiers**
   - English base model: `eng_prop_model/SemEval_Trained_Intermediate(final)` (Transformers sequence classifier fine-tuned on SemEval propaganda data).
   - Hindi adapter: `hprop-lora-adapter/` LoRA weights trained on synthetic + curated Hindi samples using the English backbone.
2. **Translation Strategy** – Hindi replies are translated to English on-demand via `googletrans` to avoid loading large Marian MT models, keeping memory low in Streamlit deployments.
3. **Emotion Aggregation** – Every reply is scored with the GoEmotions encoder (`eng_emo_model`). Scores are collapsed to Ekman clusters, aggregated (mean/variance/entropy), and fed into the meta-classifier.
4. **Meta-Classifier** – Logistic model (`models/meta_classifier.joblib`) trained on `prop_datasets/tree_width/meta_features.csv` fuses text_score + emotion stats. Training notebook/scripts live in `scripts/`.

Stage 2 provides the meta probability used both for the UI verdict and as an additional feature for Stage 3.

### 2.3 Stage 3 – Graph Attention Network (GAT)
1. **Graph Construction** – 1,154 conversations from `prop_datasets/tree_width/merged_conversations.jsonl` were converted into a graph where nodes represent conversations and edges capture reply-tree proximity + temporal overlap.
2. **Node Features** – Concatenate multilingual MPNet embeddings (768 dims) with the Stage-2 meta probability (1 dim) = 769-dim node features stored in `artifacts/gat_propaganda/node_embeddings.pt`.
3. **Model** – Three-layer `torch_geometric.nn.GATConv` network trained offline; checkpoints live in `artifacts/gat_propaganda/gat_model.pt`.
4. **Dynamic Node Injection** – New/Hindi samples are embedded on the fly, connected to the K=5 most similar historical nodes via cosine similarity, and evaluated without retraining. The UI labels these as “🆕 Synthetic node”.
5. **Visualization** – Stage 3 renders an attention subgraph, top neighbors, and their text previews, helping analysts inspect why a sample was flagged.

---

## 3. Datasets & Sample Assets

| Dataset / Asset | Location | Description |
| --- | --- | --- |
| `prop_datasets/tree_width/merged_conversations.jsonl` | Root tweet + replies with ground-truth propaganda labels used for GAT and UI samples. |
| `prop_datasets/tree_width/meta_features.csv` | Stage-2 training matrix (text predictor + emotion aggregates + labels). |
| `prop_datasets/time_order/*` | Additional chronological splits for experimentation. |
| Synthetic Hindi threads | Embedded directly in `app_complete.py` and exposed via Streamlit selector for demo purposes. |
| Hugging Face models | `eng_prop_model/`, `eng_emo_model/`, `hprop-lora-adapter/` store tokenizers + weights for offline use. |

> **Note**: All datasets stay local to avoid redistributing copyrighted content. Replace them with your own conversations by matching the same schema (tweet_id, root_text, replies[], true_label).

---

## 4. Models & Components

| Component | Source | Notes |
| --- | --- | --- |
| Claim pipeline LLM | Gemini 2.5 Flash (via `src/gemini_client.py`) | Falls back to an echo LLM when `GEMINI_API_KEY` is absent for offline demos. |
| Claim detection/extraction prompts | `.github/copilot-instructions.md` | Contains full few-shot templates and acceptance criteria. |
| Evidence retrieval | `src/agents.py` | Wraps Google Fact Check API, Gemini Search, and KB stubs. |
| Propaganda detectors | `eng_prop_model/`, `hprop-lora-adapter/` | English backbone + Hindi LoRA adapter loaded with PEFT. |
| Emotion encoder | `eng_emo_model/` | GoEmotions → Ekman collapse implemented in `app_complete.py`. |
| Meta-classifier | `models/meta_classifier.joblib` | Logistic regression (sklearn). |
| Sentence embeddings | `SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')` | Used for Stage 2 features and Stage 3 nodes. |
| Graph neural network | `artifacts/gat_propaganda/gat_model.pt` | 3-layer GAT with attention visualization support. |
| Translation | `googletrans==3.1.0a0` | Keeps footprint small vs. Marian MT. |

---

## 5. Evaluation & Metrics

| Component | Dataset / Script | Key Metrics |
| --- | --- | --- |
| Meta-classifier fusion | `scripts/test_meta_classifier.py` on `prop_datasets/tree_width/meta_features.csv` (1,154 balanced samples) | Accuracy **0.660**, F1 **0.649** at threshold 0.5. Best F1 **0.693** at threshold 0.4; precision/recall ≈0.65. |
| GAT (Stage 3) | Training summary in `artifacts/gat_propaganda/README.md` (80/10/10 split on the same graph) | Test Accuracy **81.9%**, F1 **83.2%**, AUC **86.2%**, calibrated threshold **0.55**. +22% accuracy and +28% F1 over the meta-classifier baseline. |
| Baselines (from artifacts README) | Meta-classifier vs. text-only models | Meta baseline: Accuracy **67%**, F1 **65%**, AUC ~70%. Text-only baseline: Accuracy **54%**, F1 **40%**, AUC ~55%. |

Additional highlights pulled directly from `artifacts/gat_propaganda/README.md`:
- Graph covers 1,154 conversations, 29,406 undirected edges, average degree 50.96, attention weights μ≈0.039.
- Training used focal loss (α=0.75, γ=1.5), Adam (lr=5e-3), dropout 0.4, early stopping on validation AUC (patience 50), no self-loops.
- Best validation AUC **84.5%** around epoch 199; learned edges, embeddings, and attention summaries are stored next to `gat_model.pt` for reproducibility.
---

## 6. Setup & Usage

### 6.1 Install Dependencies
```powershell
pip install -r requirements.txt
```

Large models (propaganda, emotion, LoRA, GAT) are already committed for offline use. GPU is optional but speeds up Stage 2.

### 6.2 Environment Variables
```powershell
$env:GEMINI_API_KEY="your_api_key"
$env:GEMINI_MODEL="gemini-2.5-flash"   # optional override
$env:GOOGLE_FACT_CHECK_API_KEY="optional"
```
Without keys the verification stage falls back to a deterministic echo model for demos.

### 6.3 Run the Streamlit UI
```powershell
streamlit run app_complete.py
```
Features:
- Toggle stages on/off from the sidebar.
- Pick random samples, specify indices, or load curated Hindi examples.
- Inspect claims, evidence, propaganda scores, emotion table, and the GAT attention visualization.

### 6.4 Command-line Demo (LLM pipeline only)
```powershell
python src/main.py
```
Runs the multi-agent verification graph with your `GEMINI_API_KEY` or the offline echo fallback.

### 6.5 Testing
```powershell
pytest -q
# or evaluate the meta-classifier directly
python scripts/test_meta_classifier.py
```

---

## 7. Extending the System
- **New Data**: Drop JSONL conversations into `prop_datasets/tree_width/`, run `scripts/build_meta_dataset.py`, retrain the meta-classifier, and regenerate embeddings/GAT checkpoints.
- **Retrieval Plugins**: Implement your own connectors in `src/agents.py` for newsroom databases or custom knowledge bases.
- **LLM Choice**: Swap Gemini with any LangChain-compatible model by passing `llm_override` to `build_graph()` (useful for on-prem deployments).
- **Visualization**: Stage 3 currently shows max-8 neighbors. Adjust `build_neighbor_graph_figure()` if you need larger ego nets.

---

## 8. Repository Structure (excerpt)
```
├── app_complete.py              # Streamlit UI orchestrating all stages
├── src/
│   ├── agents.py                # Claim detection, retrieval, verification builders
│   ├── graph.py                 # LangGraph/simple runner for Stage 1
│   └── schemas.py               # Pydantic models shared across agents
├── scripts/
│   ├── build_meta_dataset.py    # Aggregates emotion features + text scores
│   ├── test_meta_classifier.py  # Prints metrics summarized above
│   └── train_gat_propaganda.py  # Training entry point for GAT (optional)
├── artifacts/gat_propaganda/    # Trained GAT weights + embeddings + learned graph
├── eng_prop_model/, hprop-lora-adapter/, eng_emo_model/
└── docs/                        # Original instructions, guardrails, notebooks
```

---

## 9. Contact & Next Steps
- File issues/ideas in this repo or extend the Streamlit app with your own datasets.
- Consider hooking Stage 3 decisions into moderation tooling or dashboards for human analysts.
- Contributions welcome: add tests, improve retrieval connectors, or port the UI to FastAPI.

Happy fact-checking! 🚀
