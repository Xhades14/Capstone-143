# Evidence Retrieval and Verification Pipeline# Evidence Retrieval and Verification Pipeline



A production-ready multi-agent system for fact-checking social media content using AI-powered evidence retrieval and verification.A production-ready multi-agent system for fact-checking social media content using AI-powered evidence retrieval and verification.



## 🚀 Features## 🚀 Features



- **Claim Detection Pre-Filter**: Filters out opinions, jokes, and personal statements- **Automated Claim Extraction**: Token-level claim identification with confidence scoring

- **Automated Claim Extraction**: Token-level claim identification with confidence scoring- **Authoritative Evidence Retrieval**: Sources from medical journals, fact-checking sites, CDC, WHO

- **Authoritative Evidence Retrieval**: Prioritizes CDC, NIH, medical journals, fact-checking sites- **AI-Powered Verification**: Gemini 2.5-Flash for intelligent claim verification

- **AI-Powered Verification**: Gemini 2.5-Flash with deterministic settings (temperature=0)- **Safe Decision Making**: Conservative approach flags uncertain content for human review

- **Confidence-Based Scoring**: Uses support/refute confidence instead of domain counts

- **Streamlit Frontend**: Interactive web interface for real-time fact-checking## 🏗️ Architecture



## 🏗️ Architecture**Multi-Agent Pipeline:**

1. **Claim Extractor** → Identifies claims with token-level precision

**Multi-Agent Pipeline:**2. **Evidence Retriever** → Sources from Google Fact Check API + Gemini grounded search  

1. **Claim Detector** → Pre-filters verifiable claims vs opinions3. **Claim Verifier** → Evaluates evidence stance and confidence

2. **Claim Extractor** → Identifies claims with token-level precision4. **Final Decider** → Makes policy-compliant content decisions

3. **Evidence Retriever** → Sources from Gemini grounded search

4. **Claim Verifier** → Evaluates evidence with confidence-based scoring**Evidence Sources:**

5. **Final Decider** → Makes deterministic policy-compliant decisions- Google Fact Check API (primary)

- Gemini grounded search with real-time web results

## 🛠️ Installation- Authoritative domains: CDC, BMJ, FactCheck.org, medical institutions



```powershell## Quickstart (Windows PowerShell)

pip install -r requirements.txt```powershell

$env:GEMINI_API_KEY="your_api_key_here"# Option A: Global Python (no virtual env)

```pip install -r requirements.txt

pytest -q

## 🚀 Quick Start```



### Streamlit Web Interface (Recommended)Optional: You can set API keys as environment variables (no .env required):

```powershell- GEMINI_API_KEY (for real Gemini model calls)

streamlit run app.py- GEMINI_MODEL (optional, defaults to `gemini-2.5-flash`)

```- GOOGLE_FACT_CHECK_API_KEY (optional; retrieval placeholder)

If keys aren’t set, the app falls back to a local echo model for offline runs.

### Command Line

```powershellGemini-only

python src/main.py- This scaffold uses Gemini only (no OpenAI). The `src/gemini_client.py` adapter calls the Generative Language REST API.

```- To use a specific Gemini model, set `GEMINI_MODEL` or pass `gemini_model` to `build_graph()`.



### Python API## Run demo

```python```powershell

from src.graph import PipelineApp# Runs with a local echo LLM if no GEMINI_API_KEY is set

app = PipelineApp()python -c "from src.main import run_demo; run_demo()"

result = app.invoke({"post_id": "1", "post_text": "mRNA vaccines cause cancer"})```

```

## Configuration

## 📖 Documentation- Model: `build_graph(gemini_model="gemini-2.5-flash", llm_override=None)` lets you inject a custom LLM or select a Gemini model. Without `GEMINI_API_KEY`, a local echo model is used for offline runs.

- Retrieval: `src/agents.py` contains placeholders. Wire in Google Fact Check, Gemini grounded search, and FAISS/BM25 indices as needed.

- **GUARDRAILS.md**: Deterministic measures

- **src/agents.py**: Agent implementations## Tests

- **src/schemas.py**: Data models- `tests/test_pipeline.py` runs a smoke test using `FakeListChatModel` to avoid external calls.



## 🎯 Example Claims## Notes

- This is a scaffold to operationalize the instructions. Replace retrieval stubs and refine prompts to meet acceptance tests in `.github/copilot-instructions.md`.

**Factual (processed):**
- "mRNA vaccines cause cancer"

**Non-claims (filtered):**
- "I will never take a vaccine"
