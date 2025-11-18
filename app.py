"""
Streamlit Frontend for Evidence Retrieval and Verification Pipeline
"""
import streamlit as st
import sys
import os
from typing import Dict, Any, List
import json

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import from src
from src.graph import build_graph
from src.schemas import Claim, ClaimEvidence, EvidenceItem, VerifierResult, FinalDecision

# Create a wrapper class for consistency
class PipelineApp:
    def __init__(self):
        self.app = build_graph()
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.app.invoke(state)

# Alias for easier reference in the code
Evidence = ClaimEvidence

# Page configuration
st.set_page_config(
    page_title="Fact Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .claim-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .evidence-card {
        background-color: #ffffff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .verdict-support {
        color: #28a745;
        font-weight: bold;
    }
    .verdict-refute {
        color: #dc3545;
        font-weight: bold;
    }
    .verdict-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_app' not in st.session_state:
    st.session_state.pipeline_app = PipelineApp()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def render_verdict_badge(label: str) -> str:
    """Render a colored verdict badge"""
    color_map = {
        "high_conf_true": ("✅", "verdict-support", "High Confidence TRUE"),
        "high_conf_fake": ("❌", "verdict-refute", "High Confidence FAKE"),
        "send_downstream": ("⚠️", "verdict-neutral", "Needs Review"),
        "SUPPORT": ("✅", "verdict-support", "SUPPORTED"),
        "REFUTE": ("❌", "verdict-refute", "REFUTED"),
        "NEUTRAL": ("⚠️", "verdict-neutral", "NEUTRAL")
    }
    emoji, css_class, text = color_map.get(label, ("❓", "verdict-neutral", label))
    return f"{emoji} <span class='{css_class}'>{text}</span>"

def render_confidence_bar(confidence: float, label: str = "") -> None:
    """Render a confidence progress bar"""
    color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
    st.markdown(f"**{label}**" if label else "")
    st.progress(confidence)
    st.caption(f"{confidence:.2%}")

def render_claim_detection(has_claim: bool, reasoning: str) -> None:
    """Render claim detection results"""
    with st.expander("🔍 Claim Detection", expanded=False):
        status = "Yes" if has_claim else "No"
        if has_claim:
            st.success(f"✅ Contains claim: **{status}**")
        else:
            st.warning(f"❌ Contains claim: **{status}**")

def render_claims(claims: List[Claim]) -> None:
    """Render extracted claims"""
    st.subheader("📋 Extracted Claims")
    
    if not claims:
        st.info("No claims extracted from the text.")
        return
    
    # Filter out any empty claims that somehow got through
    valid_claims = []
    for claim in claims:
        c = claim.model_dump() if hasattr(claim, 'model_dump') else claim
        normalized = (c.get('normalized_claim') or '').strip()
        if normalized:  # Only include claims with non-empty normalized text
            valid_claims.append(claim)
    
    if not valid_claims:
        st.info("No verifiable claims extracted from the text.")
        return
    
    for i, claim in enumerate(valid_claims, 1):
        # Convert to dict if it's a Pydantic model
        c = claim.model_dump() if hasattr(claim, 'model_dump') else claim
        
        normalized = c.get('normalized_claim', 'N/A')
        with st.expander(f"Claim {i}: {normalized}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                original = c.get('span_text', '')
                
                st.markdown(f"**Normalized Claim:** {normalized}")
                
                # Only show original if it's different from normalized
                if original and original.strip() != normalized.strip():
                    st.markdown(f"**Original Text:** *\"{original}\"*")
                
                st.markdown(f"**Language:** {c.get('lang', 'N/A')}")
                
                # Only show English translation if it's different from normalized
                english = c.get('english', '')
                if english and english.strip() != normalized.strip():
                    st.markdown(f"**English Translation:** {english}")
            
            with col2:
                st.markdown("**Extraction Confidence**")
                render_confidence_bar(c.get('confidence', 0))
            
            # Token details in collapsed section
            with st.expander("Token Details", expanded=False):
                st.json({
                    "tokens": c.get('tokens', []),
                    "token_mask": c.get('token_mask', []),
                    "tokenization": c.get('tokenization', 'whitespace')
                })

def render_evidence(evidence: Evidence, claim_text: str) -> None:
    """Render evidence for a claim"""
    # Convert to dict if it's a Pydantic model
    ev = evidence.model_dump() if hasattr(evidence, 'model_dump') else evidence
    
    with st.expander(f"🔎 Evidence for: \"{claim_text}\"", expanded=True):
        evidence_list = ev.get('evidence', [])
        
        if not evidence_list:
            st.warning("No evidence found for this claim.")
            return
        
        # Group by source type
        source_types = {}
        for item in evidence_list:
            source_type = item.get('source_type', 'unknown')
            source_types.setdefault(source_type, []).append(item)
        
        # Display source type summary
        if source_types:
            st.markdown(f"**Total Sources:** {len(evidence_list)}")
            cols = st.columns(len(source_types))
            for col, (source_type, items) in zip(cols, source_types.items()):
                with col:
                    st.metric(label=source_type.replace('_', ' ').title(), value=len(items))
        
        st.markdown("---")
        
        # For Gemini Search results, show summary snippet once, then list all URLs
        gemini_items = [item for item in evidence_list if item.get('source_type') == 'gemini']
        other_items = [item for item in evidence_list if item.get('source_type') != 'gemini']
        
        # Show Gemini grounded summary if present
        if gemini_items:
            st.markdown("### 🤖 Gemini Search Summary")
            
            # Get the grounded snippet (they may share the same snippet text)
            snippets = list(set([item.get('snippet', '') for item in gemini_items if item.get('snippet')]))
            
            for snippet in snippets:
                if snippet:
                    st.info(snippet)
            
            # List all source URLs that contributed to this summary
            st.markdown("**📚 Sources Contributing to Summary:**")
            for idx, item in enumerate(gemini_items, 1):
                url = item.get('url', '')
                domain = item.get('domain', 'Unknown')
                title = item.get('title', '')
                
                if url:
                    # Format as clickable link with title/domain
                    display_text = title if title and title != "No title" else domain
                    st.markdown(f"{idx}. [{display_text}]({url})")
                else:
                    st.markdown(f"{idx}. {domain}")
        
        # Show other evidence types separately with their own snippets
        if other_items:
            if gemini_items:
                st.markdown("---")
            
            st.markdown("### 📄 Additional Evidence Sources")
            
            for idx, item in enumerate(other_items, 1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    domain = item.get('domain', 'Unknown')
                    url = item.get('url', '#')
                    source_type = item.get('source_type', 'unknown').replace('_', ' ').title()
                    
                    st.markdown(f"**{idx}. Source: [{domain}]({url})**")
                    st.caption(f"Type: {source_type}")
                    
                    # Show individual snippet for non-Gemini sources
                    snippet = item.get('snippet', '')
                    if snippet:
                        st.info(snippet)
                    
                    # Title if available and different from domain
                    title = item.get('title')
                    if title and title != domain:
                        st.caption(f"Title: {title}")
                
                with col2:
                    # Relevance score if available
                    score = item.get('score')
                    if score is not None:
                        st.markdown("**Relevance**")
                        render_confidence_bar(score)
                
                if idx < len(other_items):
                    st.markdown("---")

def render_verification(verifier_result: VerifierResult, claim_text: str) -> None:
    """Render verification results"""
    # Convert to dict if it's a Pydantic model
    vr = verifier_result.model_dump() if hasattr(verifier_result, 'model_dump') else verifier_result
    
    with st.expander(f"✅ Verification: \"{claim_text}\"", expanded=True):
        # Overall verdict
        st.markdown("### Overall Verdict")
        overall_verdict = vr.get('overall_verdict', {})
        if overall_verdict:
            verdict_html = render_verdict_badge(overall_verdict.get("label", "NEUTRAL"))
            st.markdown(verdict_html, unsafe_allow_html=True)
        
        # Confidence metrics
        st.markdown("### Confidence Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_confidence_bar(
                overall_verdict.get("confidence", 0) if overall_verdict else 0,
                "Overall Confidence"
            )
        
        with col2:
            render_confidence_bar(
                vr.get('support_confidence', 0),
                "Support Confidence"
            )
        
        with col3:
            render_confidence_bar(
                vr.get('refute_confidence', 0),
                "Refute Confidence"
            )
        
        # Claim score
        claim_score = vr.get('claim_score')
        if claim_score is not None:
            st.markdown("### Claim Score")
            st.progress(claim_score)
            st.caption(f"{claim_score:.2%}")
        
        # Reasoning
        reasoning = vr.get('reasoning')
        if reasoning:
            st.markdown("### Reasoning")
            st.write(reasoning)

def render_final_decision(decision: FinalDecision, manipulation_score: float, retrieval_coverage: float, claims_count: int) -> None:
    """Render final decision"""
    # Convert to dict if it's a Pydantic model
    d = decision.model_dump() if hasattr(decision, 'model_dump') else decision
    
    st.subheader("🎯 Final Decision")
    
    # Main verdict card
    label = d.get('label', 'N/A')
    verdict_html = render_verdict_badge(label)
    st.markdown(f"## {verdict_html}", unsafe_allow_html=True)
    
    # Show what this decision means
    if label == "send_downstream":
        st.info("⚠️ **Action Required:** This post should be sent for further analysis (propaganda detection, context analysis, or human review)")
    elif label == "high_conf_fake":
        st.error("❌ **High Confidence:** This post contains FALSE information based on authoritative evidence")
    elif label == "high_conf_true":
        st.success("✅ **High Confidence:** This post contains TRUE information based on authoritative evidence")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Manipulation Score",
            value=f"{manipulation_score:.2%}",
            delta="High Risk" if manipulation_score > 0.6 else "Low Risk",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Retrieval Coverage",
            value=f"{retrieval_coverage:.2%}",
            delta="Good" if retrieval_coverage >= 0.7 else "Poor",
            delta_color="normal" if retrieval_coverage >= 0.7 else "inverse"
        )
    
    with col3:
        st.metric(
            label="Claims Processed",
            value=claims_count
        )
    
    # Decision reasoning
    why = d.get('why')
    if why:
        st.markdown("### Decision Reasoning")
        st.info(why)
    
    # Post verification score if available
    post_score = d.get('post_verification_score')
    if post_score is not None:
        st.markdown(f"**Post Verification Score:** {post_score:.2f}")
    
    # Refuting URLs if available
    refuting_urls = d.get('top_refuting_urls', [])
    if refuting_urls:
        st.markdown("### Top Refuting Sources")
        for i, url in enumerate(refuting_urls[:3], 1):
            st.markdown(f"{i}. {url}")

def process_text(text: str) -> Dict[str, Any]:
    """Process text through the pipeline - language detection is automatic"""
    state = {
        "post_id": "streamlit_1",
        "post_text": text
        # lang_hint is optional - pipeline will auto-detect language
    }
    
    result = st.session_state.pipeline_app.invoke(state)
    return result

# Main UI
def main():
    st.title("🔍 Evidence Retrieval & Verification System")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API Key configured")
        else:
            st.error("❌ Gemini API Key not found")
            st.info("Set GEMINI_API_KEY environment variable")
        
        st.markdown("---")
        
        # About
        st.header("📖 About")
        st.markdown("""
        This system uses a multi-agent pipeline to:
        
        1. **Detect Claims** - Filter verifiable claims
        2. **Extract Claims** - Identify claim spans (auto-detect language)
        3. **Retrieve Evidence** - Search credible sources
        4. **Verify Claims** - Analyze evidence
        5. **Final Decision** - Deterministic verdict
        
        **Deterministic Features:**
        - Temperature = 0
        - Automatic language detection
        - Confidence-based scoring
        - Explicit validation rules
        - Prioritizes authoritative sources
        """)
        
        st.markdown("---")
        
        # Examples
        st.header("💡 Examples")
        
        if st.button("Example 1: English Claim", use_container_width=True):
            st.session_state.text_input_area = "I read that mRNA vaccines cause cancer!"
            if 'results' in st.session_state:
                del st.session_state.results
        
        if st.button("Example 2: Opinion (filtered)", use_container_width=True):
            st.session_state.text_input_area = "I will never take a COVID vaccine..."
            if 'results' in st.session_state:
                del st.session_state.results
        
        if st.button("Example 3: Hindi Claim", use_container_width=True):
            st.session_state.text_input_area = "मुझे बताया गया कि mRNA वैक्सीन से कैंसर होता है।"
            if 'results' in st.session_state:
                del st.session_state.results
        
        if st.button("Example 4: Climate Claim", use_container_width=True):
            st.session_state.text_input_area = "Climate change is a hoax created by scientists."
            if 'results' in st.session_state:
                del st.session_state.results
    
    # Main content area
    st.markdown("---")
    
    # Input section
    st.header("📝 Input Text")
    
    text_input = st.text_area(
        "Enter text to verify:",
        height=150,
        placeholder="Enter a claim or statement to fact-check...",
        help="Paste any text containing claims you want to verify. Language will be auto-detected.",
        key="text_input_area"
    )
    
    process_button = st.button(
        "🚀 Verify Claim",
        type="primary",
        use_container_width=True,
        disabled=not text_input.strip()
    )
    
    # Process button logic
    if process_button and text_input.strip():
        st.session_state.processing = True
        
        with st.spinner("🔄 Processing through pipeline..."):
            try:
                results = process_text(text_input.strip())
                st.session_state.results = results
                st.session_state.processing = False
                st.success("✅ Processing complete!")
            except Exception as e:
                st.error(f"❌ Error processing text: {str(e)}")
                st.exception(e)
                st.session_state.processing = False
                return
    
    # Results section
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if st.session_state.results:
        st.markdown("---")
        st.header("📊 Results")
        
        results = st.session_state.results
        
        # Claim detection
        claim_detection = results.get("claim_detection", {})
        claims = results.get("claims", [])
        
        # Filter out empty claims
        valid_claims = []
        for claim in claims:
            c = claim.model_dump() if hasattr(claim, 'model_dump') else claim
            normalized = (c.get('normalized_claim') or '').strip()
            if normalized:
                valid_claims.append(claim)
        
        if claim_detection:
            # Show the actual result: Yes if valid claims were extracted, No otherwise
            has_claims_extracted = len(valid_claims) > 0
            render_claim_detection(
                has_claims_extracted,
                claim_detection.get("reasoning", "")
            )
        
        # If no valid claims were extracted, show message and stop
        if not valid_claims:
            st.info("💭 This text appears to be an opinion or personal statement, not a verifiable claim.")
            return
        
        # Claims (pass valid claims only)
        render_claims(valid_claims)
        
        # Evidence and verification for each claim
        # Build enriched structure from separate state fields
        claims = results.get("claims", [])
        claim_evidence_list = results.get("claim_evidence", [])
        verifier_results = results.get("verifier_results", [])
        
        if claims and (claim_evidence_list or verifier_results):
            st.markdown("---")
            st.subheader("🔬 Evidence & Verification")
            
            # Create lookup dicts
            evidence_by_id = {ev.claim_id: ev for ev in claim_evidence_list}
            verifier_by_id = {vr.claim_id: vr for vr in verifier_results}
            
            for claim in claims:
                evidence = evidence_by_id.get(claim.claim_id)
                verifier_result = verifier_by_id.get(claim.claim_id)
                
                if evidence:
                    render_evidence(evidence, claim.normalized_claim)
                
                if verifier_result:
                    render_verification(verifier_result, claim.normalized_claim)
        
        # Final decision
        if results.get("final_decision"):
            st.markdown("---")
            # Extract manipulation_score from ManipulationScore object if present
            manipulation_obj = results.get("manipulation")
            manipulation_score = manipulation_obj.manipulation_score if manipulation_obj else 0.0
            
            render_final_decision(
                results["final_decision"],
                manipulation_score,
                results.get("retrieval_coverage", 0.0),
                len(results.get("claims", []))
            )
        
        # Raw JSON (collapsed)
        with st.expander("🔧 Raw JSON Output", expanded=False):
            # Convert objects to dicts for JSON display
            claim_detection = results.get("claim_detection", {})
            manipulation = results.get("manipulation")
            
            display_results = {
                "post_id": results.get("post_id"),
                "post_text": results.get("post_text"),
                "claim_detection": claim_detection,
                "claims_count": len(results.get("claims", [])),
                "evidence_count": len(results.get("claim_evidence", [])),
                "verifier_results_count": len(results.get("verifier_results", [])),
                "manipulation_score": manipulation.manipulation_score if manipulation else 0.0,
                "retrieval_coverage": results.get("retrieval_coverage", 0.0),
                "final_decision_label": results.get("final_decision").label if results.get("final_decision") else None
            }
            st.json(display_results)
        
        # Export button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Clear Results"):
                st.session_state.results = None
                st.rerun()
        
        with col2:
            # Export results as JSON
            manipulation = results.get("manipulation")
            export_data = {
                "post_text": results.get("post_text"),
                "timestamp": results.get("post_id"),
                "final_decision": {
                    "label": results.get("final_decision").label if results.get("final_decision") else None,
                    "manipulation_score": manipulation.manipulation_score if manipulation else 0.0,
                    "retrieval_coverage": results.get("retrieval_coverage", 0.0)
                },
                "claims_count": len(results.get("claims", []))
            }
            
            st.download_button(
                label="📥 Export JSON",
                data=json.dumps(export_data, indent=2),
                file_name="verification_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
