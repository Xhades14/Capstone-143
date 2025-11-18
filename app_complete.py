"""
Complete Pipeline Frontend - Integrates Verification + Propaganda Detection + GAT
"""
# CRITICAL: Set environment variables BEFORE any imports
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import streamlit as st
import sys
import json
import random
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from peft import PeftModel
from deep_translator import GoogleTranslator

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.graph import build_graph
from src.schemas import Claim, ClaimEvidence, FinalDecision
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import Tensor
import time
try:
    from googletrans import Translator as GoogleTranslator
except ImportError:
    GoogleTranslator = None

# Emotion label mappings must match build_meta_dataset.py to keep features aligned
GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EKMAN_TO_FINE = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    ],
    "sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}
EKMAN_TO_INDICES = {
    cluster: [GOEMOTIONS_LABELS.index(label) for label in fine]
    for cluster, fine in EKMAN_TO_FINE.items()
}

HINDI_SYNTHETIC_SAMPLES = [
    {
        "tweet_id": "synthetic_hi_prop_1",
        "root_text": "इन खबरों को फैलाओ: नए कृषि कानून किसानों की जमीन सीधे कॉरपोरेट को सौंप देंगे।",
        "replies": [
            "ये बिल किसानों को गुलाम बना देंगे, लोग जागो!",
            "मेरे गाँव में सब कह रहे हैं कि ज़मीन छीनी जाएगी।",
            "सरकार सिर्फ़ उद्योगपतियों के लिए काम कर रही है।",
            "कोई इसे सुप्रीम कोर्ट तक लेकर क्यों नहीं जाता?",
            "मैंने सुना है कि फसल खरीद ही बंद हो जाएगी।",
            "सबको इस खबर को हर ग्रुप में भेजना चाहिए।",
        ],
        "true_label": 1,
        "language": "hi",
        "scenario": "agrarian_fear_propaganda",
        "source": "synthetic_hi",
    },
    {
        "tweet_id": "synthetic_hi_prop_2",
        "root_text": "महामारी के इलाज के नाम पर सरकार ज़हरीला टीका लोगों को मजबूरी में लगवा रही है।",
        "replies": [
            "टीका लगते ही पड़ोस की आंटी तीन दिन तक बेहोश रहीं।",
            "यह सब आबादी कम करने की साजिश है।",
            "डॉक्टर भी सच बोलने से डर रहे हैं।",
            "मेरी चचेरी बहन ने मना किया है अपने बच्चों को टीका लगाने से।",
            "अब तो स्कूल भी दबाव बना रहे हैं, यह गलत है।",
            "वीडियो शेयर करो ताकि लोग जागें।",
        ],
        "true_label": 1,
        "language": "hi",
        "scenario": "vaccine_mistrust_campaign",
        "source": "synthetic_hi",
    },
    {
        "tweet_id": "synthetic_hi_nonprop_1",
        "root_text": "जिले में मुफ्त टीकाकरण शिविर 15 जुलाई से खुलेगा, सभी परिवार आएँ।",
        "replies": [
            "बहुत अच्छा, दादी को लेकर जाऊँगा।",
            "क्या रविवार को भी खुलेगा?",
            "डॉक्टरों की सूची कहाँ मिलेगी?",
            "ग्रामीणों के लिए बस की व्यवस्था भी कर दो।",
            "महिला स्वास्थ्य कार्यकर्ता गाँव-गाँव सूचना दे रही हैं।",
        ],
        "true_label": 0,
        "language": "hi",
        "scenario": "public_health_info",
        "source": "synthetic_hi",
    },
    {
        "tweet_id": "synthetic_hi_nonprop_2",
        "root_text": "राज्य सरकार ने इस साल की छात्रवृत्ति तिथियाँ जारी कीं, आवेदन पोर्टल खुल गया है।",
        "replies": [
            "लिंक भेजो ताकि कॉलेज समूह में साझा कर सकूँ।",
            "क्या पिछली बार की तरह दस्तावेज़ स्कैन करने होंगे?",
            "बहुत विद्यार्थियों को राहत मिलेगी।",
            "मेरे भाई ने पिछले साल इसी योजना से फीस भरी थी।",
            "आवेदन की अंतिम तिथि क्या है?",
            "काउंसलर ने भी यही सूचना व्हाट्सऐप पर भेजी है।",
        ],
        "true_label": 0,
        "language": "hi",
        "scenario": "scholarship_announcement",
        "source": "synthetic_hi",
    },
    {
        "tweet_id": "synthetic_hi_factcheck_1",
        "root_text": "मैंने पढ़ा कि mRNA वैक्सीन से कैंसर पक्का होता है, सरकार यह सच्चाई छिपा रही है।",
        "replies": [
            "मेरे अंकल ने भी यही मैसेज भेजा, डर लग रहा है।",
            "कहते हैं एक वैज्ञानिक ने वीडियो में साबित किया है।",
            "अगर यह सच है तो अस्पताल क्यों नहीं बोलते?",
            "कोई भरोसेमंद लिंक साझा करो।",
            "लोगों को चेतावनी देनी होगी।",
            "टीका लगवाने से पहले दो बार सोचो।",
        ],
        "true_label": 1,
        "language": "hi",
        "scenario": "fact_check_halt",
        "source": "synthetic_hi",
        "fact_check_only": True,
    },
]


def collapse_goemotions_to_ekman(vector: np.ndarray) -> np.ndarray:
    collapsed = np.zeros(len(EKMAN_LABELS), dtype=np.float32)
    for idx, label in enumerate(EKMAN_LABELS):
        indices = EKMAN_TO_INDICES[label]
        if not indices:
            continue
        collapsed[idx] = float(np.max(vector[indices]))

    total = collapsed.sum()
    if total > 0:
        collapsed /= total
    return collapsed

# ============= MODEL LOADING =============

@st.cache_resource
def load_verification_pipeline():
    """Load LLM-based verification pipeline"""
    return build_graph()

@st.cache_resource
def load_propaganda_model():
    """Load English propaganda detection model"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = "eng_prop_model/SemEval_Trained_Intermediate(final)"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_hindi_propaganda_model():
    """Load Hindi LoRA-adapted propaganda model"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    base_model_path = "eng_prop_model/SemEval_Trained_Intermediate(final)"
    adapter_path = "hprop-lora-adapter"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_emotion_model():
    """Load English emotion detection model"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = "eng_emo_model/models"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_meta_classifier():
    """Load meta-classifier"""
    return joblib.load("models/meta_classifier.joblib")

@st.cache_resource
def load_sentence_embedder():
    """Load sentence transformer for GAT input"""
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

@st.cache_resource
def load_hi_en_translator():
    """Load lightweight Hindi to English translator (googletrans fallback only to avoid memory issues)."""
    if GoogleTranslator is None:
        st.warning("googletrans not available. Install with: pip install googletrans==3.1.0a0")
        return None
    try:
        return GoogleTranslator()
    except Exception as exc:
        st.warning(f"Could not initialize translator: {exc}")
        return None

@st.cache_resource
def load_gat_model():
    """Load trained GAT model and graph"""
    # Define GAT architecture (same as training)
    class GATClassifier(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_dim: int, heads: int, dropout: float):
            super().__init__()
            self.dropout = dropout
            self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
            self.conv3 = GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        
        def forward(self, x, edge_index, return_attention=False):
            h1 = self.conv1(x, edge_index)
            h1 = F.elu(h1)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            
            h2 = self.conv2(h1, edge_index)
            h2 = F.elu(h2)
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            
            if return_attention:
                logits, (att_edge_index, att_weights) = self.conv3(
                    h2, edge_index, return_attention_weights=True
                )
                return logits.view(-1), h1.detach(), h2.detach(), (att_edge_index, att_weights)
            
            logits = self.conv3(h2, edge_index)
            return logits.view(-1), h1.detach(), h2.detach(), None
    
    # Load model
    model = GATClassifier(in_channels=769, hidden_dim=128, heads=8, dropout=0.2)
    model.load_state_dict(torch.load("artifacts/gat_propaganda/gat_model.pt", map_location='cpu'))
    model.eval()
    
    # Load graph data
    with open("artifacts/gat_propaganda/learned_graph.gpickle", "rb") as f:
        graph = pickle.load(f)
    
    # Load node embeddings and features (from training)
    embeddings_data = torch.load("artifacts/gat_propaganda/node_embeddings.pt", map_location='cpu')
    
    # Load dataset for tweet mapping
    df = pd.read_json("prop_datasets/tree_width/merged_conversations.jsonl", lines=True)
    if "post_id" in df.columns and "tweet_id" not in df.columns:
        df = df.rename(columns={"post_id": "tweet_id"})
    if "label" in df.columns and "true_label" not in df.columns:
        df = df.rename(columns={"label": "true_label"})
    df = df.sort_values("tweet_id").reset_index(drop=True)
    
    return model, graph, embeddings_data, df

@st.cache_data
def load_sample_conversations():
    """Load merged conversations dataset"""
    df = pd.read_json("prop_datasets/tree_width/merged_conversations.jsonl", lines=True)
    if "post_id" in df.columns and "tweet_id" not in df.columns:
        df = df.rename(columns={"post_id": "tweet_id"})
    if "label" in df.columns and "true_label" not in df.columns:
        df = df.rename(columns={"label": "true_label"})
    df = df.sort_values("tweet_id").reset_index(drop=True)
    if "language" not in df.columns:
        df["language"] = "en"
    if "scenario" not in df.columns:
        df["scenario"] = "dataset_record"
    if "source" not in df.columns:
        df["source"] = "dataset"

    synthetic_df = pd.DataFrame(HINDI_SYNTHETIC_SAMPLES)
    combined = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
    return combined.reset_index(drop=True)


def format_percent(value: Any, digits: int = 2, default: str = "N/A") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}%}"
    return default


def call_with_retry(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            wait = backoff ** attempt
            st.warning(f"Verifier call failed (attempt {attempt}/{retries}): {exc}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    st.error("Verifier failed after retries. Showing partial results.")
    raise last_error

# ============= INFERENCE FUNCTIONS =============

def predict_propaganda(tokenizer, model, text: str) -> float:
    """Run propaganda detection on single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()  # Probability of propaganda class


def translate_hindi_to_english(
    texts: List[str],
    translator,
) -> List[str]:
    """Translate Hindi texts using lightweight googletrans API."""
    if not texts or not translator:
        return texts

    outputs: List[str] = []
    for text in texts:
        try:
            result = translator.translate(text, src='hi', dest='en')
            outputs.append(result.text if result and result.text else text)
        except Exception:  # pragma: no cover - API fallback best effort
            outputs.append(text)
    return outputs

def predict_emotions(tokenizer, model, texts: List[str]) -> List[np.ndarray]:
    """Run emotion detection and collapse GoEmotions to Ekman clusters per reply"""
    collapsed_vectors: List[np.ndarray] = []
    if not texts:
        return collapsed_vectors

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        collapsed = collapse_goemotions_to_ekman(probs)
        collapsed_vectors.append(collapsed)

    return collapsed_vectors


def compute_emotion_features(collapsed_vectors: List[np.ndarray]) -> Dict[str, float]:
    """Aggregate collapsed emotion vectors using the same logic as build_meta_dataset.py"""
    if collapsed_vectors:
        stack = np.vstack(collapsed_vectors)
        mean_vec = stack.mean(axis=0)
        variance = float(stack.var(axis=0).mean())
    else:
        mean_vec = np.zeros(len(EKMAN_LABELS), dtype=np.float32)
        variance = 0.0

    total = float(mean_vec.sum())
    if total > 0:
        normalized = mean_vec / total
        entropy = float(-np.sum(normalized * np.log(normalized + 1e-12)))
    else:
        entropy = 0.0

    features = {f"mean_{label}": float(mean_vec[idx]) for idx, label in enumerate(EKMAN_LABELS)}
    features["entropy"] = entropy
    features["variance"] = variance
    return features


def find_knn_neighbors(new_embedding: np.ndarray, existing_embeddings: torch.Tensor, k: int = 5) -> List[int]:
    """Find K nearest neighbors using cosine similarity between embeddings."""
    # Convert new embedding to tensor and normalize
    new_emb_tensor = torch.from_numpy(new_embedding).float()
    new_emb_norm = torch.nn.functional.normalize(new_emb_tensor.unsqueeze(0), dim=1)
    
    # Normalize existing embeddings (use only the embedding part, skip prop_score)
    existing_emb_only = existing_embeddings[:, :-1]  # Remove last column (prop_score)
    existing_emb_norm = torch.nn.functional.normalize(existing_emb_only, dim=1)
    
    # Compute cosine similarity
    similarities = torch.mm(new_emb_norm, existing_emb_norm.t()).squeeze(0)
    
    # Get top K indices
    top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
    
    return top_k_indices.tolist()

def run_gat_inference(model, graph, embeddings_data, df, node_idx: int, 
                      prop_score: float, embedding: np.ndarray, threshold: float = 0.55,
                      is_new_node: bool = False, k_neighbors: int = 5) -> Dict[str, Any]:
    """Run GAT inference on existing graph node or dynamically add new node with KNN edges."""
    # Reconstruct edge_index from graph
    edges = list(graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    # Get stored features
    x = embeddings_data["input"]  # All node features from training
    num_training_nodes = x.size(0)
    
    # Handle new nodes by adding them dynamically
    if is_new_node or node_idx >= num_training_nodes:
        # Create new node feature vector
        new_node_features = torch.cat([
            torch.from_numpy(embedding).float(),
            torch.tensor([prop_score], dtype=torch.float32)
        ]).unsqueeze(0)
        
        # Find K nearest neighbors from existing nodes
        knn_indices = find_knn_neighbors(embedding, x, k=k_neighbors)
        
        # Create edges: new_node <-> each KNN neighbor (bidirectional)
        new_node_idx = num_training_nodes  # Assign next available index
        new_edges_out = [[new_node_idx, neighbor] for neighbor in knn_indices]
        new_edges_in = [[neighbor, new_node_idx] for neighbor in knn_indices]
        all_new_edges = new_edges_out + new_edges_in
        
        new_edge_tensor = torch.tensor(all_new_edges, dtype=torch.long).t()
        
        # Append new node features and edges
        x = torch.cat([x, new_node_features], dim=0)
        edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
        
        # Use the new node index for inference
        inference_idx = new_node_idx
        ground_truth = "N/A (synthetic)"
    else:
        inference_idx = node_idx
        ground_truth = df.iloc[node_idx].get("true_label", "N/A")
    
    # Run model
    with torch.no_grad():
        logits, _, _, attention = model(x, edge_index, return_attention=True)
        prob = torch.sigmoid(logits[inference_idx]).item()
        pred = int(prob >= threshold)
    
    # Get neighbors and their attention weights
    att_edge_index, att_weights = attention
    att_weights = att_weights.mean(dim=1)  # Average over heads
    
    # Find edges involving this node
    neighbor_info = []
    for i in range(att_edge_index.size(1)):
        src, dst = att_edge_index[0, i].item(), att_edge_index[1, i].item()
        if src == inference_idx:
            weight = att_weights[i].item()
            # Only show neighbors from training set (not the synthetic node itself)
            if dst < num_training_nodes:
                neighbor_text = df.iloc[dst]["root_text"][:80] if dst < len(df) else "N/A"
                neighbor_label = df.iloc[dst].get("true_label", "N/A") if dst < len(df) else "N/A"
                neighbor_info.append({
                    "neighbor_idx": dst,
                    "attention_weight": weight,
                    "text_preview": neighbor_text,
                    "true_label": neighbor_label
                })
    
    # Sort by attention weight
    neighbor_info.sort(key=lambda x: x["attention_weight"], reverse=True)
    
    return {
        "node_idx": node_idx,
        "probability": prob,
        "prediction": pred,
        "label": "propaganda" if pred == 1 else "non-propaganda",
        "top_neighbors": neighbor_info[:10],  # Top 10 most attended neighbors
        "ground_truth": ground_truth,
        "is_synthetic": is_new_node or node_idx >= num_training_nodes
    }


def build_neighbor_graph_figure(node_idx: int, neighbors: List[Dict[str, Any]], label: str,
                                max_neighbors: int = 8):
    """Create a matplotlib figure highlighting the node and its most attended neighbors."""
    sub_neighbors = neighbors[:max_neighbors]
    graph = nx.Graph()
    graph.add_node(node_idx, role="target", label=label)

    for neighbor in sub_neighbors:
        neighbor_idx = neighbor.get("neighbor_idx")
        attention_weight = float(neighbor.get("attention_weight", 0.0))
        neighbor_label = "propaganda" if neighbor.get("true_label") == 1 else "non-propaganda"
        graph.add_node(neighbor_idx, role="neighbor", label=neighbor_label, weight=attention_weight)
        graph.add_edge(node_idx, neighbor_idx, weight=max(attention_weight, 0.0))

    if graph.number_of_nodes() == 1:
        positions = {node_idx: (0.0, 0.0)}
    else:
        positions = nx.spring_layout(graph, seed=42, weight="weight")

    fig, ax = plt.subplots(figsize=(6, 6))
    node_colors = []
    node_sizes = []
    edge_widths = []

    for node in graph.nodes:
        node_data = graph.nodes[node]
        if node == node_idx:
            node_colors.append("#d62728" if label == "propaganda" else "#2ca02c")
            node_sizes.append(1600)
        else:
            neighbor_label = node_data.get("label", "non-propaganda")
            node_colors.append("#ff9896" if neighbor_label == "propaganda" else "#98df8a")
            node_sizes.append(900)

    for _, _, data in graph.edges(data=True):
        edge_widths.append(max(1.2, data.get("weight", 0.0) * 8))

    nx.draw_networkx_edges(graph, positions, width=edge_widths, alpha=0.6, edge_color="#7f7f7f", ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.5,
        edgecolors="#333333",
        ax=ax,
    )

    labels = {node_idx: f"Node {node_idx}"}
    for node in graph.nodes:
        if node != node_idx:
            labels[node] = str(node)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=9, ax=ax)

    ax.set_title("Attention Graph: Node and Neighbors")
    ax.axis("off")
    fig.tight_layout()
    return fig

# ============= STREAMLIT UI =============

st.set_page_config(
    page_title="Multi-Stage Propaganda Detection Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Multi-Stage Propaganda & Fact Verification System")
st.markdown("""
This pipeline combines **LLM-based fact verification** with **propaganda detection** and **Graph Neural Networks** 
to provide comprehensive misinformation analysis.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    verification_enabled = st.checkbox("Enable Verification Stage", value=True, 
                                       help="Stage 1: LLM-based claim verification")
    propaganda_enabled = st.checkbox("Enable Propaganda Detection", value=True,
                                     help="Stage 2: Propaganda + Emotion analysis")
    gat_enabled = st.checkbox("Enable GAT Inference", value=True,
                             help="Stage 3: Graph-based final decision")
    
    st.divider()
    st.header("🎲 Sample Selection")
    
    sample_mode = st.radio("Input Mode", ["Random Sample", "Select by Index", "Hindi Synthetic Examples"])
    
    st.divider()
    st.info("💡 **Tip:** Models load on-demand when you run the pipeline")

# Load dataset (lightweight)
df = load_sample_conversations()
synthetic_examples = df[df.get("source") == "synthetic_hi"]
synthetic_count = len(synthetic_examples)
if synthetic_count:
    st.success(f"✅ Dataset loaded: {len(df)} conversations including {synthetic_count} Hindi demos")
else:
    st.success(f"✅ Dataset loaded: {len(df)} conversations available")

# Sample selection
if sample_mode == "Random Sample":
    if st.button("🎲 Get Random Sample", type="primary"):
        st.session_state.selected_idx = random.randint(0, len(df) - 1)
elif sample_mode == "Select by Index":
    selected_idx = st.number_input("Select Tweet Index", min_value=0, max_value=len(df)-1, value=0)
    if st.button("Load Tweet"):
        st.session_state.selected_idx = selected_idx
else:
    if synthetic_count == 0:
        st.warning("कोई हिंदी डेमो उपलब्ध नहीं")
    else:
        option_map = {}
        option_labels = []
        for idx, row in synthetic_examples.iterrows():
            label = "Propaganda" if row.get("true_label", 0) == 1 else "Non-Propaganda"
            scenario = row.get("scenario", "demo")
            option_label = f"{row['tweet_id']} · {label} · {scenario}"
            option_map[option_label] = idx
            option_labels.append(option_label)
        selected_demo = st.selectbox("Select Hindi Example", option_labels)
        if st.button("Load Hindi Example", key="load_hindi_example"):
            st.session_state.selected_idx = option_map[selected_demo]

# Display selected conversation
if "selected_idx" in st.session_state:
    idx = st.session_state.selected_idx
    row = df.iloc[idx]
    row_language = str(row.get("language", "en")).lower()
    is_hindi_sample = row_language.startswith("hi")
    language_display = "Hindi" if is_hindi_sample else "English"
    row_replies = row.get("replies", [])
    if not isinstance(row_replies, list):
        row_replies = []
    
    st.markdown("---")
    st.subheader(f"📝 Selected Conversation (Index: {idx})")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tweet ID", row["tweet_id"])
    with col2:
        st.metric("Ground Truth", "Propaganda" if row.get("true_label", 0) == 1 else "Non-Propaganda")
    with col3:
        st.metric("Replies", len(row_replies))
    with col4:
        st.metric("Language", language_display)
    if row.get("scenario") and row.get("scenario") not in {"dataset_record", "demo"}:
        st.caption(f"Scenario: {row.get('scenario')}")
    
    st.markdown("**Root Tweet:**")
    st.info(row["root_text"])
    
    with st.expander("📨 View Replies"):
        if row_replies:
            st.markdown(f"**Total Replies: {len(row_replies)}**")
            # Show first 20 in UI, but use all for processing
            for i, reply in enumerate(row_replies[:20], 1):
                st.markdown(f"**Reply {i}:** {reply}")
            if len(row_replies) > 20:
                st.caption(f"... and {len(row_replies) - 20} more replies (all used in analysis)")
        else:
            st.warning("No replies available")
    
    # Run pipeline
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        results = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ============= LAZY MODEL LOADING =============
        status_text.text("Loading required models...")
        progress_bar.progress(0.05)
        
        if verification_enabled:
            with st.spinner("Loading verification pipeline..."):
                try:
                    verification_pipeline = load_verification_pipeline()
                except Exception as e:
                    st.error(f"❌ Failed to load verification pipeline: {e}")
                    st.stop()
        
        prop_model_bundle = None
        hindi_prop_bundle = None
        translator_bundle = None
        if propaganda_enabled:
            with st.spinner("Loading propaganda & emotion models..."):
                try:
                    emo_tokenizer, emo_model = load_emotion_model()
                    meta_classifier = load_meta_classifier()
                    embedder = load_sentence_embedder()
                    if is_hindi_sample:
                        hindi_prop_bundle = load_hindi_propaganda_model()
                        translator_bundle = load_hi_en_translator()
                    else:
                        prop_model_bundle = load_propaganda_model()
                except Exception as e:
                    st.error(f"❌ Failed to load propaganda models: {e}")
                    st.exception(e)
                    st.stop()
        
        if gat_enabled:
            with st.spinner("Loading GAT model..."):
                try:
                    gat_model, gat_graph, gat_embeddings, gat_df = load_gat_model()
                except Exception as e:
                    st.error(f"❌ Failed to load GAT model: {e}")
                    st.exception(e)
                    st.stop()
        
        progress_bar.progress(0.1)
        status_text.text("Models loaded successfully!")
        
        # ============= STAGE 1: VERIFICATION =============
        if verification_enabled:
            status_text.text("Stage 1: Running claim verification...")
            progress_bar.progress(0.2)
            with st.spinner("Stage 1: Running claim verification..."):
                verification_state = {
                    "post_id": f"tweet_{idx}",
                    "post_text": row["root_text"],
                    "lang_hint": "en"
                }
                verification_result = call_with_retry(verification_pipeline.invoke, verification_state)
                results["verification"] = verification_result
            
            st.markdown("---")
            st.subheader("📋 Stage 1: Claim Verification")
            
            # Show claims extracted
            claims = verification_result.get("claims", [])
            valid_claims = [c for c in claims if (c.model_dump() if hasattr(c, 'model_dump') else c).get('normalized_claim', '').strip()]
            
            if valid_claims:
                with st.expander(f"📌 Extracted Claims ({len(valid_claims)})", expanded=True):
                    for i, claim in enumerate(valid_claims, 1):
                        c = claim.model_dump() if hasattr(claim, 'model_dump') else claim
                        st.markdown(f"**Claim {i}:** {c.get('normalized_claim', 'N/A')}")
                        st.caption(f"Confidence: {c.get('confidence', 0):.2%} | Language: {c.get('lang', 'en')}")
            
            # Show evidence summary
            evidence_list = verification_result.get("claim_evidence", [])
            if evidence_list:
                total_evidence = sum(len(ev.evidence if hasattr(ev, 'evidence') else ev.get('evidence', [])) for ev in evidence_list)
                st.info(f"🔍 Retrieved {total_evidence} evidence items from authoritative sources")
            
            # Show verification results
            verifier_results = verification_result.get("verifier_results", [])
            if verifier_results:
                with st.expander("✅ Claim Verification Results", expanded=False):
                    for vr in verifier_results:
                        vr_data = vr.model_dump() if hasattr(vr, 'model_dump') else vr
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Claim Score", format_percent(vr_data.get('claim_score')))
                        with col2:
                            st.metric("Support", format_percent(vr_data.get('support_confidence')))
                        with col3:
                            st.metric("Refute", format_percent(vr_data.get('refute_confidence')))
                        
                        if vr_data.get('reasoning'):
                            st.caption(vr_data['reasoning'])
            
            # Final decision
            final_decision = verification_result.get("final_decision")
            if final_decision:
                decision_data = final_decision.model_dump() if hasattr(final_decision, 'model_dump') else final_decision
                label = decision_data.get("label", "send_downstream")
                
                # Show manipulation score if available
                manipulation = verification_result.get("manipulation")
                if manipulation:
                    manip_data = manipulation.model_dump() if hasattr(manipulation, 'model_dump') else manipulation
                    manip_score = manip_data.get('manipulation_score', 0)
                    if manip_score > 0.6:
                        st.warning(f"⚠️ High manipulation score detected: {manip_score:.2%}")
                        st.caption(manip_data.get('explanation', ''))
                
                if label == "high_conf_true":
                    st.success("✅ **Verdict: VERIFIABLY TRUE** - Pipeline terminated")
                    st.markdown(f"**Reason:** {decision_data.get('why', 'N/A')}")
                    if decision_data.get("top_supporting_urls"):
                        with st.expander("📚 Supporting Sources"):
                            for url in decision_data["top_supporting_urls"][:5]:
                                st.markdown(f"- {url}")
                    st.stop()
                
                elif label == "high_conf_fake":
                    st.error("❌ **Verdict: VERIFIABLY FALSE** - Pipeline terminated")
                    st.markdown(f"**Reason:** {decision_data.get('why', 'N/A')}")
                    if decision_data.get("top_refuting_urls"):
                        with st.expander("📚 Refuting Sources"):
                            for url in decision_data["top_refuting_urls"][:5]:
                                st.markdown(f"- {url}")
                    st.stop()
                
                else:  # send_downstream
                    st.warning("⏩ **Verdict: SEND TO DOWNSTREAM** - Continuing to propaganda detection")
                    st.markdown(f"**Reason:** {decision_data.get('why', 'Insufficient evidence or uncertain claims')}")
                    st.info("💡 Proceeding to Stage 2: Propaganda & Emotion Analysis")
        
        progress_bar.progress(0.4)
        
        # ============= STAGE 2: PROPAGANDA + EMOTION =============
        if propaganda_enabled:
            status_text.text("Stage 2: Running propaganda and emotion detection...")
            progress_bar.progress(0.5)
            with st.spinner("Stage 2: Running propaganda and emotion detection..."):
                # Propaganda on root text (language-aware)
                if is_hindi_sample:
                    if hindi_prop_bundle is None:
                        hindi_prop_bundle = load_hindi_propaganda_model()
                    prop_tokenizer, prop_model = hindi_prop_bundle
                else:
                    if prop_model_bundle is None:
                        prop_model_bundle = load_propaganda_model()
                    prop_tokenizer, prop_model = prop_model_bundle

                prop_prob = predict_propaganda(prop_tokenizer, prop_model, row["root_text"])
                
                # Emotion on replies (translate Hindi replies silently)
                emotion_inputs = row_replies
                if is_hindi_sample and row_replies and translator_bundle:
                    try:
                        emotion_inputs = translate_hindi_to_english(row_replies, translator_bundle)
                    except Exception as exc:  # pragma: no cover - best effort fallback
                        st.caption(f"Note: Using original Hindi text for emotions (translation unavailable)")
                        emotion_inputs = row_replies

                if emotion_inputs:
                    collapsed_vectors = predict_emotions(emo_tokenizer, emo_model, emotion_inputs)
                    emotion_features = compute_emotion_features(collapsed_vectors)
                else:
                    collapsed_vectors = []
                    emotion_features = {
                        "mean_anger": 0.0, "mean_disgust": 0.0, "mean_fear": 0.0,
                        "mean_joy": 0.0, "mean_sadness": 0.0, "mean_surprise": 0.0,
                        "mean_neutral": 0.0, "entropy": 0.0, "variance": 0.0
                    }
                
                # Meta-classifier
                meta_input = pd.DataFrame([{
                    "text_pred": prop_prob,
                    **emotion_features
                }])
                meta_prob = meta_classifier.predict_proba(meta_input)[0][1]
                
                # Get embedding for GAT
                embedding = embedder.encode([row["root_text"]], convert_to_numpy=True)[0]
                
                results["propaganda"] = {
                    "text_pred": prop_prob,
                    "emotions": emotion_features,
                    "meta_prob": meta_prob,
                    "embedding": embedding
                }
            
            st.markdown("---")
            st.subheader("📊 Stage 2: Propaganda & Emotion Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Propaganda Prob", f"{prop_prob:.3f}")
            with col2:
                st.metric("Meta-Classifier Prob", f"{meta_prob:.3f}")
            with col3:
                verdict = "Propaganda" if meta_prob >= 0.5 else "Non-Propaganda"
                st.metric("Meta Verdict", verdict)
            
            st.caption(f"📊 Analyzed {len(row_replies)} replies for emotion features")
            
            st.markdown("**Emotion Features (from replies):**")
            emo_df = pd.DataFrame([emotion_features])
            st.dataframe(emo_df, use_container_width=True)
        
        progress_bar.progress(0.7)
        
        # ============= STAGE 3: GAT INFERENCE =============
        if gat_enabled:
            status_text.text("Stage 3: Running GAT inference on graph...")
            progress_bar.progress(0.8)
            
            # Check if this is a new/synthetic node
            is_synthetic = row.get("source") == "synthetic_hi"
            gat_max_nodes = len(gat_df) if 'gat_df' in locals() else 1154
            is_new_node = is_synthetic or idx >= gat_max_nodes
            
            if is_new_node:
                st.info(f"🆕 **Dynamic Node Addition:** This sample will be connected to the graph via K-nearest neighbors based on embedding similarity.")
            
            with st.spinner("Stage 3: Running GAT inference on graph..."):
                prop_score = results["propaganda"]["meta_prob"]
                embedding = results["propaganda"]["embedding"]
                
                gat_result = run_gat_inference(
                    gat_model, gat_graph, gat_embeddings, gat_df,
                    node_idx=idx, prop_score=prop_score, embedding=embedding,
                    is_new_node=is_new_node, k_neighbors=5
                )
                results["gat"] = gat_result
                
            st.markdown("---")
            st.subheader("🕸️ Stage 3: Graph Neural Network Decision")
            
            if gat_result.get("is_synthetic"):
                st.caption("🆕 Synthetic node dynamically connected to graph via KNN")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GAT Probability", f"{gat_result['probability']:.3f}")
            with col2:
                label_color = "🔴" if gat_result['label'] == "propaganda" else "🟢"
                st.metric("Final Prediction", f"{label_color} {gat_result['label'].upper()}")
            with col3:
                gt_display = gat_result['ground_truth']
                if gt_display == "N/A (synthetic)":
                    gt_display = "Propaganda" if row.get("true_label", 0) == 1 else "Non-Propaganda"
                st.metric("Ground Truth", gt_display)
            
            # Show top neighbors
            st.markdown("**🔗 Top Influential Neighbors (by attention weight):**")
            neighbors = gat_result["top_neighbors"]
            
            if neighbors:
                st.markdown("**📈 Graph View (Node + Neighbors)**")
                neighbor_fig = build_neighbor_graph_figure(
                    gat_result["node_idx"], neighbors, gat_result["label"]
                )
                st.pyplot(neighbor_fig, use_container_width=True)
                plt.close(neighbor_fig)

                for i, neighbor in enumerate(neighbors[:5], 1):  # Show top 5
                    with st.expander(f"Neighbor {i}: Node {neighbor['neighbor_idx']} (Attention: {neighbor['attention_weight']:.4f})"):
                        st.markdown(f"**Attention Weight:** {neighbor['attention_weight']:.4f}")
                        st.markdown(f"**Label:** {'Propaganda' if neighbor['true_label'] == 1 else 'Non-Propaganda'}")
                        st.markdown(f"**Text Preview:** {neighbor['text_preview']}...")
            else:
                st.info("No neighbor information available")
        
        progress_bar.progress(0.9)
        
        # ============= FINAL SUMMARY =============
        status_text.text("Generating summary...")
        progress_bar.progress(0.95)
        
        st.markdown("---")
        st.subheader("📑 Pipeline Summary")
        
        summary_data = {
            "Stage": [],
            "Component": [],
            "Result": [],
            "Confidence": []
        }
        
        if verification_enabled and results.get("verification"):
            fd = results["verification"].get("final_decision")
            if fd:
                fd_data = fd.model_dump() if hasattr(fd, 'model_dump') else fd
                summary_data["Stage"].append("Stage 1")
                summary_data["Component"].append("Fact Verification")
                summary_data["Result"].append(fd_data.get("label", "N/A").replace("_", " ").title())
                score = fd_data.get("post_verification_score")
                summary_data["Confidence"].append(f"{score:.3f}" if isinstance(score, (int, float)) else "N/A")
        
        if propaganda_enabled and results.get("propaganda"):
            summary_data["Stage"].append("Stage 2")
            summary_data["Component"].append("Propaganda Detection")
            verdict = "Propaganda" if results["propaganda"]["meta_prob"] >= 0.5 else "Non-Propaganda"
            summary_data["Result"].append(verdict)
            summary_data["Confidence"].append(f"{results['propaganda']['meta_prob']:.3f}")
        
        if gat_enabled and results.get("gat"):
            summary_data["Stage"].append("Stage 3")
            summary_data["Component"].append("GAT (Graph Neural Network)")
            summary_data["Result"].append(results["gat"]["label"].title())
            summary_data["Confidence"].append(f"{results['gat']['probability']:.3f}")
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Final verdict
        if gat_enabled and results.get("gat"):
            final_prob = results["gat"]["probability"]
            final_label = results["gat"]["label"]
            
            if final_label == "propaganda":
                st.error(f"🔴 **FINAL VERDICT: PROPAGANDA DETECTED** (Confidence: {final_prob:.1%})")
            else:
                st.success(f"🟢 **FINAL VERDICT: NON-PROPAGANDA** (Confidence: {1-final_prob:.1%})")
        
        elif propaganda_enabled and results.get("propaganda"):
            meta_prob = results["propaganda"]["meta_prob"]
            if meta_prob >= 0.5:
                st.error(f"🔴 **FINAL VERDICT: PROPAGANDA DETECTED** (Confidence: {meta_prob:.1%})")
            else:
                st.success(f"🟢 **FINAL VERDICT: NON-PROPAGANDA** (Confidence: {1-meta_prob:.1%})")

        progress_bar.progress(1.0)
        status_text.text("✅ Pipeline complete!")

# Footer
st.markdown("---")
st.markdown("""
**Pipeline Stages:**
1. **Claim Verification** - LLM-based fact-checking with evidence retrieval
2. **Propaganda + Emotion** - Text-based propaganda detection + emotion analysis of replies
3. **Graph Neural Network** - Final decision using learned graph structure and neighbor relationships
""")
