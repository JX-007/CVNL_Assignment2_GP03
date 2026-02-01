"""
Emotional Analysis Streamlit App
CVNL Assignment 2 - Emotional Sentiment Analysis

This app demonstrates the emotional/sentiment analysis feature
trained on Twitter sentiment data (negative, neutral, positive).
"""

import streamlit as st
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import numpy as np
import re
from collections import Counter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Emotional Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# EMOTION CLASSIFIER MODEL (MATCHES TRAINING ARCHITECTURE)
# ============================================================================

class EmbeddingPackable(nn.Module):
    """Wrapper for embedding layer to handle packed sequences"""
    def __init__(self, embd_layer: nn.Embedding):
        super().__init__()
        self.embd_layer = embd_layer

    def forward(self, x):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            seqs, lengths = pad_packed_sequence(x, batch_first=True)
            seqs = self.embd_layer(seqs.to(x.data.device))
            return pack_padded_sequence(
                seqs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            return self.embd_layer(x)


class BiLSTMSentiment(nn.Module):
    """Bidirectional LSTM for Sentiment/Emotion Classification
    
    Trained checkpoint configuration:
    - vocab_size: 11845
    - embed_dim: 128
    - hidden_dim: 178
    - num_layers: 4
    - num_classes: 3
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 num_classes=3, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = EmbeddingPackable(nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx))
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x_packed):
        x_emb = self.embedding(x_packed)
        _, (h_n, _) = self.lstm(x_emb)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=1)

        h = self.dropout(h)
        logits = self.fc(h)
        return logits


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(s: str) -> str:
    """Clean and normalize text"""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)      # remove URLs
    s = re.sub(r"@\w+", " ", s)                  # remove @mentions
    s = s.replace("&amp;", " and ")
    s = s.replace("#", " ")                      # keep hashtag word, drop '#'
    s = re.sub(r"[^a-z0-9\s']", " ", s)          # remove punctuation except apostrophe
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    """Tokenize text"""
    return clean_text(s).split()


# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_vocab():
    """Load vocabulary"""
    vocab_path = Path("models/rnn_intent/data/emotion_vocab.json")
    if not vocab_path.exists():
        st.error(f"Vocabulary file not found at {vocab_path}")
        return None
    
    with open(vocab_path, 'r') as f:
        data = json.load(f)
    
    # Handle nested vocabulary structure
    if isinstance(data, dict) and 'vocabulary' in data:
        word2idx = data['vocabulary']
    else:
        word2idx = data
    
    return word2idx


@st.cache_resource
def load_emotion_model():
    """Load the trained emotion classifier model
    
    Configuration matches actual checkpoint:
    - vocab_size: 11845
    - embed_dim: 128
    - hidden_dim: 178
    - num_layers: 4
    - num_classes: 3
    """
    model_path = Path("models/rnn_intent/checkpoints/emotion_best.pth")
    
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
        return None
    
    # Model configuration (MUST match checkpoint exactly)
    vocab_size = 11845
    embed_dim = 128
    hidden_dim = 178
    num_layers = 4
    num_classes = 3
    dropout = 0.3  # reasonable default
    pad_idx = 0
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # Load weights with proper handling for different PyTorch versions
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device


def predict_emotion(text: str):
    """Predict emotion for given text"""
    word2idx = load_vocab()
    model_info = load_emotion_model()
    
    if word2idx is None or model_info is None:
        return None, None, None
    
    model, device = model_info
    
    # Preprocess text
    tokens = tokenize(text)
    if not tokens:
        return None, None, "Text is empty after preprocessing"
    
    PAD, UNK = "<PAD>", "<UNK>"
    ids = [word2idx.get(w, word2idx.get(UNK, 1)) for w in tokens]
    
    # Create tensor and pack sequence
    x = torch.tensor([ids], dtype=torch.long)  # (1, seq_len)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    
    # Pack the sequence
    x_packed = pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    x_packed = x_packed.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(x_packed)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Map to emotion labels
    emotion_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    emotion = emotion_map[pred_class]
    
    return emotion, confidence, probs[0].cpu().numpy()


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("😊 Emotional Analysis System")
st.markdown("---")

st.write("""
This application analyzes the emotional sentiment of text inputs.
It uses a trained LSTM neural network to classify sentiment into three categories:
- **Negative** 😞: Expressing dissatisfaction or frustration
- **Neutral** 😐: Objective or factual statements
- **Positive** 😊: Expressing satisfaction or happiness
""")

st.markdown("---")

# Input section
st.subheader("📝 Enter Text for Analysis")
user_input = st.text_area(
    "Type or paste the text you want to analyze:",
    height=100,
    placeholder="Example: This service is amazing! I'm very happy with my experience."
)

col1, col2 = st.columns(2)

with col1:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Analysis results
if analyze_button and user_input:
    with st.spinner("Analyzing emotion..."):
        emotion, confidence, probs = predict_emotion(user_input)
    
    if emotion is None:
        st.error("Error: Could not load model or vocabulary files.")
    else:
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Main prediction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if emotion == "Positive":
                st.success(f"### Detected Emotion: {emotion} 😊")
            elif emotion == "Negative":
                st.error(f"### Detected Emotion: {emotion} 😞")
            else:
                st.info(f"### Detected Emotion: {emotion} 😐")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Probability distribution
        st.markdown("---")
        st.subheader("📈 Probability Distribution")
        
        prob_data = {
            "Negative 😞": probs[0],
            "Neutral 😐": probs[1],
            "Positive 😊": probs[2]
        }
        
        col1, col2, col3 = st.columns(3)
        
        for (emotion_name, prob), col in zip(prob_data.items(), [col1, col2, col3]):
            with col:
                st.metric(emotion_name, f"{prob:.1%}")
        
        # Visual bar chart
        st.bar_chart(prob_data)
        
        # Text preview
        st.markdown("---")
        st.subheader("📄 Analyzed Text")
        st.text_area("", value=user_input, height=80, disabled=True)

elif analyze_button and not user_input:
    st.warning("Please enter some text to analyze.")

# Example inputs
st.markdown("---")
st.subheader("💡 Try Examples")

example_texts = [
    "This is absolutely terrible! Worst experience ever!",
    "The flight was on time and everything was fine.",
    "Amazing service! Staff was incredibly helpful and friendly!"
]

for i, example in enumerate(example_texts):
    if st.button(f"Try Example {i+1}: '{example[:40]}...'"):
        st.session_state.example_clicked = example
        st.rerun()

if "example_clicked" in st.session_state:
    st.info(f"**Selected Example:** {st.session_state.example_clicked}")
    with st.spinner("Analyzing emotion..."):
        emotion, confidence, probs = predict_emotion(st.session_state.example_clicked)
    
    if emotion:
        st.success(f"**Result:** {emotion} (Confidence: {confidence:.1%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Emotional Analysis System | CVNL Assignment 2</p>
    <p>Model: BiLSTM-based Sentiment Classifier | Trained on Twitter Sentiment Data</p>
</div>
""", unsafe_allow_html=True)
