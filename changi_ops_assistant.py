"""
Changi Ops Assistant Prototype - DEMO READY VERSION
CVNL Assignment - Bonus Application (Section 2.2)

Features:
1. Text Analysis (Intent + Emotion) - Intent works now, Emotion placeholder
2. Aircraft Recognition - Placeholder (ready for teammate integration)

Team Members:
- Nur Asyira Fitri Binte Razali (S10270371F) - RNN Intent Classification ✅
- [Teammate Name] - CNN Aircraft Classification 🚧
- [Teammate Name] - Emotion/Sentiment Analysis 🚧
"""

import streamlit as st
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import re

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Changi Ops Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL: RNN INTENT CLASSIFIER
# ============================================================================

class IntentClassifierLSTM(nn.Module):
    """LSTM-based Intent Classifier for passenger queries"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout=0.3):
        super(IntentClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(hidden_cat)
        output = self.fc(dropped)
        return output

# ============================================================================
# UTILITY FUNCTIONS - RNN INTENT CLASSIFIER
# ============================================================================

def load_intent_model(model_path, vocab_path, device):
    """Load the trained RNN intent classifier"""
    try:
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = IntentClassifierLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, vocab_data, checkpoint
    
    except Exception as e:
        st.error(f"Error loading intent model: {str(e)}")
        return None, None, None

def preprocess_text_intent(text, vocab, max_len=50):
    """Convert text to tensor for intent model"""
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor([indices], dtype=torch.long)

def predict_intent(text, model, vocab, idx2intent, device):
    """Predict intent from text input"""
    input_tensor = preprocess_text_intent(text, vocab['vocabulary'])
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    intent = idx2intent[str(predicted_idx)]
    return intent, confidence, probabilities[0].cpu().numpy()

# ============================================================================
# PLACEHOLDER FUNCTIONS FOR MISSING MODELS
# ============================================================================

def simulate_emotion_prediction(text):
    """
    Placeholder emotion predictor based on simple keyword matching
    Used when teammate's emotion model is not available
    """
    text_lower = text.lower()
    
    # Simple rule-based emotion detection
    positive_keywords = ['great', 'excellent', 'wonderful', 'helpful', 'thank', 'happy', 'love', 'good', 'amazing', 'perfect']
    negative_keywords = ['terrible', 'awful', 'lost', 'delayed', 'frustrated', 'angry', 'bad', 'poor', 'worst', 'never', 'hate']
    
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    
    if positive_count > negative_count:
        return 'positive', 0.75  # Lower confidence for rule-based
    elif negative_count > positive_count:
        return 'negative', 0.75
    else:
        return 'neutral', 0.70

def simulate_aircraft_prediction(image):
    """
    Placeholder aircraft predictor
    Used when teammate's CNN model is not available
    """
    # Simulate random prediction for demo
    aircraft_types = [
        "Boeing 737-800",
        "Airbus A320",
        "Boeing 777-300ER",
        "Airbus A380"
    ]
    
    # Use image size as pseudo-random seed for consistency
    width, height = image.size
    index = (width + height) % len(aircraft_types)
    
    return aircraft_types[index], 0.65  # Lower confidence for placeholder

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.title("✈️ Changi Airport Operations Assistant")
    st.markdown("""
    **AI-Powered System for Passenger Experience & Operations**
    
    This prototype demonstrates AI capabilities for Changi Airport:
    1. **Text Analysis** - Intent classification (RNN) + Emotion analysis (placeholder)
    2. **Visual Recognition** - Aircraft identification (placeholder)
    
    ---
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🖥️ Device: **{device}**")
        
        st.markdown("### Model Status:")
        st.success("✅ Intent Classifier (RNN) - **READY**")
        st.warning("🚧 Emotion Analyzer (RNN) - Placeholder")
        st.warning("🚧 Aircraft CNN - Placeholder")
        
        st.markdown("---")
        
        st.header("🎯 Select Feature")
        feature = st.radio(
            "Choose AI capability:",
            [
                "💬 Text Analysis (Intent + Emotion)",
                "✈️ Aircraft Recognition (Visual)"
            ],
            help="Select which AI feature to use"
        )
        
        st.markdown("---")
        
        st.header("👥 Team Credits")
        st.markdown("""
        **Aircraft CNN:**
        - [Teammate Name]
        - [Student ID]
        - Status: 🚧 Integration pending
    
        **Intent Classifier (RNN):**
        - Nur Asyira Fitri Binte Razali
        - S10270371F
        - Status: ✅ **Complete**
        
        **Emotion Analyzer (RNN):**
        - [Teammate Name]
        - [Student ID]
        - Status: 🚧 Integration pending
        """)
    
    # ========================================================================
    # FEATURE 1: TEXT ANALYSIS (INTENT + EMOTION)
    # ========================================================================
    
    if "Text Analysis" in feature:
        st.header("💬 Text Analysis - Dual Classification")
        st.markdown("""
        **Intelligent text analysis** that classifies passenger messages in two ways:
        - 🎯 **Intent Classification** (LSTM Model - ✅ **WORKING**)
        - 💭 **Emotion Analysis** (Placeholder - 🚧 Awaiting teammate's model)
        
        This provides comprehensive understanding for better passenger service.
        """)
        
        # Check for intent model
        intent_model_path = "models/rnn_intent/checkpoints/intent_best.pth"
        intent_vocab_path = "models/rnn_intent/data/intent_vocab.json"
        
        intent_available = os.path.exists(intent_model_path) and os.path.exists(intent_vocab_path)
        
        if not intent_available:
            st.error(f"""
            ⚠️ **Intent model files not found!**
            
            Please ensure these files exist:
            - `{intent_model_path}`
            - `{intent_vocab_path}`
            
            Run your Jupyter notebook to generate these files.
            """)
            st.stop()
        
        # Load intent model
        with st.spinner("Loading intent model..."):
            intent_model, intent_vocab, intent_checkpoint = load_intent_model(
                intent_model_path, intent_vocab_path, device
            )
        
        if intent_model is None:
            st.stop()
        
        # Display model info
        st.markdown("### 📊 Model Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Intent Model", intent_checkpoint['model_type'])
        with col2:
            st.metric("Intent Accuracy", f"{intent_checkpoint['test_accuracy']:.2f}%")
        with col3:
            st.metric("Emotion Model", "Rule-based placeholder")
        with col4:
            st.metric("Status", "🚧 Awaiting integration")
        
        st.markdown("---")
        
        # Example queries
        example_queries = [
            "What gate does my flight depart from?",  # Neutral + flight
            "My luggage is lost and nobody is helping me!",  # Negative + ground_service
            "The staff was incredibly helpful, thank you!",  # Positive + airline
            "How do I get to the city from the airport?",  # Neutral + ground_service
        ]
        
        if 'text_query' not in st.session_state:
            st.session_state.text_query = ""
        
        st.subheader("💬 Enter Passenger Message")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area(
                "Passenger query or feedback:",
                value=st.session_state.text_query,
                placeholder="e.g., My flight is delayed and I'm very frustrated",
                height=120,
                key="text_input",
                help="Enter any passenger message - it will be analyzed for both intent and emotion"
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            for i in range(4):
                if st.button(f"Example {i+1}", key=f"text_ex_{i}", use_container_width=True):
                    st.session_state.text_query = example_queries[i]
                    st.rerun()
        
        if user_input != st.session_state.text_query:
            st.session_state.text_query = user_input
        
        # Analyze button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            current_query = st.session_state.text_query if st.session_state.text_query else user_input
            
            if current_query.strip():
                st.markdown("### 📊 Analysis Results")
                
                # Display the query
                st.info(f"**Message:** \"{current_query}\"")
                
                # Create two columns for intent and emotion
                col1, col2 = st.columns(2)
                
                # INTENT ANALYSIS (WORKING)
                with col1:
                    st.markdown("#### 🎯 Intent Classification")
                    st.caption("✅ Using trained LSTM model")
                    
                    with st.spinner("Analyzing intent..."):
                        intent, intent_conf, intent_probs = predict_intent(
                            current_query, intent_model, intent_vocab,
                            intent_vocab['idx2intent'], device
                        )
                        
                        st.success(f"**Intent:** `{intent}`")
                        conf_color = "🟢" if intent_conf > 0.8 else "🟡" if intent_conf > 0.5 else "🔴"
                        st.info(f"**Confidence:** {conf_color} {intent_conf:.2%}")
                        
                        # Routing suggestion
                        routing_map = {
                            'flight': '📍 Operations Team',
                            'airfare': '💰 Commercial Team',
                            'airline': '🏢 Airline Info Desk',
                            'ground_service': '🚕 Transport Coordination',
                            'airport': '🗺️ Customer Service',
                            'aircraft': '✈️ Technical Info',
                            'flight_time': '⏰ Flight Info System',
                            'capacity': '💺 Special Assistance'
                        }
                        st.caption(f"Route to: {routing_map.get(intent, 'General Service')}")
                        
                        # Show top predictions
                        with st.expander("🔍 View All Intent Probabilities"):
                            intent_names = [intent_vocab['idx2intent'][str(i)] for i in range(len(intent_probs))]
                            probs_percent = intent_probs * 100
                            sorted_indices = np.argsort(probs_percent)[::-1]
                            
                            for idx in sorted_indices[:5]:  # Show top 5
                                intent_name = intent_names[idx]
                                prob = probs_percent[idx]
                                st.write(f"**{intent_name}**: {prob:.2f}%")
                
                # EMOTION ANALYSIS (PLACEHOLDER)
                with col2:
                    st.markdown("#### 💭 Emotion/Sentiment Analysis")
                    st.caption("🚧 Using rule-based placeholder")
                    
                    with st.spinner("Analyzing emotion..."):
                        # Use placeholder emotion prediction
                        emotion, emotion_conf = simulate_emotion_prediction(current_query)
                        
                        # Color code based on emotion
                        if emotion == 'positive':
                            st.success(f"**Emotion:** `{emotion}`")
                        elif emotion == 'negative':
                            st.error(f"**Emotion:** `{emotion}`")
                        else:
                            st.info(f"**Emotion:** `{emotion}`")
                        
                        conf_color = "🟡"  # Always yellow for placeholder
                        st.warning(f"**Confidence:** {conf_color} {emotion_conf:.2%} (placeholder)")
                        
                        # Action suggestion
                        emotion_actions = {
                            'positive': '✅ Log as positive feedback',
                            'negative': '⚠️ Flag for service recovery',
                            'neutral': 'ℹ️ Standard processing'
                        }
                        st.caption(emotion_actions.get(emotion, 'Process feedback'))
                        
                        # Info about placeholder
                        with st.expander("ℹ️ About This Placeholder"):
                            st.markdown("""
                            **Current:** Simple rule-based emotion detection using keywords
                            
                            **When teammate provides model:**
                            - BiLSTM trained on Twitter Airline Sentiment dataset
                            - Higher accuracy (85-90%+)
                            - Better understanding of context
                            - More nuanced emotion detection
                            
                            **Files needed:**
                            - `checkpoints/emotion_best.pth`
                            - `data/emotion_vocab.json`
                            """)
                
                # Combined insights
                st.markdown("---")
                st.markdown("### 🎯 Combined Insights")
                
                # Generate combined action
                priority = "HIGH" if emotion == 'negative' else "NORMAL"
                
                st.info(f"""
                **Recommended Action:**
                - Priority: **{priority}**
                - Route to: {routing_map.get(intent, 'General Service')}
                - Tone: Handle with {"empathy and urgency" if emotion == 'negative' else "standard professionalism"}
                """)
                
                # Example of combined use case
                with st.expander("💡 Real-World Application at Changi Airport"):
                    st.markdown(f"""
                    **Passenger Message:** "{current_query}"
                    
                    **Without AI System:**
                    - Customer service reads message manually
                    - Decides department (potential misrouting)
                    - Response time: 5-10 minutes
                    
                    **With AI System:**
                    1. **Intent Detection** → Automatically route to {routing_map.get(intent, 'correct department')}
                    2. **Emotion Detection** → Priority set to {priority}
                    3. **Combined Intelligence** → Staff receives:
                       - Pre-categorized query
                       - Priority flag
                       - Suggested response tone
                    
                    **Benefits:**
                    - ⚡ Faster response (< 1 minute)
                    - 🎯 Accurate routing (98.5% accuracy)
                    - 😊 Better passenger experience
                    - 📊 Data for analytics
                    
                    **Current Status:**
                    - ✅ Intent classification: Fully functional (98.51% accuracy)
                    - 🚧 Emotion analysis: Placeholder (awaiting teammate's model)
                    """)
            else:
                st.warning("⚠️ Please enter a message to analyze")
    
    # ========================================================================
    # FEATURE 2: AIRCRAFT RECOGNITION (PLACEHOLDER)
    # ========================================================================
    
    else:  # Aircraft Recognition
        st.header("✈️ Aircraft Recognition (CNN)")
        st.markdown("""
        **Deep CNN classifier** for aircraft type identification.
        - Dataset: FGVC Aircraft / Commercial Aircraft Dataset
        - Use case: Ground operations planning, resource allocation, training
        
        🚧 **Status:** Awaiting teammate's CNN model integration
        """)
        
        st.warning("""
        **Placeholder Mode Active**
        
        This feature will use a simple simulation until teammate provides:
        - `checkpoints/best_cnn.pth` (trained CNN model)
        - `data/aircraft_classes.json` (class names)
        
        See `TEAM_INTEGRATION_GUIDE.md` for integration instructions.
        """)
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload aircraft image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of an aircraft for classification"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Uploaded Image:**")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("**Image Information:**")
                st.info(f"""
                - **Filename:** {uploaded_file.name}
                - **Size:** {image.size[0]} x {image.size[1]} pixels
                - **Format:** {image.format}
                """)
            
            if st.button("🔍 Identify Aircraft", type="primary", use_container_width=True):
                st.markdown("### 📊 Results")
                
                with st.spinner("Analyzing image..."):
                    # Use placeholder prediction
                    aircraft, confidence = simulate_aircraft_prediction(image)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Aircraft Type:** `{aircraft}`")
                    with col2:
                        st.warning(f"**Confidence:** 🟡 {confidence:.2%} (placeholder)")
                    
                    st.info("📍 **Use case:** Ground resource planning, stand allocation, crew preparation")
                    
                    # Info about placeholder
                    with st.expander("ℹ️ About This Placeholder"):
                        st.markdown("""
                        **Current:** Simple rule-based aircraft identification
                        
                        **When teammate provides CNN model:**
                        - Deep CNN trained on FGVC-Aircraft dataset
                        - Higher accuracy (85-95%+)
                        - Identifies specific variants (e.g., 737-800 vs 737-900)
                        - Supports 50+ aircraft types
                        
                        **Real-World Benefits:**
                        - Automatic gate assignment based on aircraft size
                        - Ground equipment preparation (correct tow bars, stairs)
                        - Fuel truck allocation
                        - Crew briefing automation
                        
                        **Files needed from teammate:**
                        - `checkpoints/best_cnn.pth`
                        - `data/aircraft_classes.json`
                        """)
                    
                    with st.expander("💡 Integration Instructions for Teammate"):
                        st.code("""
# Add this to your CNN notebook after training:

import torch
import json

# Save model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_classes': len(aircraft_classes),
    'aircraft_classes': aircraft_classes,
    'test_accuracy': test_accuracy
}
torch.save(checkpoint, 'checkpoints/best_cnn.pth')

# Save class names
class_data = {
    'aircraft_classes': aircraft_classes,
    'num_classes': len(aircraft_classes)
}
with open('data/aircraft_classes.json', 'w') as f:
    json.dump(class_data, f, indent=2)
                        """, language='python')
        else:
            st.info("👆 Upload an aircraft image to begin analysis")
            
            # Show example of what this will do
            st.markdown("---")
            st.markdown("### 📸 Example Use Cases")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Operations Planning**")
                st.write("""
                - Identify incoming aircraft
                - Allocate appropriate gate
                - Prepare ground equipment
                """)
            
            with col2:
                st.markdown("**Training & Education**")
                st.write("""
                - Staff training tool
                - Quick aircraft identification
                - Aviation students
                """)
            
            with col3:
                st.markdown("**Analytics**")
                st.write("""
                - Track aircraft types
                - Optimize resource allocation
                - Predict maintenance needs
                """)

if __name__ == "__main__":
    main()