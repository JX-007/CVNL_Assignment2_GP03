"""
Changi Ops Assistant Prototype
CVNL Assignment - Bonus Application (Section 2.2)

A simple Streamlit application demonstrating:
1. RNN Text Intent Classification (Functional)
2. CNN Visual Classification (Placeholder for teammate's work)

Team Members:
- Nur Asyira Fitri Binte Razali (S10270371F) - RNN Intent Classification
- [Teammate Name] - CNN Aircraft/Luggage Classification
"""

import streamlit as st
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

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
# RNN MODEL DEFINITION (From your notebook)
# ============================================================================

class IntentClassifierLSTM(nn.Module):
    """
    LSTM-based Intent Classifier
    Architecture: Embedding → Bidirectional LSTM → Dropout → Fully Connected
    """
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
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate final hidden states from both directions
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dropped = self.dropout(hidden_cat)
        output = self.fc(dropped)
        return output

# ============================================================================
# CNN MODEL PLACEHOLDER (For teammate)
# ============================================================================

class AircraftClassifierCNN(nn.Module):
    """
    Placeholder CNN for Aircraft/Luggage Classification
    TODO: Replace with actual trained model from teammate
    """
    def __init__(self):
        super(AircraftClassifierCNN, self).__init__()
        # Placeholder architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(64 * 112 * 112, 10)  # Placeholder
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# UTILITY FUNCTIONS FOR RNN
# ============================================================================

def load_rnn_model(model_path, vocab_path, device):
    """Load the trained RNN intent classifier"""
    try:
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with saved parameters
        model = IntentClassifierLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes'],
            dropout=checkpoint['dropout']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, vocab_data, checkpoint
    
    except Exception as e:
        st.error(f"Error loading RNN model: {str(e)}")
        return None, None, None

def preprocess_text(text, vocab, max_len=50):
    """Convert text to tensor for RNN model"""
    # Tokenize and convert to indices
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor([indices], dtype=torch.long)

def predict_intent(text, model, vocab, idx2intent, device):
    """Predict intent from text input"""
    # Preprocess
    input_tensor = preprocess_text(text, vocab['vocabulary'])
    input_tensor = input_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    # Get intent name
    intent = idx2intent[str(predicted_idx)]
    
    return intent, confidence, probabilities[0].cpu().numpy()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title and description
    st.title("✈️ Changi Airport Operations Assistant")
    st.markdown("""
    **AI-Powered Passenger Query Triage & Visual Classification**
    
    This prototype demonstrates two key capabilities for Changi Airport operations:
    1. **Text Intent Classification**: Automatically categorize passenger queries
    2. **Visual Classification**: Identify aircraft types or luggage categories (Placeholder)
    
    ---
    """)
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        
        # Device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🖥️ Running on: **{device}**")
        
        st.markdown("---")
        
        # Feature selector
        st.header("🎯 Select Feature")
        feature = st.radio(
            "Choose a classifier:",
            ["🔤 Text Intent Classifier (RNN)", "🖼️ Visual Classifier (CNN)"],
            help="Select which AI model to use"
        )
        
        st.markdown("---")
        
        # Credits
        st.header("👥 Team Credits")
        st.markdown("""
        **RNN Intent Classifier:**
        - Nur Asyira Fitri Binte Razali
        - Student ID: S10270371F
        
        **CNN Visual Classifier:**
        - [Teammate Name]
        - [Student ID]
        """)
    
    # ========================================================================
    # FEATURE 1: TEXT INTENT CLASSIFICATION (RNN)
    # ========================================================================
    
    if "Text Intent" in feature:
        st.header("🔤 Text Intent Classification")
        st.markdown("""
        This feature uses an **LSTM-based RNN** to classify passenger queries into 8 intent categories.
        The model was trained on aviation-domain data (ATIS dataset) and achieves **98.5% accuracy**.
        """)
        
        # Model paths
        model_path = "checkpoints/intent_best.pth"
        vocab_path = "data/intent_vocab.json"
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            st.error(f"""
            ⚠️ **Model files not found!**
            
            Please ensure the following files are in the same directory as this script:
            - `{model_path}`
            - `{vocab_path}`
            
            Run your Jupyter notebook to generate these files first.
            """)
            st.stop()
        
        # Load model
        with st.spinner("Loading RNN model..."):
            model, vocab_data, checkpoint = load_rnn_model(model_path, vocab_path, device)
        
        if model is None:
            st.stop()
        
        # Display model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", checkpoint['model_type'])
        with col2:
            st.metric("Test Accuracy", f"{checkpoint['test_accuracy']:.2f}%")
        with col3:
            st.metric("F1 Score", f"{checkpoint['final_f1']:.4f}")
        
        st.markdown("---")
        
        # Intent categories
        with st.expander("📋 View Intent Categories"):
            st.markdown("""
            The model classifies queries into these **8 aviation-relevant intents**:
            
            1. **flight** - Flight status, gate changes, delays
            2. **airfare** - Ticket pricing, fare classes
            3. **airline** - Airline information, services
            4. **ground_service** - Ground transport, taxi, buses
            5. **airport** - Terminal directions, facilities
            6. **aircraft** - Aircraft type, seating
            7. **flight_time** - Departure/arrival times
            8. **capacity** - Seating capacity, special assistance
            """)
        
        st.markdown("---")
        
        # Text input
        st.subheader("💬 Try the Classifier")
        
        # Example queries
        example_queries = [
            "What gate does my flight depart from?",
            "How much does a ticket to Singapore cost?",
            "Which airlines fly to Tokyo?",
            "How do I get to the city from the airport?",
            "Where is the baggage claim area?",
            "What type of aircraft is used for this flight?",
            "What time does my flight arrive?",
            "How many seats are on this plane?"
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area(
                "Enter passenger query:",
                placeholder="e.g., What gate does my flight depart from?",
                height=100
            )
        
        with col2:
            st.markdown("**Quick Examples:**")
            for i, example in enumerate(example_queries[:4], 1):
                if st.button(f"Example {i}", key=f"ex_{i}"):
                    user_input = example
                    st.rerun()
        
        # Predict button
        if st.button("🔍 Classify Intent", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing query..."):
                    # Predict
                    intent, confidence, all_probs = predict_intent(
                        user_input, 
                        model, 
                        vocab_data, 
                        vocab_data['idx2intent'],
                        device
                    )
                    
                    # Display results
                    st.markdown("### 📊 Classification Results")
                    
                    # Main result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Predicted Intent:** `{intent}`")
                    with col2:
                        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                        st.info(f"**Confidence:** {confidence_color} {confidence:.2%}")
                    
                    # Routing suggestion
                    routing_map = {
                        'flight': '📍 Route to: **Operations Team** (Gate info, flight status)',
                        'airfare': '💰 Route to: **Commercial Team** (Pricing, booking)',
                        'airline': '🏢 Route to: **Airline Information Desk**',
                        'ground_service': '🚕 Route to: **Ground Transport Coordination**',
                        'airport': '🗺️ Route to: **Customer Service** (Directions, facilities)',
                        'aircraft': '✈️ Route to: **Technical Information Team**',
                        'flight_time': '⏰ Route to: **Flight Information System**',
                        'capacity': '💺 Route to: **Special Assistance / Seating Services**'
                    }
                    
                    st.info(routing_map.get(intent, '📋 Route to: General Customer Service'))
                    
                    # Confidence breakdown
                    with st.expander("🔍 View Detailed Confidence Scores"):
                        st.markdown("**All Intent Probabilities:**")
                        
                        # Create dataframe for visualization
                        intent_names = [vocab_data['idx2intent'][str(i)] for i in range(len(all_probs))]
                        probs_percent = all_probs * 100
                        
                        # Sort by probability
                        sorted_indices = np.argsort(probs_percent)[::-1]
                        
                        for idx in sorted_indices:
                            intent_name = intent_names[idx]
                            prob = probs_percent[idx]
                            
                            # Color based on probability
                            if prob > 50:
                                color = "#4CAF50"  # Green
                            elif prob > 10:
                                color = "#FF9800"  # Orange
                            else:
                                color = "#9E9E9E"  # Gray
                            
                            st.markdown(f"""
                            <div style="margin: 5px 0;">
                                <span style="display: inline-block; width: 150px;">{intent_name}</span>
                                <span style="display: inline-block; width: 200px; background: {color}; 
                                     height: 20px; border-radius: 3px;" 
                                     title="{prob:.2f}%">
                                    <span style="padding-left: 5px; color: white; font-size: 12px;">
                                        {prob:.2f}%
                                    </span>
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a query to classify.")
    
    # ========================================================================
    # FEATURE 2: VISUAL CLASSIFICATION (CNN) - PLACEHOLDER
    # ========================================================================
    
    else:  # Visual Classifier selected
        st.header("🖼️ Visual Classification (CNN)")
        st.markdown("""
        This feature uses a **Convolutional Neural Network (CNN)** to classify images of:
        - Aircraft types/families (e.g., Boeing 737, Airbus A380)
        - OR Luggage categories (e.g., suitcase, backpack, carry-on)
        """)
        
        # Placeholder warning
        st.warning("""
        ⚠️ **PLACEHOLDER SECTION**
        
        This section is a placeholder for your teammate's CNN visual classifier.
        
        **To integrate the actual model:**
        1. Replace `AircraftClassifierCNN` class with the actual CNN architecture
        2. Add model loading code (similar to RNN section)
        3. Implement image preprocessing (resize, normalize, etc.)
        4. Update the prediction logic
        
        **Current functionality:** Shows UI layout only, no actual predictions.
        """)
        
        st.markdown("---")
        
        # Image upload
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (aircraft or luggage)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of aircraft or luggage for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Uploaded Image:**")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("**Image Information:**")
                st.info(f"""
                - **Filename:** {uploaded_file.name}
                - **Size:** {image.size[0]} x {image.size[1]} pixels
                - **Format:** {image.format}
                """)
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                st.info("""
                🚧 **Placeholder Prediction**
                
                This would show:
                - Predicted class (e.g., "Boeing 737-800" or "Large Suitcase")
                - Confidence score
                - Top-3 predictions with probabilities
                
                **Integration needed:** Add your teammate's trained CNN model here.
                """)
                
                # Placeholder result display
                st.markdown("### 📊 Classification Results (Placeholder)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("**Predicted Class:** `[Aircraft/Luggage Type]`")
                with col2:
                    st.info("**Confidence:** 🟢 [XX.XX%]")
                
                # Placeholder routing
                st.info("📍 **Suggested Action:** [Based on classification]")
        
        else:
            st.info("👆 Upload an image to get started")
            
            # Example images section
            st.markdown("---")
            st.subheader("📸 Example Test Images")
            st.markdown("""
            You can test with:
            - Images from the FGVC-Aircraft dataset
            - Images from the Airport Luggage dataset
            - Your own photos (taken in public areas only)
            
            **Note:** Actual classification requires the trained CNN model from your teammate.
            """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()