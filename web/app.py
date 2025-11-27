"""
Streamlit Web App for CNN Image Classification
Upload images and get predictions in real-time!
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model

# Page configuration with theme
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit menu and header
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# CIFAR-100 (20 superclasses) class names
CLASS_NAMES = {
    'aquatic_mammals': '🐋 Aquatic Mammals', 'fish': '🐟 Fish', 'flowers': '🌸 Flowers',
    'food_containers': '🍽️ Food Containers', 'fruit_and_vegetables': '🥕 Fruits & Vegetables',
    'household_electrical_devices': '💡 Electrical Devices', 'household_furniture': '🪑 Furniture',
    'insects': '🐝 Insects', 'large_carnivores': '🦁 Large Carnivores',
    'large_man-made_outdoor_things': '🏗️ Man-made Outdoor', 'large_natural_outdoor_scenes': '🌲 Natural Scenes',
    'large_omnivores_and_herbivores': '🐘 Large Herbivores', 'medium_mammals': '🦊 Medium Mammals',
    'non-insect_invertebrates': '🦀 Invertebrates', 'people': '👤 People', 'reptiles': '🦎 Reptiles',
    'small_mammals': '🐹 Small Mammals', 'trees': '🌳 Trees', 'vehicles_1': '🚗 Vehicles 1',
    'vehicles_2': '🚛 Vehicles 2'
}
CLASS_NAMES_LIST = list(CLASS_NAMES.keys())

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* Remove ALL white backgrounds - most aggressive */
html, body {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
    background-color: #1A1A2E !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Hide Streamlit header at top completely */
header[data-testid="stHeader"],
div[data-testid="stHeader"],
.stApp > header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
button[data-testid="baseButton-header"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Remove top padding that creates white space */
.stApp > div:first-child,
div[data-testid="stAppViewContainer"] > div:first-child,
section[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

html, body {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
    background-color: #1A1A2E !important;
}

/* Top header area - remove white */
header[data-testid="stHeader"],
div[data-testid="stHeader"],
.stApp > header {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
    background-color: #1A1A2E !important;
}

/* Remove white from top toolbar - COMPLETE REMOVAL */
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
header[data-testid="stHeader"],
div[data-testid="stHeader"],
.stApp > header,
button[data-testid="baseButton-header"],
div[class*="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* Remove top padding completely */
.stApp {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

div[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.block-container {
    padding-top: 0.5rem !important;
}

/* Main content area - remove white background */
.main .block-container {
    background: transparent !important;
    max-width: 1150px;
    padding-top: 1.5rem;
}

.stApp {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
}

/* Remove white backgrounds from all elements */
div[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
}

/* Center layout & spacing */
.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    background: transparent !important;
}

/* Sidebar - Make sure it's always visible */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FF6B6B, #FF8E53, #FF6B9D) !important;
    color: white !important;
    min-width: 300px !important;
}

/* Sidebar toggle button */
button[data-testid="baseButton-header"] {
    display: block !important;
    visibility: visible !important;
}

/* Sidebar text - ensure visibility with strong contrast */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] code {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7) !important;
    font-weight: 500 !important;
}

/* Make captions more visible */
[data-testid="stSidebar"] .stCaption {
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.6) !important;
    opacity: 1 !important;
    font-weight: 400 !important;
}

/* Sidebar headings */
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5) !important;
}

/* Small pill for model tag */
.model-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.3);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #FFFFFF !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
    font-weight: 600;
}

/* Hero section */
.hero {
    border-radius: 20px;
    padding: 2.3rem;
    background: linear-gradient(120deg, #FF6B6B, #4ECDC4, #45B7D1);
    color: #FFFFFF !important;
    box-shadow: 0px 18px 40px rgba(0,0,0,0.4);
    margin-bottom: 2rem;
}

.hero * {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.6) !important;
}

.hero div {
    color: #FFFFFF !important;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFFFFF !important;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
}
.hero-sub {
    font-size: 1rem;
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5) !important;
    opacity: 1 !important;
}

/* Feature cards */
.feature-card {
    border-radius: 16px;
    padding: 1.5rem;
    background: linear-gradient(135deg, #16213E 0%, #1A1A2E 100%);
    border: 2px solid #4ECDC4;
    box-shadow: 0px 10px 30px rgba(78,205,196,0.3);
    transition: 0.2s ease;
    text-align: center;
    color: #E5E7EB;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: #FF6B6B;
    box-shadow: 0px 18px 30px rgba(255,107,107,0.4);
}

/* Remove all white backgrounds */
div[class*="element-container"],
div[class*="stMarkdown"],
section[data-testid="stMain"] {
    background: transparent !important;
}

/* Upload section */
.upload-card {
    border-radius: 20px;
    padding: 2rem;
    background: linear-gradient(135deg, #16213E 0%, #1A1A2E 100%);
    border: 2px dashed #4ECDC4;
    box-shadow: 0px 10px 35px rgba(78,205,196,0.2);
    margin: 1.5rem 0;
}

/* Prediction card */
.prediction-card {
    border-radius: 20px;
    padding: 2rem;
    background: linear-gradient(120deg, #FF6B6B, #4ECDC4, #45B7D1);
    color: #FFFFFF !important;
    box-shadow: 0px 18px 40px rgba(0,0,0,0.4);
    text-align: center;
    margin: 1.5rem 0;
}

.prediction-card * {
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5) !important;
}

/* Top prediction items */
.pred-item {
    border-radius: 12px;
    padding: 1rem;
    background: linear-gradient(135deg, #16213E 0%, #0F3460 100%);
    border: 1px solid #4ECDC4;
    margin: 0.75rem 0;
    color: #E5E7EB;
    box-shadow: 0px 4px 15px rgba(78,205,196,0.2);
}

/* Settings label */
.settings-label {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #FFFFFF !important;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.4) !important;
}

/* Image container */
.image-container {
    border-radius: 16px;
    padding: 1rem;
    background: linear-gradient(135deg, #16213E 0%, #1A1A2E 100%);
    border: 2px solid #FF6B6B;
    margin: 1rem 0;
    box-shadow: 0px 4px 15px rgba(255,107,107,0.3);
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
}

/* Slider text visibility */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider span,
[data-testid="stSidebar"] .stSlider div {
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
}

/* Make all Streamlit widget labels visible in sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText {
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.6) !important;
}

/* Info boxes styling */
.stInfo {
    background-color: rgba(78,205,196,0.15);
    border-left: 4px solid #4ECDC4;
    border-radius: 8px;
    color: #E5E7EB;
}

.stWarning {
    background-color: rgba(255,107,107,0.15);
    border-left: 4px solid #FF6B6B;
    border-radius: 8px;
    color: #E5E7EB;
}

/* Force dark background - remove ALL white */
.stApp {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
}

div[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
}

section[data-testid="stMain"] {
    background: transparent !important;
}

div[class*="element-container"],
div[class*="stMarkdown"],
div[class*="column"] {
    background: transparent !important;
}

/* File uploader area */
div[data-testid="stFileUploader"] {
    background: transparent !important;
}

div[data-baseweb="file-uploader"] {
    background: rgba(22,33,62,0.6) !important;
    border-color: #4ECDC4 !important;
}

div[data-baseweb="file-uploader"] > div {
    background: rgba(22,33,62,0.6) !important;
}

/* Remove ALL white backgrounds - comprehensive */
div[class*="stColumn"],
div[class*="stHorizontalBlock"],
section[data-testid="stMain"] > div,
div[class*="element-container"] > div,
div[data-testid*="column"],
div[data-testid*="horizontal"] {
    background: transparent !important;
}

/* Force dark on any remaining white elements */
div[style*="background-color: white"],
div[style*="background: white"],
div[style*="background-color: #fff"],
div[style*="background: #fff"],
div[style*="background-color: #ffffff"],
div[style*="background: #ffffff"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Override Streamlit's default white backgrounds */
.css-1d391kg,
.css-1v0mbdj,
.css-1y4p8pa,
.css-18e3th9 {
    background: transparent !important;
}

/* Make sure main area has dark background */
.main {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%) !important;
}

/* Remove top white space completely */
div[data-testid="stHeader"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove any white padding/margin at top */
.stApp > div:first-child {
    background: transparent !important;
    padding-top: 0 !important;
}

/* Force dark on all possible white elements */
div[class*="css"] {
    background: transparent !important;
}

/* Remove Streamlit's default header and top white space */
#MainMenu {
    visibility: hidden;
    display: none !important;
}

footer {
    visibility: hidden;
    display: none !important;
}

.stDeployButton {
    display: none !important;
}

/* Hide Streamlit header completely */
header[data-testid="stHeader"],
div[data-testid="stHeader"],
.stApp > header {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Remove top padding that creates white space */
.stApp > div:first-child,
div[data-testid="stAppViewContainer"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Force remove any white at top */
div[style*="padding-top"],
div[style*="margin-top"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path='../models/cnn_cifar100_20.pth'):
    """Load the trained model (cached for performance)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try multiple possible paths
        possible_paths = [
            '../models/cnn_cifar100_20.pth',
            'models/cnn_cifar100_20.pth',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cnn_cifar100_20.pth')
        ]
        
        model_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if model_path is None:
            return None, "Model file not found. Please train the model first by running: python train.py"
        
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {'num_classes': 20})
        model = get_model(num_classes=config.get('num_classes', 20), dropout_rate=0.4)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def preprocess_image(image):
    """Preprocess image for the model - improved for better accuracy"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)


def predict_image(model, image, device, top_k=3):
    """Make prediction on image"""
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=1)
    
    confidence = confidence.cpu().item()
    predicted_idx = predicted.cpu().item()
    top_k_probs = top_k_probs.cpu().numpy()[0]
    top_k_indices = top_k_indices.cpu().numpy()[0]
    
    top_k_results = []
    for i in range(top_k):
        top_k_results.append({
            'class': CLASS_NAMES_LIST[top_k_indices[i]],
            'confidence': top_k_probs[i] * 100
        })
    
    return {
        'predicted_class': CLASS_NAMES_LIST[predicted_idx],
        'confidence': confidence * 100,
        'top_k': top_k_results
    }


def main():
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📋 About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0,0,0,0.7);">
            <h3 style="color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin: 0.5rem 0;">🧠 Deep CNN</h3>
            <h5 style="color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin: 0.25rem 0;">Trained on CIFAR-100</h5>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #FFFFFF; text-shadow: 1px 1px 3px rgba(0,0,0,0.6); font-size: 0.9rem; margin: 0.5rem 0;">
            20 superclasses • 20 categories (Aquatic, Fish, Flowers, Food, Fruits, …)
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="settings-label">Show top predictions</div>', unsafe_allow_html=True)
        top_k = st.slider("Number of top predictions", 1, 10, 3, label_visibility="collapsed", help="Number of top predictions to display")
        st.markdown(f"""
        <p style="color: #FFFFFF; text-shadow: 1px 1px 3px rgba(0,0,0,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
            Currently showing: <strong>{top_k}</strong> predictions
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            '<span class="model-pill">⚠️ Model limitations</span>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <p style="color: #FFFFFF; text-shadow: 1px 1px 3px rgba(0,0,0,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
            Works best on CIFAR-100 style images (32×32, single centered object).
        </p>
        """, unsafe_allow_html=True)

    # ---------- MAIN HERO ----------
    st.markdown(
        """
        <div class="hero">
            <div style="font-size: 3em; margin-bottom: 0.5rem;">🖼️🔍</div>
            <div class="hero-title">CNN Image Classifier</div>
            <div class="hero-sub">
                Upload an image and get instant predictions using a deep CNN trained on CIFAR-100.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # ---------- FEATURES ROW ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-card">🎯<br/><strong>20 Superclasses</strong><br/><span style="font-size:0.85rem;opacity:0.9;">From animals to vehicles and everyday objects.</span></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="feature-card">⚡<br/><strong>Instant Results</strong><br/><span style="font-size:0.85rem;opacity:0.9;">Predictions update as soon as you upload a new image.</span></div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="feature-card">🤖<br/><strong>AI Powered</strong><br/><span style="font-size:0.85rem;opacity:0.9;">Built with a deep convolutional neural network.</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # Load model
    model, device_or_error = load_model()
    
    if model is None:
        st.error(f"❌ {device_or_error}")
        st.info("💡 **To train the model, run:** `python train.py` in the main project directory")
        st.stop()
    
    device = device_or_error

    # ---------- UPLOAD SECTION ----------
    st.markdown(
        '<div class="upload-card"><h3 style="color: #E5E7EB; margin-bottom: 0.5rem;">📤 Upload Image</h3><p style="color: #94a3b8; margin-bottom: 1rem;">Drag & drop or browse a file to classify.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        " ",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PREDICTION SECTION ----------
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #E5E7EB; margin-bottom: 1rem;">📷 Your Image</h4>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h4 style="color: #E5E7EB; margin-bottom: 1rem;">🔮 Prediction Results</h4>', unsafe_allow_html=True)
            
            # Make prediction
            with st.spinner("🤖 AI is analyzing your image..."):
                result = predict_image(model, image, device, top_k=top_k)
            
            # Main prediction card
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Confidence indicator
            if confidence >= 70:
                conf_emoji = "🟢"
                conf_text = "High Confidence"
            elif confidence >= 40:
                conf_emoji = "🟡"
                conf_text = "Medium Confidence"
            else:
                conf_emoji = "🔴"
                conf_text = "Low Confidence"
            
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 3em; margin-bottom: 0.5rem;">{conf_emoji}</div>
                <div style="font-size: 1.8em; font-weight: 700; margin-bottom: 0.5rem;">
                    {CLASS_NAMES[predicted_class]}
                </div>
                <div style="font-size: 2.5em; font-weight: 800; margin: 1rem 0;">
                    {confidence:.2f}%
                </div>
                <div style="font-size: 0.9em; opacity: 0.9;">
                    {conf_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top predictions
            st.markdown('<h5 style="color: #E5E7EB; margin-top: 1.5rem;">Top Predictions</h5>', unsafe_allow_html=True)
            for i, pred in enumerate(result['top_k'], 1):
                class_name = pred['class']
                conf = pred['confidence']
                
                st.markdown(f"""
                <div class="pred-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #E5E7EB;">
                            {i}. {CLASS_NAMES[class_name]}
                        </span>
                        <span style="font-weight: 700; color: #4ECDC4;">
                            {conf:.2f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf / 100)
            
            # Additional info
            with st.expander("ℹ️ More Information", expanded=False):
                st.markdown(f"""
                <div style="color: #E5E7EB;">
                    <p><strong>Class:</strong> {CLASS_NAMES[predicted_class]}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    <p><strong>Model:</strong> Deep CNN (12M+ params, 50 epochs)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if confidence < 50:
                    st.warning("⚠️ Low confidence prediction. The image might not match CIFAR-100 classes well.")
                
                st.info("💡 **Tip**: For best results, use images similar to CIFAR-100 dataset (small, centered objects on simple backgrounds)")


if __name__ == "__main__":
    main()
