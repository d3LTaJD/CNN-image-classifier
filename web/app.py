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

# Page configuration
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Preprocess image for the model"""
    transform = transforms.Compose([
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
    # Title and header
    st.title("🖼️ CNN Image Classifier")
    st.markdown("### Upload an image and get instant predictions!")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("Deep CNN trained on CIFAR-100 (20 superclasses)")
        st.markdown("**20 Categories**: " + ", ".join([v.split()[1] for v in CLASS_NAMES.values()][:5]) + "...")
        
        st.markdown("---")
        show_top_k = st.slider("Show top predictions", 1, 10, 3)
        st.markdown("---")
        st.markdown("**Note**: Images resized to 32x32 pixels")
    
    # Load model
    model, device_or_error = load_model()
    
    if model is None:
        st.error(f"❌ {device_or_error}")
        st.info("💡 **To train the model, run:** `python train.py` in the main project directory")
        st.stop()
    
    device = device_or_error
    
    # File uploader
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Upload an image to classify"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Your Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🔮 Prediction Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                result = predict_image(model, image, device, top_k=show_top_k)
            
            # Display main prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            color = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
            st.markdown("### Main Prediction")
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; 
                        border-radius: 10px; margin: 10px 0;'>
                <h2 style='margin: 0;'>{color} {CLASS_NAMES[predicted_class]}</h2>
                <h3 style='margin: 10px 0; color: #1f77b4;'>{confidence:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Top predictions
            st.markdown("### Top Predictions")
            for i, pred in enumerate(result['top_k'], 1):
                class_name = pred['class']
                conf = pred['confidence']
                
                # Progress bar for confidence
                st.markdown(f"**{i}. {CLASS_NAMES[class_name]}**")
                st.progress(conf / 100)
                st.caption(f"{conf:.2f}%")
                st.markdown("---")
        
        with st.expander("ℹ️ More Information"):
            st.markdown(f"**Class**: {CLASS_NAMES[predicted_class]} | **Confidence**: {confidence:.2f}%")
            st.markdown("**Model**: Deep CNN (12M+ params, 50 epochs)")
    
    else:
        st.info("👆 **Upload an image above to get started!**")


if __name__ == "__main__":
    main()

