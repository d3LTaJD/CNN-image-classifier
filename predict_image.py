"""
Predict on custom images using trained CNN model
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import numpy as np

from model import get_model


CLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]


def load_image(image_path):
    """Load image from file"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_image(model, image_path, device='cpu', show_top_k=3):
    """Predict class for single image"""
    image = load_image(image_path)
    if image is None:
        return None
    
    img_tensor = preprocess_image(image).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k=show_top_k, dim=1)
    
    confidence = confidence.cpu().item()
    predicted_idx = predicted.cpu().item()
    top_k_probs = top_k_probs.cpu().numpy()[0]
    top_k_indices = top_k_indices.cpu().numpy()[0]
    
    top_k_results = []
    for i in range(show_top_k):
        top_k_results.append({
            'class': CLASS_NAMES[top_k_indices[i]],
            'confidence': top_k_probs[i] * 100
        })
    
    return {
        'predicted_class': CLASS_NAMES[predicted_idx],
        'confidence': confidence * 100,
        'top_k': top_k_results,
        'image': image
    }


def display_prediction(image_path, result):
    """Print prediction results"""
    print("\n" + "=" * 60)
    print(f"Image: {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"\nTop {len(result['top_k'])} Predictions:")
    print("-" * 60)
    for i, pred in enumerate(result['top_k'], 1):
        print(f"{i}. {pred['class']:15s}: {pred['confidence']:5.2f}%")
    print("=" * 60)


def predict_single_image(image_path, model_path='models/cnn_cifar100_20.pth', device='cpu'):
    """Predict on single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Train the model first: python train.py")
        return
    
    print("Loading model...")
    device = torch.device(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {'num_classes': 20})
    
    model = get_model(num_classes=config['num_classes'], dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"\nAnalyzing: {image_path}")
    result = predict_image(model, image_path, device, show_top_k=3)
    
    if result:
        display_prediction(image_path, result)
        
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(result['image'])
            axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].axis('off')
            info_text = f"Predicted: {result['predicted_class']}\n"
            info_text += f"Confidence: {result['confidence']:.2f}%\n\nTop Predictions:\n"
            for i, pred in enumerate(result['top_k'], 1):
                info_text += f"{i}. {pred['class']}: {pred['confidence']:.2f}%\n"
            
            axes[1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            os.makedirs('outputs', exist_ok=True)
            output_path = f"outputs/prediction_{os.path.basename(image_path)}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved to: {output_path}")
        except ImportError:
            pass
    else:
        print("Prediction failed.")


def predict_folder(folder_path, model_path='models/cnn_cifar100_20.pth', device='cpu'):
    """Predict on all images in folder"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"No images found in: {folder_path}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    print("\nLoading model...")
    device = torch.device(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {'num_classes': 20})
    
    model = get_model(num_classes=config['num_classes'], dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("\n" + "=" * 60)
    print("Predictions:")
    print("=" * 60)
    
    for image_path in image_files:
        result = predict_image(model, image_path, device, show_top_k=3)
        if result:
            display_prediction(image_path, result)
            print()


def main():
    parser = argparse.ArgumentParser(description='Predict on custom images')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--folder', type=str, help='Path to image folder')
    parser.add_argument('--model', type=str, default='models/cnn_cifar100_20.pth', help='Model path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    if args.image:
        predict_single_image(args.image, args.model, args.device)
    elif args.folder:
        predict_folder(args.folder, args.model, args.device)
    else:
        print("Error: Provide --image or --folder")
        print("\nUsage:")
        print("  python predict_image.py --image path/to/image.jpg")
        print("  python predict_image.py --folder path/to/folder")
        print("\nOr interactive mode:")
        print("  python predict_image.py")
        
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        image_path = input("\nEnter image path: ").strip().strip('"')
        
        if image_path:
            if os.path.isdir(image_path):
                predict_folder(image_path, args.model, args.device)
            elif os.path.isfile(image_path):
                predict_single_image(image_path, args.model, args.device)
            else:
                print(f"Error: Path not found: {image_path}")


if __name__ == "__main__":
    main()
