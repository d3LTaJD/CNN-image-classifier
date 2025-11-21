"""
Quick test script - Test model on CIFAR-100 test images
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

from model import get_model

def quick_test(num_images=10):
    """Test model on random CIFAR-100 test images"""
    print("=" * 60)
    print("Quick Test on CIFAR-100 Test Images")
    print("=" * 60)
    
    # Load model
    model_path = 'models/cnn_cifar100_20.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first: python train.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = get_model(num_classes=20, dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Convert to coarse labels
    coarse_labels = [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ]
    
    for i in range(len(test_dataset.targets)):
        test_dataset.targets[i] = coarse_labels[test_dataset.targets[i]]
    
    class_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    # Test on random images
    print(f"\nTesting on {num_images} random images...")
    print("-" * 60)
    
    correct = 0
    for _ in range(num_images):
        idx = random.randint(0, len(test_dataset) - 1)
        image, true_label = test_dataset[idx]
        
        # Predict
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted = predicted.cpu().item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100
        
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} True: {class_names[true_label]:25s} | "
              f"Predicted: {class_names[predicted]:25s} | "
              f"Confidence: {confidence:.1f}%")
    
    print("-" * 60)
    accuracy = (correct / num_images) * 100
    print(f"\nAccuracy: {correct}/{num_images} = {accuracy:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    import os
    quick_test(num_images=20)

