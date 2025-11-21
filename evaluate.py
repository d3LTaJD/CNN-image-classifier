"""
Evaluation Script for Trained CNN Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os

from model import get_model
from utils import visualize_predictions, plot_confusion_matrix
from sklearn.metrics import classification_report


def load_test_data(batch_size=64):
    """Load CIFAR-100 test dataset with 20 superclasses"""
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
    
    # Convert fine labels to coarse labels (20 superclasses)
    coarse_labels = [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ]
    
    # Convert fine labels to coarse labels
    if isinstance(test_dataset.targets, list):
        test_dataset.targets = [coarse_labels[label] for label in test_dataset.targets]
    else:
        targets_list = test_dataset.targets.tolist() if hasattr(test_dataset.targets, 'tolist') else list(test_dataset.targets)
        test_dataset.targets = [coarse_labels[label] for label in targets_list]
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    class_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    return test_loader, class_names


def evaluate_model(model_path, device='cpu', visualize=True):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to run evaluation on
        visualize: Whether to generate visualizations
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Setup device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, class_names = load_test_data(batch_size=64)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {'num_classes': 10})
    
    model = get_model(num_classes=config['num_classes'], dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    if 'test_acc' in checkpoint:
        print(f"Best test accuracy during training: {checkpoint['test_acc']:.2f}%")
    
    # Evaluate
    print("\nEvaluating model...")
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.2f}%")
    print(f"Correct Predictions: {correct}/{total}")
    
    # Classification report
    print("\n" + "-" * 60)
    print("Per-class Performance:")
    print("-" * 60)
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Generate visualizations
    if visualize:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, class_names)
        
        # Prediction samples
        visualize_predictions(model, test_loader, class_names, num_samples=16, device=device)
        
        print("\nVisualizations saved to outputs/ directory")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained CNN model')
    parser.add_argument('--model', type=str, default='models/cnn_cifar100_20.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        device=args.device,
        visualize=not args.no_vis
    )

