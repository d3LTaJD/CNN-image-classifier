"""
Visualization utilities
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def visualize_samples(data_loader, class_names, num_samples=8):
    """Show sample images from dataset"""
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        img = img.numpy().transpose((1, 2, 0))
        
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {class_names[labels[i]]}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample CIFAR-100 Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved sample images to outputs/sample_images.png")


def plot_training_history(history):
    """Plot loss and accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved training history to outputs/training_history.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved confusion matrix to outputs/confusion_matrix.png")


def save_predictions(predictions, labels, class_names):
    """Save classification report"""
    os.makedirs('outputs', exist_ok=True)
    
    with open('outputs/classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(classification_report(labels, predictions, target_names=class_names))
        f.write("\n\nPer-class Accuracy:\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            class_mask = np.array(labels) == i
            if class_mask.sum() > 0:
                class_acc = (np.array(predictions)[class_mask] == i).mean() * 100
                f.write(f"{class_name:15s}: {class_acc:.2f}%\n")
    
    print("  ✓ Saved classification report to outputs/classification_report.txt")
    plot_confusion_matrix(labels, predictions, class_names)


def visualize_predictions(model, data_loader, class_names, num_samples=16, device='cpu'):
    """Visualize predictions on test samples"""
    model.eval()
    
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    images = images.cpu()
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        img = img.numpy().transpose((1, 2, 0))
        
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        
        axes[i].imshow(img)
        title = f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)'
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved predictions visualization to outputs/predictions_visualization.png")
