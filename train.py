"""
Training Script for CNN Image Classification on CIFAR-100 (20 superclasses)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from model import get_model
from utils import visualize_samples, plot_training_history, save_predictions


# Configuration - Optimized for Higher Accuracy
CONFIG = {
    'batch_size': 128,  # Increased for better GPU utilization
    'num_epochs': 50,   # More epochs for better learning
    'learning_rate': 0.001,
    'num_classes': 20,  # CIFAR-100 with 20 superclasses
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_model': True,
    'model_path': 'models/cnn_cifar100_20.pth',
    'visualize': True,
    'weight_decay': 1e-4,  # L2 regularization
}


def get_data_loaders(batch_size=64):
    """
    Load and prepare CIFAR-100 dataset with 20 superclasses (coarse labels)
    
    Returns:
        train_loader, test_loader, class_names
    """
    # Enhanced data augmentation for training set (improves accuracy)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Simple transform for test set (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-100 datasets
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Convert fine labels to coarse labels (20 superclasses)
    # CIFAR-100 coarse class mapping (maps 100 fine classes to 20 coarse classes)
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
    
    # Convert labels to coarse labels
    # CIFAR-100 returns fine labels (0-99), we need to convert to coarse (0-19)
    def convert_to_coarse(dataset, coarse_labels):
        # Convert targets list to list if it's not already
        if isinstance(dataset.targets, list):
            dataset.targets = [coarse_labels[label] for label in dataset.targets]
        else:
            # If it's a tensor, convert to list first
            targets_list = dataset.targets.tolist() if hasattr(dataset.targets, 'tolist') else list(dataset.targets)
            dataset.targets = [coarse_labels[label] for label in targets_list]
        return dataset
    
    train_dataset = convert_to_coarse(train_dataset, coarse_labels)
    test_dataset = convert_to_coarse(test_dataset, coarse_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # CIFAR-100 coarse class names (20 superclasses)
    class_names = [
        'aquatic_mammals',      # 0
        'fish',                 # 1
        'flowers',              # 2
        'food_containers',      # 3
        'fruit_and_vegetables', # 4
        'household_electrical_devices', # 5
        'household_furniture',  # 6
        'insects',              # 7
        'large_carnivores',     # 8
        'large_man-made_outdoor_things', # 9
        'large_natural_outdoor_scenes', # 10
        'large_omnivores_and_herbivores', # 11
        'medium_mammals',        # 12
        'non-insect_invertebrates', # 13
        'people',               # 14
        'reptiles',             # 15
        'small_mammals',        # 16
        'trees',                # 17
        'vehicles_1',           # 18
        'vehicles_2'            # 19
    ]
    
    return train_loader, test_loader, class_names


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    
    Returns:
        average_loss, accuracy, predictions
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def main():
    """Main training function"""
    print("=" * 60)
    print("CNN Image Classification - CIFAR-100 (20 Superclasses)") 
    print("=" * 60)
    
    # Setup device
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    # Create models directory
    if CONFIG['save_model']:
        os.makedirs('models', exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-100 dataset (20 superclasses)...")
    train_loader, test_loader, class_names = get_data_loaders(CONFIG['batch_size'])
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some samples
    if CONFIG['visualize']:
        print("\nVisualizing sample images...")
        visualize_samples(train_loader, class_names, num_samples=8)
    
    # Initialize model
    print("\nInitializing improved deep CNN model...")
    model = get_model(num_classes=CONFIG['num_classes'], dropout_rate=0.4)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model capacity increased for better accuracy!")
    
    # Loss and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=CONFIG['learning_rate'], 
                          weight_decay=CONFIG.get('weight_decay', 1e-4))
    
    # Better learning rate scheduler - reduce LR when plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print("=" * 60)
    
    best_test_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate based on test loss (ReduceLROnPlateau)
        scheduler.step(test_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if test_acc > best_test_acc and CONFIG['save_model']:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': CONFIG
            }, CONFIG['model_path'])
            print(f"  ✓ Saved best model (Test Acc: {test_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history)
    
    # Final evaluation with detailed results
    print("\nFinal Evaluation:")
    print("-" * 60)
    test_loss, test_acc, predictions, labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Save predictions for analysis
    save_predictions(predictions, labels, class_names)
    
    print("\n" + "=" * 60)
    print("All done! Check the results in the outputs/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()

