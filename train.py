"""
Training script for CIFAR-100 (20 superclasses)
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


CONFIG = {
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'num_classes': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_model': True,
    'model_path': 'models/cnn_cifar100_20.pth',
    'visualize': True,
    'weight_decay': 1e-4,
}


def get_data_loaders(batch_size=64):
    """Load CIFAR-100 and convert to 20 superclasses"""
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    # Map 100 fine classes to 20 coarse classes
    coarse_labels = [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ]
    
    def convert_to_coarse(dataset, coarse_labels):
        if isinstance(dataset.targets, list):
            dataset.targets = [coarse_labels[label] for label in dataset.targets]
        else:
            targets_list = dataset.targets.tolist() if hasattr(dataset.targets, 'tolist') else list(dataset.targets)
            dataset.targets = [coarse_labels[label] for label in targets_list]
        return dataset
    
    train_dataset = convert_to_coarse(train_dataset, coarse_labels)
    test_dataset = convert_to_coarse(test_dataset, coarse_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    class_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    return train_loader, test_loader, class_names


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
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
    
    return running_loss / len(test_loader), 100 * correct / total, all_predictions, all_labels


def main():
    print("=" * 60)
    print("CNN Image Classification - CIFAR-100 (20 Superclasses)") 
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    if CONFIG['save_model']:
        os.makedirs('models', exist_ok=True)
    
    print("\nLoading CIFAR-100 dataset...")
    train_loader, test_loader, class_names = get_data_loaders(CONFIG['batch_size'])
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    if CONFIG['visualize']:
        print("\nVisualizing sample images...")
        visualize_samples(train_loader, class_names, num_samples=8)
    
    print("\nInitializing model...")
    model = get_model(num_classes=CONFIG['num_classes'], dropout_rate=0.4)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    print("=" * 60)
    
    best_test_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(test_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
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
    
    print("\nGenerating training plots...")
    plot_training_history(history)
    
    print("\nFinal Evaluation:")
    print("-" * 60)
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    save_predictions(predictions, labels, class_names)
    
    print("\n" + "=" * 60)
    print("All done! Check the results in the outputs/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
