"""
Extract CIFAR-100 test images and save them for testing
"""

import torch
from torchvision import datasets, transforms
from PIL import Image
import os

def extract_test_images(num_images_per_class=5, output_dir='test_images'):
    """
    Extract test images from CIFAR-100 dataset
    
    Args:
        num_images_per_class: Number of images to extract per class
        output_dir: Directory to save images
    """
    print("=" * 60)
    print("Extracting CIFAR-100 Test Images")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CIFAR-100 test dataset
    print("\nLoading CIFAR-100 test dataset...")
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=None  # No transform, get raw images
    )
    
    # Coarse labels mapping (20 superclasses)
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
    
    # Class names
    class_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]
    
    # Convert fine labels to coarse labels
    for i in range(len(test_dataset.targets)):
        test_dataset.targets[i] = coarse_labels[test_dataset.targets[i]]
    
    # Count images per class
    class_counts = {i: 0 for i in range(20)}
    
    print(f"\nExtracting {num_images_per_class} image per class (20 total)...")
    print(f"Saving to: {output_dir}/")
    print("-" * 60)
    
    saved_count = 0
    
    for idx, (image, label) in enumerate(test_dataset):
        class_idx = label
        
        # Check if we need more images for this class
        if class_counts[class_idx] < num_images_per_class:
            class_name = class_names[class_idx]
            
            # Create class directory
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            filename = f"{class_name}_{class_counts[class_idx] + 1}.png"
            filepath = os.path.join(class_dir, filename)
            image.save(filepath)
            
            class_counts[class_idx] += 1
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} images...")
    
    print("-" * 60)
    print(f"\n✅ Extraction complete!")
    print(f"Total images saved: {saved_count}")
    print(f"Location: {os.path.abspath(output_dir)}")
    print("\nImages organized by class:")
    for i, class_name in enumerate(class_names):
        count = class_counts[i]
        if count > 0:
            print(f"  {class_name}: {count} images")
    
    print("\n" + "=" * 60)
    print("You can now test these images in the web app!")
    print("=" * 60)

if __name__ == "__main__":
    # Extract 1 image per class (20 total)
    extract_test_images(num_images_per_class=1, output_dir='test_images')

