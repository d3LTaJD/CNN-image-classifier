"""
Check if GPU/CUDA is available and properly configured
"""

import torch
import sys

print("=" * 60)
print("GPU/CUDA Availability Check")
print("=" * 60)

# Check PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")

# Check CUDA availability
print(f"\n2. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   ✓ GPU is available!")
    print(f"\n3. CUDA Version: {torch.version.cuda}")
    print(f"4. cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"5. Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n6. GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test GPU computation
    print("\n7. Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✓ GPU computation successful!")
    except Exception as e:
        print(f"   ✗ GPU computation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Your system is ready for GPU training!")
    print("=" * 60)
    print("\nThe training script will automatically use GPU.")
    
else:
    print("\n   ✗ GPU is NOT available")
    print("\n3. Possible reasons:")
    print("   - No NVIDIA GPU installed")
    print("   - PyTorch installed without CUDA support")
    print("   - CUDA drivers not installed")
    print("   - GPU not compatible with CUDA")
    
    print("\n" + "=" * 60)
    print("Solutions:")
    print("=" * 60)
    
    # Check if it's a CPU-only PyTorch installation
    print("\n4. Checking PyTorch installation type...")
    if 'cpu' in torch.__version__.lower() or not torch.cuda.is_available():
        print("   ⚠️  You have CPU-only PyTorch installed")
        print("\n   To install PyTorch with GPU support:")
        print("   1. Visit: https://pytorch.org/get-started/locally/")
        print("   2. Select your configuration:")
        print("      - OS: Windows/Linux/Mac")
        print("      - Package: pip")
        print("      - CUDA: 11.8 or 12.1 (if you have NVIDIA GPU)")
        print("\n   Example command for CUDA 11.8:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n   Example command for CUDA 12.1:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n5. Check your GPU:")
    print("   - Open Device Manager (Windows)")
    print("   - Look under 'Display adapters'")
    print("   - If you see NVIDIA GPU, you can use it!")
    
    print("\n6. Check CUDA installation:")
    print("   - Run: nvidia-smi (in Command Prompt)")
    print("   - If it shows GPU info, CUDA drivers are installed")
    
    print("\n" + "=" * 60)
    print("Note: Training will work on CPU, but it's slower.")
    print("GPU training is 5-10x faster!")
    print("=" * 60)

print("\n" + "=" * 60)
print("Current Device Selection in Code:")
print("=" * 60)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"This is what train.py will use automatically.")

