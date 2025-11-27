# 🧠 CNN Image Classification - CIFAR-100 (20 Superclasses)

A comprehensive Deep Convolutional Neural Network (CNN) project for image classification on the CIFAR-100 dataset with 20 superclasses. This project includes a complete deep learning pipeline from training to deployment, featuring a beautiful web interface for real-time predictions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📋 Project Overview

This project implements a **Deep CNN architecture** to classify images from the CIFAR-100 dataset into **20 superclasses**:

- 🐋 Aquatic Mammals
- 🐟 Fish
- 🌸 Flowers
- 🍽️ Food Containers
- 🥕 Fruits & Vegetables
- 💡 Household Electrical Devices
- 🪑 Household Furniture
- 🐝 Insects
- 🦁 Large Carnivores
- 🏗️ Large Man-made Outdoor Things
- 🌲 Large Natural Outdoor Scenes
- 🐘 Large Omnivores and Herbivores
- 🦊 Medium Mammals
- 🦀 Non-insect Invertebrates
- 👤 People
- 🦎 Reptiles
- 🐹 Small Mammals
- 🌳 Trees
- 🚗 Vehicles 1
- 🚛 Vehicles 2

## 🌐 Live Demo

**Try the web app:** [Deploy on Streamlit Cloud](https://share.streamlit.io)

The web application allows you to:
- 📤 Upload images via drag & drop
- 🔮 Get instant predictions with confidence scores
- 📊 View top-K predictions
- 🎨 Beautiful, modern UI with dark theme

## 🏗️ Architecture

The **Deep CNN model** consists of:

- **4 Convolutional Blocks** with increasing filters (64→128→256→512)
- **Batch Normalization** after each convolutional layer
- **Max Pooling** layers for downsampling (32×32 → 16×16 → 8×8 → 4×4 → 2×2)
- **Dropout** for regularization (0.4 in FC layers, 0.3 in conv blocks)
- **4 Fully Connected** layers (2048 → 1024 → 512 → 256 → 20)
- **Total Parameters**: ~12M+ trainable parameters

### Architecture Details:
```
Input: 3×32×32 (RGB images)
  ↓
Conv Block 1: 3→64→64→128 filters → MaxPool → 16×16
  ↓
Conv Block 2: 128→128→256→256 filters → MaxPool → 8×8
  ↓
Conv Block 3: 256→512→512 filters → MaxPool → 4×4
  ↓
Conv Block 4: 512→512→512 filters → MaxPool → 2×2
  ↓
Flatten: 512×2×2 = 2048
  ↓
FC1: 2048 → 1024 (ReLU + Dropout)
  ↓
FC2: 1024 → 512 (ReLU + Dropout)
  ↓
FC3: 512 → 256 (ReLU + Dropout)
  ↓
FC4: 256 → 20 (Output: 20 classes)
```

## 📁 Project Structure

```
.
├── model.py              # Deep CNN model architecture
├── train.py              # Training script with data augmentation
├── evaluate.py           # Model evaluation script
├── predict_image.py      # Predict on custom images
├── utils.py              # Visualization utilities
├── requirements.txt      # Python dependencies (includes Streamlit)
├── README.md             # This file
├── STREAMLIT_DEPLOY.md   # Streamlit Cloud deployment guide
├── MODEL_LIMITATIONS.md  # Known limitations and solutions
├── .streamlit/           # Streamlit configuration
│   └── config.toml       # Streamlit Cloud settings
├── web/                   # Web application
│   ├── app.py            # Streamlit web app
│   └── requirements.txt # Web dependencies
├── data/                  # CIFAR-100 dataset (auto-downloaded)
├── models/                # Saved model checkpoints
│   └── cnn_cifar100_20.pth  # Trained model (139MB, using Git LFS)
└── outputs/               # Generated plots and reports
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── predictions_visualization.png
    └── classification_report.txt
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning repository)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/d3LTaJD/CNN-image-classifier.git
   cd CNN-image-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: For GPU support (recommended for faster training), install PyTorch with CUDA:
   ```bash
   # Visit https://pytorch.org/ to get the correct command for your system
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Git LFS** (for model file)
   ```bash
   git lfs install
   git lfs pull
   ```

## 💻 Usage

### 1. Training the Model

Train the Deep CNN model on CIFAR-100 dataset:

```bash
python train.py
```

**Training Configuration** (can be modified in `train.py`):
- **Batch size**: 128
- **Number of epochs**: 50
- **Learning rate**: 0.001
- **Optimizer**: Adam (with L2 regularization)
- **Learning rate scheduler**: ReduceLROnPlateau (reduces LR when loss plateaus)
- **Data augmentation**: Random flip, crop, color jitter, rotation

The script will:
- ✅ Automatically download CIFAR-100 dataset
- ✅ Convert 100 fine classes to 20 superclasses
- ✅ Display training progress with progress bars
- ✅ Save the best model checkpoint to `models/cnn_cifar100_20.pth`
- ✅ Generate training curves and visualizations
- ✅ Show per-epoch accuracy and loss

### 2. Evaluating the Model

Evaluate a trained model:

```bash
python evaluate.py --model models/cnn_cifar100_20.pth
```

This will:
- Load the saved model
- Evaluate on test set
- Generate confusion matrix
- Create prediction visualizations
- Save detailed classification report

### 3. Predict on Your Own Images

Test the model on your own custom images:

```bash
# Single image
python predict_image.py --image path/to/your/image.jpg

# Batch prediction (folder)
python predict_image.py --folder path/to/image/folder

# Interactive mode
python predict_image.py
```

**Example:**
```bash
python predict_image.py --image my_image.jpg
```

This will show:
- Predicted class (e.g., "flowers")
- Confidence percentage
- Top 3 predictions with confidence scores
- Visualization saved to `outputs/` folder

**Note**: The model was trained on CIFAR-100 (20 superclasses). Your images will be automatically preprocessed (resize → center crop → resize to 32×32).

### 4. Web App Interface 🌐

Run a beautiful web interface to upload and test images:

```bash
cd web
streamlit run app.py
```

**Features:**
- 📤 Drag & drop image upload
- 🔮 Real-time predictions with confidence scores
- 📊 Top-K predictions (configurable 1-10)
- 🎨 Beautiful, modern dark theme UI
- ⚡ Fast inference with model caching

The app will open automatically in your browser at `http://localhost:8501`

**Note:** Make sure the model file exists (`models/cnn_cifar100_20.pth`) before using the web app.

### 5. Testing Model Architecture

Test the model with dummy input:

```bash
python model.py
```

This will:
- Create a model instance
- Test forward pass with dummy input
- Display model architecture
- Show total number of parameters

## 📊 Expected Results

After training for 50 epochs, you should achieve:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~70-80%
- **Training Time**: ~20-40 minutes (on CPU) or ~5-10 minutes (on GPU)

**Note**: Results may vary based on hardware and random initialization.

## 📈 Outputs

The project generates several outputs in the `outputs/` directory:

1. **sample_images.png** - Sample images from the dataset
2. **training_history.png** - Loss and accuracy curves over epochs
3. **confusion_matrix.png** - Confusion matrix for test predictions
4. **predictions_visualization.png** - Visual comparison of predictions vs ground truth (16 samples)
5. **classification_report.txt** - Detailed per-class performance metrics (precision, recall, F1-score)

## 🚀 Deployment

### Streamlit Cloud (Recommended)

This project is ready for deployment on Streamlit Cloud!

**Quick Deploy:**
1. Push your code to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository: `d3LTaJD/CNN-image-classifier`
6. Main file path: `web/app.py`
7. Click "Deploy"

**Detailed instructions:** See [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md)

**Note:** The model file (139MB) is stored using Git LFS, which Streamlit Cloud supports automatically.

## 🔧 Customization

### Modify Model Architecture

Edit `model.py` to change:
- Number of convolutional layers
- Filter sizes and channels
- Dropout rates
- Fully connected layer sizes
- Activation functions

### Adjust Training Parameters

Edit `train.py` CONFIG dictionary to modify:
- Batch size
- Number of epochs
- Learning rate
- Data augmentation techniques
- Optimizer settings

### Use Different Dataset

To use a different dataset:
1. Modify data loading in `train.py` (`get_data_loaders` function)
2. Update number of classes in CONFIG
3. Update class names in all relevant files
4. Adjust model architecture if needed

## 🎓 Learning Objectives

By working on this project, you will learn:
- ✅ Deep CNN architecture design
- ✅ PyTorch framework basics
- ✅ Data preprocessing and augmentation
- ✅ Training deep learning models
- ✅ Model evaluation and metrics
- ✅ Visualization of results
- ✅ Web app development with Streamlit
- ✅ Model deployment on cloud platforms

## 📚 Key Concepts Demonstrated

1. **Convolutional Layers**: Feature extraction from images
2. **Pooling Layers**: Dimensionality reduction
3. **Batch Normalization**: Stabilizing training and improving convergence
4. **Dropout**: Preventing overfitting
5. **Data Augmentation**: Improving generalization
6. **Learning Rate Scheduling**: Adaptive learning rate adjustment
7. **Model Deployment**: Web interface and cloud hosting

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in `train.py` (CONFIG['batch_size'])
   - Use smaller model architecture

2. **Dataset download fails**
   - Check internet connection
   - Manually download CIFAR-100 from https://www.cs.toronto.edu/~kriz/cifar.html

3. **Slow training**
   - Use GPU if available
   - Reduce number of epochs for testing
   - Use smaller batch size

4. **Model file not found**
   - Make sure Git LFS is installed: `git lfs install`
   - Pull LFS files: `git lfs pull`
   - Or train the model first: `python train.py`

5. **Import errors in web app**
   - Make sure you're in the project root directory
   - Install all dependencies: `pip install -r requirements.txt`
   - Check that `model.py` is in the parent directory

## 📝 Project Report Ideas

For your college project report, you can include:
- Introduction to CNNs and image classification
- Dataset description (CIFAR-100 with 20 superclasses)
- Model architecture explanation
- Training methodology and hyperparameters
- Results and analysis (accuracy, confusion matrix)
- Confusion matrix interpretation
- Discussion of improvements and limitations
- Conclusion and future work
- Web application demonstration

## 🚀 Extensions & Improvements

Ideas to enhance the project:
- ✅ Add more data augmentation techniques
- ⬜ Implement transfer learning with ResNet/VGG
- ⬜ Add model ensembling
- ✅ Create a web interface with Streamlit
- ✅ Deploy model using Streamlit Cloud
- ⬜ Experiment with different optimizers (SGD, AdamW)
- ⬜ Implement early stopping
- ⬜ Add TensorBoard logging
- ⬜ Support for batch image upload in web app
- ⬜ Add model explainability (Grad-CAM)

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Created for college mini project on Deep Learning.

**GitHub Repository**: [d3LTaJD/CNN-image-classifier](https://github.com/d3LTaJD/CNN-image-classifier)

## 🙏 Acknowledgments

- CIFAR-100 dataset creators (Alex Krizhevsky, Vinod Nair, Geoffrey Hinton)
- PyTorch development team
- Streamlit team for the amazing framework
- Open-source community

## 📖 Additional Resources

- [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md) - Known limitations and solutions
- [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md) - Detailed deployment guide
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Happy Learning! 🎉**

If you have any questions or encounter issues, feel free to:
- Open an issue on GitHub
- Explore the code and experiment with different configurations
- Check the troubleshooting section above

**Star ⭐ this repository if you find it helpful!**
