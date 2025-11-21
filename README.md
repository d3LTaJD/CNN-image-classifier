# CNN Image Classification - CIFAR-10 Dataset

A comprehensive Convolutional Neural Network (CNN) project for image classification on the CIFAR-10 dataset. This project is designed as a mini project for college students to learn about deep learning and computer vision.

## 📋 Project Overview

This project implements a CNN architecture to classify images from the CIFAR-10 dataset into 10 categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚛 Truck

## 🏗️ Architecture

The CNN model consists of:
- **3 Convolutional Blocks** with Batch Normalization
- **Max Pooling** layers for downsampling
- **Dropout** for regularization
- **2 Fully Connected** layers for classification
- **Total Parameters**: ~3.8M trainable parameters

## 📁 Project Structure

```
.
├── model.py              # CNN model architecture
├── train.py              # Training script
├── evaluate.py           # Model evaluation script
├── predict_image.py      # Predict on your own custom images
├── utils.py              # Utility functions for visualization
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── web/                  # Web application
│   ├── app.py           # Streamlit web app
│   ├── requirements.txt # Web dependencies
│   └── run.bat          # Quick start script
├── data/                # CIFAR-10 dataset (auto-downloaded)
├── models/              # Saved model checkpoints
└── outputs/             # Generated plots and reports
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

### Setup Steps

1. **Clone or download this project**
   ```bash
   cd "cognitive project"
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

## 💻 Usage

### 1. Training the Model

Train the CNN model on CIFAR-10 dataset:

```bash
python train.py
```

**Training Configuration** (can be modified in `train.py`):
- Batch size: 64
- Number of epochs: 20
- Learning rate: 0.001
- Optimizer: Adam
- Learning rate scheduler: StepLR (decays every 7 epochs)

The script will:
- ✅ Automatically download CIFAR-10 dataset
- ✅ Display training progress with progress bars
- ✅ Save the best model checkpoint to `models/cnn_cifar10.pth`
- ✅ Generate training curves and visualizations

### 2. Evaluating the Model

Evaluate a trained model:

```bash
python evaluate.py --model models/cnn_cifar10.pth
```

This will:
- Load the saved model
- Evaluate on test set
- Generate confusion matrix
- Create prediction visualizations
- Save classification report

### 3. Predict on Your Own Images

Test the model on your own custom images:

```bash
# Single image
python predict_image.py --image path/to/your/image.jpg

# Or use interactive mode (double-click predict_image.bat on Windows)
python predict_image.py
```

**Example:**
```bash
python predict_image.py --image my_cat.jpg
```

This will show:
- Predicted class (e.g., "cat")
- Confidence percentage
- Top 3 predictions
- Visualization saved to `outputs/` folder

**Note**: The model was trained on CIFAR-10 (10 classes). Your images will be automatically resized to 32x32 pixels.

### 4. Web App Interface 🌐

Run a beautiful web interface to upload and test images:

```bash
cd web
streamlit run app.py
```

Or double-click `web/run.bat` on Windows!

**Features:**
- 📤 Drag & drop image upload
- 🔮 Real-time predictions
- 📊 Visual confidence scores
- 🎨 Beautiful, user-friendly interface

The app will open automatically in your browser at `http://localhost:8501`

**Note:** Make sure to train the model first (`python train.py`) before using the web app.

### 5. Testing Model Architecture

Test the model with dummy input:

```bash
python model.py
```

## 📊 Expected Results

After training for 20 epochs, you should achieve:
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~75-82%
- **Training Time**: ~20-40 minutes (on CPU) or ~5-10 minutes (on GPU)

## 📈 Outputs

The project generates several outputs in the `outputs/` directory:

1. **sample_images.png** - Sample images from the dataset
2. **training_history.png** - Loss and accuracy curves
3. **confusion_matrix.png** - Confusion matrix for test predictions
4. **predictions_visualization.png** - Visual comparison of predictions vs ground truth
5. **classification_report.txt** - Detailed per-class performance metrics

## 🔧 Customization

### Modify Model Architecture

Edit `model.py` to change:
- Number of convolutional layers
- Filter sizes
- Dropout rates
- Fully connected layer sizes

### Adjust Training Parameters

Edit `train.py` CONFIG dictionary to modify:
- Batch size
- Number of epochs
- Learning rate
- Data augmentation

### Use Different Dataset

To use a different dataset:
1. Modify data loading in `train.py` (`get_data_loaders` function)
2. Update number of classes in CONFIG
3. Update class names

## 🎓 Learning Objectives

By working on this project, you will learn:
- ✅ CNN architecture design
- ✅ PyTorch framework basics
- ✅ Data preprocessing and augmentation
- ✅ Training deep learning models
- ✅ Model evaluation and metrics
- ✅ Visualization of results

## 📚 Key Concepts Demonstrated

1. **Convolutional Layers**: Feature extraction from images
2. **Pooling Layers**: Dimensionality reduction
3. **Batch Normalization**: Stabilizing training
4. **Dropout**: Preventing overfitting
5. **Data Augmentation**: Improving generalization
6. **Transfer Learning**: Can be extended with pre-trained models

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in `train.py` (CONFIG['batch_size'])

2. **Dataset download fails**
   - Check internet connection
   - Manually download CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html

3. **Slow training**
   - Use GPU if available
   - Reduce number of epochs for testing
   - Use smaller batch size

## 📝 Project Report Ideas

For your college project report, you can include:
- Introduction to CNNs and image classification
- Dataset description (CIFAR-10)
- Model architecture explanation
- Training methodology
- Results and analysis
- Confusion matrix interpretation
- Discussion of improvements
- Conclusion and future work

## 🚀 Extensions & Improvements

Ideas to enhance the project:
- Add more data augmentation techniques
- Implement transfer learning with ResNet/VGG
- Add model ensembling
- Create a web interface with Flask/Streamlit
- Deploy model using Gradio
- Experiment with different optimizers
- Implement early stopping
- Add TensorBoard logging

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Created for college mini project on Deep Learning.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- PyTorch development team
- Open-source community

---

**Happy Learning! 🎉**

If you have any questions or encounter issues, feel free to explore the code and experiment with different configurations!

