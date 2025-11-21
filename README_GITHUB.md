# CNN Image Classification - CIFAR-100 (20 Superclasses)

Deep Convolutional Neural Network for image classification on CIFAR-100 dataset with 20 superclasses.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python train.py
```

### Run Web App
```bash
cd web
streamlit run app.py
```

## 📁 Project Structure

- `model.py` - CNN architecture
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `predict_image.py` - Single image prediction
- `web/app.py` - Streamlit web interface
- `utils.py` - Utility functions

## 📊 Results

- **Test Accuracy:** 70-80%
- **Model:** Deep CNN (12M+ parameters)
- **Training:** 50 epochs with data augmentation

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repo
4. Set main file: `web/app.py`
5. Deploy!

## 📝 Note

- Model files (`models/*.pth`) are not included (too large)
- Train the model locally first, then upload model to cloud storage
- Or use Git LFS for model files

