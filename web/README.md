# Web App - CNN Image Classifier

A beautiful web interface for testing your CNN model on custom images!

## 🚀 Quick Start

### 1. Install Web Dependencies

```bash
cd web
pip install -r requirements.txt
```

Or from the main project directory:
```bash
pip install streamlit
```

### 2. Make Sure Model is Trained

First, train the model (if not already done):
```bash
cd ..
python train.py
```

### 3. Run the Web App

```bash
cd web
streamlit run app.py
```

Or from the main directory:
```bash
streamlit run web/app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📸 Features

- ✅ **Easy Image Upload** - Drag and drop or click to upload
- ✅ **Real-time Predictions** - Instant results
- ✅ **Top-K Predictions** - See top 3 (or more) predictions
- ✅ **Confidence Scores** - Visual progress bars
- ✅ **Beautiful UI** - Clean and modern interface
- ✅ **Responsive Design** - Works on different screen sizes

## 🎯 How to Use

1. **Start the app**: Run `streamlit run app.py`
2. **Upload image**: Click "Browse files" or drag and drop
3. **View results**: See predictions with confidence scores
4. **Try more**: Upload different images to test!

## 📁 File Structure

```
web/
├── app.py              # Main Streamlit application
├── requirements.txt    # Web dependencies
└── README.md          # This file
```

## 🔧 Troubleshooting

**Error: "Model file not found"**
- Make sure you've trained the model first: `python train.py`
- Check that `models/cnn_cifar10.pth` exists

**Error: "Module not found: streamlit"**
- Install: `pip install streamlit`

**Port already in use**
- Streamlit will automatically use the next available port
- Or specify: `streamlit run app.py --server.port 8502`

## 💡 Tips

- Works best with images similar to CIFAR-10 classes
- Images are automatically resized to 32x32 pixels
- Clear, centered images give better results
- Try multiple images to see how the model performs!

## 🌐 Sharing the App

To share your app:
1. Deploy to Streamlit Cloud (free)
2. Or use other platforms like Heroku, AWS, etc.

---

**Enjoy testing your CNN model! 🎉**

