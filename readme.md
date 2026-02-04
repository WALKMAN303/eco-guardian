# 🌍 Eco Guardian - AI-Powered Waste Classification System

An intelligent waste detection system using YOLOv8 to classify recyclable materials in real-time, promoting environmental sustainability.

## 🎯 Project Overview

Eco Guardian uses computer vision and deep learning to automatically detect and classify waste into 5 main categories: Plastic, Metal, Glass, Paper, and Organic. This system can be deployed in smart bins, recycling facilities, or as an educational tool to promote proper waste segregation.

## 📊 Model Performance

- **Dataset**: Hybrid Dataset (TACO + Custom Images)
  - TACO Dataset: 3,147 images
  - Custom Images: 15 personal waste photos
  - Total: 3,162 training images
- **Model**: YOLOv8n (Nano)
- **Classes**: 5 (simplified from 59 original classes)
- **Overall mAP@50**: 20.3%
- **Best Performance**: Plastic (38.2% mAP@50)
- **Inference Speed**: 4.0ms per image (~250 FPS)

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|---------|-----------|
| Plastic | 38.5% | 49.8% | **38.2%** | 27.2% |
| Metal | 37.1% | 33.0% | 26.1% | 19.9% |
| Paper | 26.4% | 31.2% | 21.2% | 16.1% |
| Glass | 21.8% | 6.1% | 7.5% | 5.5% |
| Organic | 19.8% | 16.4% | 8.3% | 4.7% |

## 🛠️ Technology Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Web Interface**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Language**: Python 3.8+

## 📁 Project Structure

```
waste-detection-app/
├── models/
│   └── best.pt              # Trained YOLOv8 model
├── app/
│   ├── app.py     # Web interface
│   ├── webcam.py   # Real-time webcam detection
├── results/
│   ├── results.png          # Training plots
│   └── confusion_matrix.png # Confusion matrix
├── sample_images/           # Test images
├── requirements.txt
├── waste_detection.py       #main file
├── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eco-guardian.git
cd eco-guardian
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your trained model

Place your `best.pt` model file in the `models/` directory.

## 💻 Usage

### Web Interface (Streamlit)

```bash
cd app
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

### Real-time Webcam Detection

```bash
cd app
python detect_realtime.py
```

**Controls:**
- Press `Q` to quit
- Press `S` to save a screenshot

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

## 🎨 Features

- ✅ **Image Upload Detection**: Upload photos for waste classification
- ✅ **Real-time Webcam Detection**: Live waste detection via webcam
- ✅ **Fast Inference**: 250 FPS detection speed
- ✅ **Multi-class Detection**: Classify 5 waste categories

## 📈 Training Process

### Dataset Preparation

1. Combined TACO dataset (3,147 images) with custom waste photos (including 15 personal photos)
2. Total dataset: 3,162 images across 59 original classes
3. Identified data scarcity issue (~79 images per class)
4. Strategically merged 59 classes into 5 main recyclable categories
5. Applied data augmentation in Roboflow to increase diversity

### Model Training

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='taco_5classes/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50
)
```

### Results

- Training completed successfully for 100 epochs
- Best model performance achieved throughout training
- Final mAP@50: 20.3%
- Plastic detection achieved 38.2% mAP@50 (best performing class)
- Fast inference speed: 4.0ms per image (~250 FPS)

## 🎯 Use Cases

1. **Smart Bins**: Automated waste sorting in public spaces
2. **Recycling Facilities**: Pre-sorting waste for processing
3. **Educational Tool**: Teaching proper waste segregation
4. **Environmental Monitoring**: Track waste composition
5. **Mobile App**: On-device waste classification

## 🔮 Future Enhancements

- [ ] Improve model accuracy (target: 40%+ mAP@50)
- [ ] Integrate with IoT smart bins
- [ ] Multi-language support
- [ ] Add waste volume estimation
- [ ] Deploy as REST API
- [ ] Expand to more waste categories

## 📊 Model Details

**Why YOLOv8n?**
- Fast inference speed (250 FPS)
- Good balance of accuracy and performance
- Suitable for edge deployment
- Low computational requirements

**Class Simplification Strategy:**
- Original: 59 fine-grained classes
- Final: 5 main recyclable categories
- Reason: Increased samples per class from ~53 to ~630
- Result: 50x improvement from initial 0.4% to 20.3% mAP

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TACO Dataset**: [Trash Annotations in Context](http://tacodataset.org/)
- **Ultralytics**: YOLOv8 framework
- **Roboflow**: Dataset management and preprocessing

## 📧 Contact

Your Name - [arjunsreechakram@gmail.com](mailto:arjunsreechakram@gmail.com)

Project Link: [https://github.com/WALKMAN303/waste-detection](https://github.com/WALKMAN303/waste-detection)
