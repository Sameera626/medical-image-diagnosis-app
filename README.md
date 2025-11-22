#   Medical Images Diagnosis APP 

Doctors can upload a chest X-ray image and the system automatically classifies it as Normal or Diseased. The app also produces Grad-CAM visualizations to highlight the regions that influenced the prediction, helping clinicians interpret the model's decision.

---

## Features

- Upload and preprocess chest X-ray images using the TensorFlow Keras API.
- Build a classification model based on EfficientNetB0 with additional dense layers for improved feature learning.
- Train the model on labeled training data and optimize using the Adam optimizer.
- Evaluate model performance using accuracy metrics, a classification report, and a confusion matrix.
- Generate Grad-CAM heatmaps to visualize the regions most relevant to each prediction.
- Simple interface for clinicians to upload images, view predictions, and inspect explanation heatmaps.

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Sameera626/medical-image-diagnosis-app.git
cd medical-image-diagnosis-app
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

##  Running the Project

### Start streamlit app
```bash
cd app
python -m streamlit run streamlit_app.py
```

---

##  Tech Stack

- **Python**: Programming language used for the application.
- **TensorFlow/Keras**: Deep learning framework for building and training the image classification model.
- **Streamlit**: Framework for creating the web-based user interface.
- **NumPy**: Library for numerical computations and array operations.
- **Scikit-learn**: Library for evaluation metrics.
- **OpenCV**: Library for image preprocessing.

---

##  License

MIT License © 2025 [Sameera Athukorala]

---


