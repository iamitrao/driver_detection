# Driver Detection (Safe vs Unsafe) 🚗

## What this project is about
This project is a **driver safety detection system** where the model predicts whether the driver is in a **Safe** or **Unsafe** condition based on face and eye behavior.

The main idea is to detect unsafe driving conditions like **sleepiness, yawning, distraction, or gaze not focused on the road**, and then give a **warning/alarm**.

Instead of directly training on images, I first extract important facial features using **MediaPipe Face Mesh + Iris tracking**, convert them into numeric values, store them in a CSV file, and then train a **Random Forest model** on that dataset.

---

## Folder Structure
driver_detection/
│
├── data/
│ ├── safe/
│ └── unsafe/
|--- csv file
|--- model.pkl
├── image_to_numeric_data.ipynb
├── model_training.ipynb
└── model_testing.ipynb



### `data/safe/`
Contains images where the driver is driving normally (eyes open, normal gaze).

### `data/unsafe/`
Contains images where the driver is distracted, sleepy, yawning, or unsafe.

---

## How the project works

### 1. Feature Extraction (Image → Numeric Data)
Notebook: **image_to_numeric_data.ipynb**

This notebook reads images from both folders (safe/unsafe) and extracts facial and eye-related features using:
- MediaPipe Face Mesh
- MediaPipe Iris landmarks

Some extracted features include:
- eye status (open/close)
- gaze direction
- head orientation
- yawning related features

All extracted features are stored into a **CSV file**, which becomes the dataset for training.

---

### 2. Model Training
Notebook: **model_training.ipynb**

In this notebook:
- the generated CSV file is loaded
- preprocessing is done
- dataset is split into train and test
- Random Forest model is trained
- model performance is checked using accuracy and other metrics

Random Forest is used because it performs well on structured numeric features and gives stable results.

---

### 3. Model Testing + Warning/Alarm Logic
Notebook: **model_testing.ipynb**

This notebook is used for final testing.

Along with prediction, I added a simple threshold-based system:
- if unsafe prediction happens for a short time → **Warning**
- if unsafe prediction continues continuously → **Alarm**

This makes the project more realistic for real-world driver monitoring.

---

## Technologies Used
- Python
- OpenCV
- MediaPipe (Face Mesh + Iris Tracking)
- Pandas, NumPy
- Scikit-learn
- Random Forest Classifier
- Jupyter Notebook

---

## How to run this project

### Install required libraries
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib

