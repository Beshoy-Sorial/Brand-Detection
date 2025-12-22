# Brand Detection System

**Digital Image Processing Project – Cairo University**

A comprehensive logo/brand detection system comparing classical computer vision techniques with modern deep learning approaches, featuring a Flutter mobile application for real-time brand verification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Supported Brands](#supported-brands)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the System](#running-the-system)
- [Evaluation & Testing](#evaluation--testing)
- [System Workflow](#system-workflow)
- [Results & Analysis](#results--analysis)

---

## 🎯 Overview

This project implements two distinct approaches for brand detection:

### 1. Classical Computer Vision Pipeline
- **SIFT** (Scale-Invariant Feature Transform) for feature extraction
- **Descriptor matching** using FLANN-based matcher
- **Homography verification** for geometric consistency
- **Rule-based logo verification** for final classification

### 2. Deep Learning Approach
- **Vision Transformer (ViT)** architecture
- **Pretrained model**: `Falconsai/brand_identification`
- Used for performance comparison and benchmarking

### Frontend
- **Flutter mobile application** for image capture and result visualization
- Real-time communication with backend via USB connection

---

## 🏷️ Supported Brands

The system currently detects three major sports brands:

- **Nike**
- **Adidas**
- **Puma**

### Output Format

```
✓ REAL → BRAND NAME (Nike/Adidas/Puma)
✗ FAKE → UNKNOWN BRAND
```

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Flutter App    │
│  (Mobile)       │
└────────┬────────┘
         │ USB Connection
         ▼
┌─────────────────┐
│  Python Backend │
│  (Flask Server) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│   CV   │ │   DL   │
│ SIFT   │ │  ViT   │
└────────┘ └────────┘
```

---

## ⚠️ Prerequisites

### Required Hardware & Software

| Component | Requirement | Purpose |
|-----------|-------------|---------|
| **Python** | 3.9 or higher | Backend processing |
| **Android Device** | Physical phone | Mobile app deployment |
| **USB Cable** | Type-C | Device-laptop connection |
| **Flutter SDK** | Latest stable | Mobile app development |
| **Connection** | USB **only** | Socket communication |

### Critical Requirements

> ⚠️ **IMPORTANT**: The system will **NOT** work over Wi-Fi. USB connection is mandatory for stable socket communication.

**Before Running:**
1. ✅ Backend server must be running
2. ✅ Mobile app must be installed
3. ✅ Phone must be connected via USB
4. ✅ Run app in **release mode** for optimal performance

---

## 📁 Project Structure

```
Brand-Detection/
├── src/                          # Python backend server
│   ├── main.py                   # Flask server entry point
│   ├── test_brand_detection.py   # Classical CV testing
│   └── run_brand_dnn.py          # Deep learning testing
│
├── Logos/                        # Reference brand logos
│   ├── nike.png
│   ├── adidas.png
│   └── puma.png
│
├── Experiment Results/           # Test dataset
│   ├── real/                     # Authentic brand images
│   │   ├── nike/
│   │   ├── adidas/
│   │   └── puma/
│   └── fake/                     # Non-brand/counterfeit images
│
└── frontend/                     # Flutter mobile application
    └── brand_detection/
        └── lib/                  # Dart source code
```

---

## 🔧 Installation & Setup

### 1. Backend Setup

```bash
# Navigate to backend directory
cd Brand-Detection/src

# Install dependencies
pip install -r requirements.txt

# Verify installation
python --version  # Should be 3.9+
```

### 2. Frontend Setup

```bash
# Navigate to Flutter project
cd Brand-Detection/frontend/brand_detection

# Get Flutter dependencies
flutter pub get

# Verify Flutter installation
flutter doctor
```

---

## 🚀 Running the System

### Step 1: Start the Backend Server

```bash
cd Brand-Detection/src
python main.py
```

**Expected Output:**
```
* Running on http://127.0.0.1:5555
* Server started successfully
```

### Step 2: Launch the Mobile App

```
install mobile app from github realese
```

**Why Release Mode?**
- ⚡ Better performance
- 🔄 Stable socket communication  
- 📸 Accurate real-time image transfer
- 🐛 Fewer debugging overheads

### Step 3: Connect and Test

1. Connect Android device via USB
2. Enable USB debugging on phone
3. Launch app and capture/select brand image
4. View real-time detection results

---

## 🧪 Evaluation & Testing

### Classical Computer Vision Evaluation

Test the SIFT-based approach on the experimental dataset:

```bash
cd Brand-Detection/src
python test_brand_detection.py
```

**What This Does:**
- Processes all images in `Experiment Results/`
- Applies SIFT feature extraction and matching
- Verifies homography for geometric consistency
- Generates accuracy metrics and detailed logs

**Evaluation Criteria:**

| Image Type | Correct Prediction |
|------------|-------------------|
| **Real Photos** | Output matches folder name (nike/adidas/puma) |
| **Fake Photos** | Output is NOT one of the target brands |

### Deep Learning Model Evaluation

Compare results with the pretrained Vision Transformer:

```bash
cd Brand-Detection/src
python run_brand_dnn.py
```

**What This Does:**
- Uses `Falconsai/brand_identification` model
- Classifies all test images
- Validates predictions against ground truth
- Produces comparative accuracy metrics

### Performance Metrics

Both approaches are evaluated using:

- ✅ **Accuracy** (overall correct predictions)
- 📊 **Precision & Recall** (per-brand performance)
- 🎯 **Success cases** (correctly classified images)
- ❌ **Failure cases** (misclassifications with analysis)
- 📝 **Detailed logging** (for debugging and improvement)

---

## 🔄 System Workflow

```
1. User captures/selects image in Flutter app
          ↓
2. Image sent to backend via USB socket
          ↓
3. Backend processes with CV/DL pipeline
          ↓
4. Brand classification performed
          ↓
5. Result returned to mobile app
          ↓
6. User views: REAL → BRAND or FAKE → UNKNOWN
```

---

## 📊 Results & Analysis

### Approach Comparison

| Aspect | Classical CV (SIFT) | Deep Learning (ViT) |
|--------|-------------------|-------------------|
| **Speed** | Fast (milliseconds) | Moderate (model inference) |
| **Accuracy** | High for clear logos | Very high overall |
| **Robustness** | Sensitive to occlusion | Handles variations well |
| **Resource** | Low computational cost | Requires more memory/GPU |
| **Interpretability** | High (feature matching) | Low (black box) |

### Strengths & Weaknesses

**Classical CV Strengths:**
- ⚡ Fast inference
- 🔍 Explainable feature matching
- 💾 Low resource requirements

**Classical CV Weaknesses:**
- ❌ Struggles with heavy rotation/distortion
- 🌫️ Poor performance in low contrast
- 📐 Requires good logo visibility

**Deep Learning Strengths:**
- 🎯 High accuracy on diverse images
- 🔄 Handles variations robustly
- 📸 Works with partial logos

**Deep Learning Weaknesses:**
- ⏱️ Slower inference
- 🖥️ Requires more computational resources
- 🔒 Less interpretable decisions

---

## 🎓 Conclusion

This project successfully demonstrates:

- **End-to-end brand detection system** from image capture to classification
- **Comparative analysis** of traditional vs. modern computer vision techniques
- **Real-world mobile application** with practical USB-based communication
- **Comprehensive evaluation framework** for both approaches

The system highlights that while classical computer vision techniques offer speed and interpretability, deep learning models provide superior accuracy and robustness in varied real-world conditions.

---



