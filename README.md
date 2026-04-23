# 🧠 Deep Reflection Removal using Multi-Stage Deep Learning

A complete deep learning pipeline for removing reflections from images using a multi-stage training strategy with refraction alignment, encoder-decoder architecture, and perceptual optimization.

---

## 📌 Overview

Reflection removal is a challenging problem in digital image processing due to the complex interaction between reflection and transmission layers.

This project proposes a **multi-stage deep learning framework** that progressively improves image quality through:

* Synthetic training
* Real-world fine-tuning
* Perceptual + edge-based optimization

---

## 🚀 Key Features

✅ Multi-stage training pipeline
✅ Refraction-based alignment module
✅ ResNet + U-Net inspired architecture
✅ Perceptual + Edge loss optimization
✅ Quantitative evaluation (PSNR, SSIM)
✅ Streamlit web application

---

## 🏗️ Model Architecture

```
Input Image (I)
      ↓
Refraction Module (Alignment)
      ↓
Aligned Image
      ↓
ResNet Encoder
      ↓
Decoder with Skip Connections
      ↓
Output Image (T̂)
```

---

## 🔄 Training Pipeline

```
Stage 1:
Synthetic Dataset → Model Training → Intermediate Model

Stage 2:
Real Dataset → Fine-Tuning → Improved Model

Stage 3:
Perceptual + Edge Loss → Final Model
```

---

## 📂 Dataset

The model is trained on multiple datasets:

### 📌 Synthetic Dataset

* Input: Blended images
* Ground Truth: Clean transmission images

### 📌 Real Dataset (Real20)

* Real-world reflection images

### 📌 SIR2 Dataset

* Additional real-world training samples

### 📊 Total Data

* Synthetic: ~13,700 images
* Real: ~1,018 images

---

## ⚙️ Implementation Details

| Component      | Value                          |
| -------------- | ------------------------------ |
| Framework      | PyTorch                        |
| Language       | Python                         |
| GPU            | NVIDIA Tesla T4 (Google Colab) |
| Image Size     | 256 × 256                      |
| Optimizer      | Adam                           |
| Learning Rate  | 1e-4                           |
| Loss Functions | L1 + Perceptual + Edge         |
| Epochs         | 5 + 2 + 2                      |

---

## 📊 Results

### 🔢 Quantitative Results

| Model      | PSNR      | SSIM      |
| ---------- | --------- | --------- |
| Baseline   | 13.30     | 0.479     |
| Refraction | 13.38     | 0.484     |
| Stage 1    | 13.08     | 0.484     |
| Final V1   | 20.03     | 0.573     |
| Final V2   | 20.74     | 0.608     |
| Final V3   | **20.81** | **0.618** |

---

### 📈 Improvements

* Significant PSNR improvement (~+7 dB)
* Strong SSIM improvement (better structural quality)
* Improved edge preservation and visual clarity

---

## 🖼️ Sample Results

| Input                      | Output                       |
| -------------------------- | ---------------------------- |
| ![input](assets/input.png) | ![output](assets/output.png) |

---

## 📉 Evaluation Metrics

### 🔹 PSNR

Measures pixel-level reconstruction quality between predicted and ground truth images.

### 🔹 SSIM

Measures structural similarity, focusing on edges, textures, and perceptual quality.

---

## ⚠️ Limitations

* Struggles with very strong reflections
* Performance depends on dataset diversity
* Slight blur in fine textures

---

## 🔮 Future Work

* Use Transformer-based architectures
* Improve dataset diversity
* Real-time video reflection removal
* Deploy on mobile/edge devices

---

## 💻 Streamlit Web App

Run the app locally:

```bash
streamlit run app.py
```

### Features:

* Upload image
* Reflection removal
* Output visualization
* PSNR & SSIM display

---

## 📁 Project Structure

```
Reflection_Removal/
│
├── models/
│   ├── baseline.pth
│   ├── final_model_v3.pth
│
├── datasets/
│
├── app.py
├── train.ipynb
├── evaluate.ipynb
├── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/Deep-Reflection-Removal.git
cd Deep-Reflection-Removal
pip install -r requirements.txt
```

---

## 📜 Research Paper

This project includes a full academic research implementation covering:

* Literature Review
* Methodology
* Multi-stage Training Strategy
* Experimental Analysis

---

## 👨‍💻 Author

**Hamza Haroon**
AI & Deep Learning Enthusiast

---

## ⭐ Acknowledgment

Datasets and inspiration from:

* SIR2 Dataset
* Real20 Dataset
* Deep Image Processing Research Community

---

## 📌 License

This project is intended for academic and research purposes only.

