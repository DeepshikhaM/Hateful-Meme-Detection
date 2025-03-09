# Hateful-Meme-Detection
A multimodal machine learning approach using VisualBERT, RoBERTa, and ViLBERT to detect hate speech in memes by integrating textual and visual information.

# 🖼️ Hateful Meme Detection

## 📌 Project Overview
This project focuses on detecting hateful memes by leveraging **multimodal machine learning models** that integrate both **text and image analysis**. The research was conducted using state-of-the-art transformer-based models like **VisualBERT, RoBERTa, and ViLBERT** to enhance the detection of hate speech in memes. Given the complex nature of hateful memes, which blend offensive text with neutral or unrelated images, our approach provides a more **nuanced** and **context-aware detection system**.

---

## 🎯 Problem Statement
Traditional **text-based hate speech detection** fails when offensive content is embedded within **memes**—images with text overlays that introduce **sarcasm, irony, or hidden meanings**. The challenge is to develop a robust **multimodal classification model** capable of identifying harmful memes by analyzing both their **linguistic and visual features**.

---

## 🏗️ Methodology
### 1️⃣ **Data Preprocessing**
- Used **Facebook Hateful Memes Challenge** dataset containing **memes labeled as hateful or non-hateful**.
- **Text Preprocessing:** Tokenization, padding, and encoding using transformers.
- **Image Preprocessing:** Standardization (resizing, normalization).
- **Data Augmentation:** Applied transformations (flipping, rotation) to enhance generalization.

### 2️⃣ **Model Architecture**
- Extracted **textual features** using **RoBERTa and BERT variants**.
- Extracted **image features** using **CNN-based architectures**.
- Used **multimodal fusion techniques** to combine textual and visual representations.
- **Final Classifier:** A **BERT-based model** that jointly processes both **textual and visual information**.

### 3️⃣ **Training & Evaluation**
- **Supervised Learning Approach** with fine-tuning.
- Evaluated using **Accuracy, Precision, Recall, and F1-Score**.
- **Comparison of models:**
  - **VisualBERT** achieved the **highest accuracy (78%)** compared to RoBERTa and ViLBERT.
  - Improvements from previous iterations were significant in **precision and recall metrics**.

---

## 📊 Key Findings
- **VisualBERT** outperforms both **RoBERTa** and **ViLBERT** for multimodal hate meme classification.
- The model successfully captures **subtle visual and textual cues** that distinguish hateful from non-hateful memes.
- Results emphasize the need for **multimodal hate speech detection** rather than relying solely on **text-based models**.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn** (Data Processing & Visualization)
- **TensorFlow, PyTorch** (Deep Learning Frameworks)
- **Transformers (Hugging Face)** (Model Implementation)
- **OpenCV** (Image Preprocessing)
- **Facebook Hateful Memes Dataset** (Benchmark Dataset)

---
##  Future Work
- **Enhancing Model Robustness**: Explore **CLIP** and **GPT-4V** for better multimodal understanding.
- **Bias Mitigation**: Address dataset biases to prevent **false positives/negatives**.
- **Deployment**: Convert the model into a **real-time API or web app** for hate meme moderation.

---

## ✨ Contributors
- **Deepshikha Mahato**
- **Shilpa Kuppili**
- **Pinal Gajjar**

