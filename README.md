# Artificial Neural Network and Deep Learning Projects 🧠

This repository contains the projects developed for the **Artificial Neural Network and Deep Learning (AN2DL)** course at Politecnico di Milano. 
The repository highlights practical experience in building, training, and optimizing deep learning models for Computer Vision tasks using **TensorFlow/Keras**.

---

## 🔬 Project 1: Blood Cell Image Classification
📁 **Folder:** `HW1_BloodCellClassification`

**Task:** Multi-class image classification of blood cells into 8 distinct categories[cite: 2].

In this project, we tackled a complex image classification problem using **Transfer Learning**[cite: 2]. Due to a slight class imbalance, we implemented a custom augmentation strategy to prevent the model from overfitting on dominant classes[cite: 2].

**Key Skills & Techniques:**
* **Model Architecture:** Fine-tuned a pre-trained **EfficientNetB3** model[cite: 2].
* **Optimization:** Replaced the top layers with Global Average Pooling (GAP), Dense layers, Batch Normalization, and Dropout layers for robust feature extraction and regularization[cite: 2].
* **Data Augmentation:** Implemented an **Adaptive Data Augmentation** strategy, varying the augmentation percentage based on class representation to successfully mitigate class imbalance[cite: 2].
* **Dataset Cleaning:** Identified and removed contaminated/outlier images to ensure dataset integrity[cite: 2].

---

## 🪐 Project 2: Mars Terrain Semantic Segmentation
📁 **Folder:** `HW2_MarsTerrainSegmentation`

**Task:** Pixel-level semantic segmentation of $64\\times128$ grayscale images of Martian terrain into 5 surface classes[cite: 4].

This project required building a segmentation model completely from scratch, without the use of pre-trained weights[cite: 4]. The primary challenges were severe class imbalance (e.g., very few "large rock" pixels) and sparse labels heavily dominated by the background[cite: 4].

**Key Skills & Techniques:**
* **Model Architecture:** Designed and trained a custom **Residual U-Net** from scratch, integrating residual connections to improve gradient flow and feature reuse[cite: 4].
* **Custom Loss Function:** Developed a specialized loss function combining **Weighted Dice Loss**, **Focal Loss**, and **Sparse Categorical Cross-Entropy** to force the network to focus on hard-to-classify and underrepresented pixels[cite: 4, 5].
* **Custom Metrics:** Implemented a tailored **Mean Intersection Over Union (MIoU)** metric that ignores the background class, ensuring a rigorous and reliable evaluation of the actual terrain features[cite: 4, 5].
* **Data Augmentation:** Applied geometric augmentations (flips, rotations, translations) suited for grayscale texture-based classification[cite: 4, 5].
