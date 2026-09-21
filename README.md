# Artificial Neural Network and Deep Learning — Course Projects

Two computer-vision projects developed for the **Artificial Neural Network and Deep Learning**
course at Politecnico di Milano, in TensorFlow/Keras. One built on transfer learning, one
trained entirely from scratch.

**Team NNMD** — Nicolò Giallongo, Davide Goretti, Marco Fumagalli.
Both projects were developed jointly by the three of us.

## [1 — Blood Cell Classification](HW1_BloodCellClassification/)

Multi-class classification of blood cell images into 8 categories, using a fine-tuned
EfficientNetB3 with an adaptive augmentation strategy to counter class imbalance.

| | Test set | Augmented test set |
|---|---|---|
| Accuracy | **0.9152** | 0.1740 |
| Loss | 0.2593 | 2.2760 |

The gap between the two is the honest headline of this project: the model reaches 91.5% on the
held-out test set but collapses to 17.4% — barely above the 12.5% chance level for 8 classes —
on the same images after augmentation. It is a sharp demonstration of how brittle a fine-tuned
classifier can be to a shift in the input distribution.

## [2 — Mars Terrain Semantic Segmentation](HW2_MarsTerrainSegmentation/)

Pixel-level segmentation of 64×128 grayscale Martian terrain images into 5 surface classes,
with a residual U-Net built from scratch — no pre-trained weights allowed.

> **Final MIoU score: 0.61**, with a custom metric that excludes the dominant background class

Reaching it took discarding two architectures that looked more promising on paper: a Double
U-Net and an attention U-Net both underperformed, while reducing the bottleneck from 1024 to
256 channels and adding residual connections is what actually moved the score.

## Structure

```text
AN2DL-Polimi-Projects/
├── HW1_BloodCellClassification/   # EfficientNetB3, transfer learning, adaptive augmentation
└── HW2_MarsTerrainSegmentation/   # residual U-Net from scratch, custom composite loss
```

Each folder contains the notebook and the full written report.

## Tools

TensorFlow/Keras · NumPy · Matplotlib

---

*Politecnico di Milano, academic year 2024/2025.*
