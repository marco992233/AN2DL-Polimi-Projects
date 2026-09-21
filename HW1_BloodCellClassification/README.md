# Blood Cell Image Classification

Multi-class classification of blood cell images into eight categories, using transfer learning
on a fine-tuned **EfficientNetB3**.

*Team NNMD — Nicolò Giallongo, Davide Goretti, Marco Fumagalli. Politecnico di Milano,
November 2024.*

## Results

| | Test set | Augmented test set |
|---|---|---|
| Accuracy | **0.9152** | 0.1740 |
| Loss | 0.2593 | 2.2760 |

91.5% on the held-out test set. On the augmented test set — the same images after a uniform
augmentation pass — accuracy falls to 17.4%, barely above the 12.5% chance level for eight
classes.

We report both because the gap is the most informative result of the project. A 74-point drop on
transformed versions of the same data shows how strongly a fine-tuned classifier can depend on
the input distribution it was trained against, and it is a limit worth knowing before trusting
such a model on data collected under different conditions.

## Approach

**Dataset cleaning.** Two contaminating images, unrelated to the task, were identified during
visual inspection and removed.

**Threshold-based splitting.** Rather than a uniform split, each class was divided by a
population threshold: over-represented classes contributed a smaller fraction to the validation
set. This deliberately *reduced* imbalance in validation while *increasing* it in training —
the training-side imbalance was then handled by augmentation. The validation set was further
split into an effective validation set, a standard test set and an augmented test set, so the
effect of augmentation on accuracy could be measured directly.

**Adaptive data augmentation.** The same augmentations were applied across all classes, but the
augmentation *percentage* varied with class representation: under-represented classes were
augmented more heavily. This targeted the training set specifically, keeping validation and test
sets dominated by original images. The augmented test set instead used a uniform augmentation
percentage, preserving the original class distribution.

## Architecture

Input 96×96 RGB → **EfficientNetB3** (ImageNet weights, top excluded, initially frozen) →
Global Average Pooling → Dense 512 → BatchNorm + ReLU → Dropout → Dense 256 → BatchNorm + ReLU →
Dropout → Dense 8 + softmax.

EfficientNetB3 was chosen for its accuracy-per-parameter ratio, which is favourable against
VGG16 and ResNet50 on datasets of this size. Global Average Pooling was preferred over flattening
to keep the parameter count low and reduce overfitting.

## Training

Training proceeded in stages rather than as a single fit:

1. Unconstrained initial training to let the network establish basic representations. Adam and
   RMSProp were both tried; **SGD** gave the best convergence and final performance.
2. Dropout rate, learning rate and momentum adjusted dynamically based on the training and
   validation curves.
3. `ReduceLROnPlateau` to lower the learning rate when validation performance stalled.
4. Augmentation increased when validation accuracy plateaued, while checking that accuracy on
   the non-augmented test set did not degrade.
5. EfficientNet layers progressively unfrozen for fine-tuning, then the process repeated.

## Structure

```
HW1_BloodCellClassification/
├── BloodCell_Classification_EfficientNet.ipynb   # full pipeline, with outputs and figures
└── Report_challenge1.pdf                         # written report
```

The notebook keeps its execution outputs, so the training curves and the confusion matrix are
visible on GitHub without running anything.

## Report

[`Report_challenge1.pdf`](Report_challenge1.pdf) — full methodology, the accuracy and loss
curves, and the confusion matrix.
