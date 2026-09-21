# Mars Terrain Semantic Segmentation

Pixel-level segmentation of 64×128 grayscale images of Martian terrain into five surface
classes, with a **residual U-Net trained entirely from scratch** — the assignment forbade
pre-trained weights.

*Team NMD — Nicolò Giallongo, Davide Goretti, Marco Fumagalli. Politecnico di Milano,
December 2024.*

## Result

> **Final MIoU score: 0.61**

Measured with a custom Mean Intersection over Union that **excludes the background class**. The
background dominates the dataset, so including it would inflate the score while saying little
about the terrain features that actually matter — soil, bedrock, sand and large rocks.

## What the problem actually was

Two difficulties shaped every design decision:

**Class imbalance.** The pixel distribution across the five classes is strongly skewed: classes
1 and 3 dominate, while class 4 (large rocks) is severely under-represented. A network trained
naively simply learns to predict the majority classes.

**Sparse labels.** The background occupies a large fraction of every image, so the loss has to
prioritise non-background pixels without being drowned by them.

Dataset cleaning came first: an outlier mask — *the alien Paul*, visually unrelated to the rest
of the data — was found during exploration. Every sample was compared against it and duplicate
instances were flagged and removed from both images and masks. Comparing the pixel distribution
before and after cleaning confirmed that the imbalance was **not** an artefact of the corrupted
samples, which moved the burden onto the loss function.

## Custom loss

Three complementary components, combined as

```
Total Loss = α · WeightedDice + β · Focal(γ) + SparseCategoricalCrossEntropy
```

- **Weighted Dice** maximises overlap between predicted and true masks directly, with
  class weights inversely proportional to pixel frequency
- **Focal loss** downweights well-classified pixels so learning concentrates on the hard ones,
  with γ controlling how sharply
- **Sparse categorical cross-entropy**, computed ignoring the background, as a baseline penalty

The weights α and β were chosen empirically, trading off class imbalance, sparse labels and
overall precision.

## Architecture, and what was discarded

The final model is a U-Net with residual connections and a **256-channel bottleneck**, ending in
a 1×1 convolution with one filter per terrain class and a softmax over the class dimension.

Getting there meant rejecting three alternatives that looked stronger on paper:

| Attempt | Outcome |
|---|---|
| Standard U-Net, 1024-channel bottleneck | underperformed |
| Double U-Net (more capacity) | **worse** than the standard one |
| Attention U-Net | did not meet expectations |
| Bottleneck reduced to 256 channels | significant improvement |
| Residual connections added | further improvement — better gradient flow and feature reuse |

The lesson is the one worth keeping: on a small dataset trained from scratch, added capacity
made things worse, and the gain came from constraining the model and improving gradient flow.

## Data augmentation

Horizontal and vertical flips, random rotations and translations. Geometric transformations
only — on grayscale imagery the model has to rely on texture and shape rather than colour, and
terrain features can appear at any orientation and position.

## Structure

```
HW2_MarsTerrainSegmentation/
├── MarsTerrain_Segmentation_UNet.ipynb   # full pipeline, with outputs and figures
└── Report_Challenge2.pdf                 # written report
```

## Report

[`Report_Challenge2.pdf`](Report_Challenge2.pdf) — full methodology, the pixel distribution
analysis, the loss derivation and the training curves.
