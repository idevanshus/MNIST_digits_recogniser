# Training Guide

This document provides a technical walkthrough of the model training process for the NeuralDigit project.

## 1. Dataset & Preprocessing

We use the standard MNIST dataset provided by `torchvision.datasets`.

### Data Augmentation
To make the model robust to hand-drawn input in the browser, we apply several affine transformations during training:
- **Rotation**: Up to ±15 degrees.
- **Translation**: Up to 10% in both horizontal and vertical directions.
- **Scaling**: Between 90% and 110%.
- **Normalization**: Standard MNIST mean (0.1307) and standard deviation (0.3081).

## 2. Training Hyper-parameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Better weight decay handling than Adam. |
| **Learning Rate**| 1e-3 (max 1e-2) | Managed by OneCycleLR scheduler. |
| **Weight Decay** | 1e-2 | Regularization to prevent overfitting. |
| **Batch Size** | 128 | Balanced for performance and gradient stability. |
| **Epochs** | 15 | Sufficient for >99% accuracy with modern scheduler. |

## 3. Advanced Techniques

### OneCycle Learning Rate Scheduler
We use the **OneCycleLR** strategy, which starts with a low LR, ramps up to the maximum, and then decays to near zero. This "Super-Convergence" allows for faster training and better generalization.

### Label Smoothing
Instead of using hard 1.0/0.0 targets, we use `label_smoothing=0.1`. This prevents the model from becoming too confident in its predictions, making it more robust to noisy or ambiguous inputs (common in hand-drawing).

## 4. Run Training

Ensure you have the dependencies installed:
```bash
pip install torch torchvision
```

Execute the training script:
```bash
python train.py
```

The script will automatically detect and use **Apple Silicon (MPS)**, **CUDA**, or **CPU**. The best model is saved as `mnist_cnn.pt`.

## 5. Performance Metrics

Typical results after 15 epochs:
- **Test Accuracy**: 99.2% - 99.4%
- **Average Loss**: ~0.024
- **Convergence Time**: ~2-5 minutes on modern hardware.
