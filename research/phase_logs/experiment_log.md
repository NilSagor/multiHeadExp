# Phase 1 Experiment: Set Anomaly Detection on CIFAR100

## Hypothesis 
**Statement:** using a MultiHeadAttention-based architecture on pre-extracted ResNet features can achieve > 25% top-1 accuracy in identifying the single anomalous image within a set of 10 CIFAR100 images (9 same-class, 1 different class).

**Rationale:** Attention mechanisms should learn to compare image features and identify outliers.

## Dataset & Sampling Strategy 

## 2. Success Criteria
- **Primary:** Top-1 accuracy > 30% (3x random chance)
- **Secondary:** Top-5 accuracy > 70%
- **Technical:** Training converges within 50 epochs without NaN

## 3. Dataset Configuration
- **Source:** CIFAR100 from torchvision
- **Classes used:** First 20 classes (for Phase 1)
- **Set composition:** 10 images = 9 (class A) + 1 (class B ≠ A)
- **Dataset splits:**
  - Training: 10,000 sets
  - Validation: 2,000 sets  
  - Test: 2,000 sets
- **Image size:** 32x32 → resized to 224x224 for ResNet

### Core consideration
- set size: 10 images total
- composition: 9 from class A, 1 from class B (A not equal B)
- Number of sets: 10,000 for training, 2,000 for validaiton, 2, 000 for test
- Image size: 32 x 32 RGB (native CIFAR100)

### for phase 1 simplicity
- Choose 20 classes from CIFAR100 (out of 100)
- Ensure they're diverse (not visually similar)
- Track which classes to exlude in phase 2

### Sampling 
 - randomly select anchor class c1
 - randomly select anomaly class (c2 $\neg$ c1)
 - sample 9 images from c1
 - sample 1 image from c2
 - Shuffle oreder to remove positional bias
 - Record ground truth index (0-9) 



 ## 4. Model Architecture
**Feature Extractor:**
- ResNet18 (pretrained on ImageNet)
- Frozen weights (no gradient updates)
- Output: 512-dim feature vectors

**Attention Block:**
- MultiHeadAttention (4 heads, 128 dim each)
- Dropout: 0.1
- LayerNorm after residual

**Classification Head:**
- Shared MLP across positions
- Input: 512 → 256 → 1 (anomaly score per position)

## 5. Training Configuration
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Batch size:** 32 sets = 320 images
- **Loss:** CrossEntropyLoss (10-class classification)
- **Epochs:** 50 max, early stopping patience: 10
- **Seed:** 42

## 6. Baselines for Comparison
1. **Random:** 10% top-1, 50% top-5
2. **Distance-based:** Find farthest from mean feature
3. **Simple MLP:** Flatten all features → MLP

## 7. Risk Assessment
**Failure modes:**
1. Accuracy < 10% → Check data sampling
2. Overfitting in first 5 epochs → Add more dropout
3. No learning → Check gradient flow, feature extraction

**Debug plan:**
- Visualize attention weights
- Plot feature distributions
- Check set composition

## 8. Expected Timeline
- Data pipeline: 2 hours
- Model implementation: 3 hours  
- Initial training: 1 hour (10 epochs)
- Debugging: 4 hours
- Final training: 2 hours
- Analysis: 2 hours
- **Total:** ~14 hours

## 9. Phase 2 Considerations (if Phase 1 succeeds)
- Scale to all 100 CIFAR classes
- Add positional encoding
- Train feature extractor end-to-end
- Experiment with different attention mechanisms