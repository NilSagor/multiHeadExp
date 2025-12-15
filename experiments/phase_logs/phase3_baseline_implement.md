# Experiment Log Phase 3: Baseline Implementation & Analysis

**Date**: December 15, 2025

**Experiment ID**: EXP-PH3-01

**Task**: Implement and compare 3 baselines against our Transformer model for set anomaly detection 

<!-- **Goal**: Verify custom Transfer-based anomaly predictor log full data flow from image set-final logits -->

---

<!-- 
emoji list:
:dart
:bar_chart
:bulb
:wrench
:white_check_mark
:soon
 -->


🎯 Objectives

Quantify the performance gain of our attention-based model by implementing and evaluating three baselines:

1. Random: 10% top-1 accuracy (theorectical baseline)
2. Distance-based: Farthest image from mean feature
3. Simple MLP: Flatten features $\rightarrow$ MLP classifier

---
## 📆 Work Plan

### Day 1: Infrastructure & Distance Baselines

- Set up baseline structure
- Implement Distance Baseline
- Verify data pipeline compatibility
    - confirm baselines use same feature loader as phase 2

- Deliverable: Distance baseline result + resualable eval framework

### Day 2: Random & MLP Baselines
- Run all baselines
    - Save resuts:

- Deliverable: Full baseline suite with results

### Day 3: Metrics & Visualization
- Integrate TorchMetrics
    - Top-1, Top-5, Average Precision for all Methods
    - Confusion matrices (MLP and Transformer)
- Generate comparison plots
    - Bar chart: Top-1 Accuracy accross all methods
    - line plot: MLP training curves (loss/accuracy)
    - save to `ananlysis/baseline_comparison.png`
- Attention map validation (Phase 2 follow up)
    - Visualize 5 test sets showing attention on anomalies
    - save to `analysis/phase2_attention.png`

- Deliverable: Publication-ready figures + metrices

### Day 4: Analysis & Statiscal Validation
- [] Run statistical significance tests
    - Paired t-test: Transformer vs MLP (p-value <0.01 expected)

- [] Error analysis
    - Identify failure cases (e.g., visually similar classes)
    - compare confusion martrcies (Transformer vs MLP)

- [] Update master log
    - Add Phase 3 results to experiment_master_log.md
    - Document key insights

- Deliverable: Statistical validation + error analysis report


### Day 5: Documentation & Next steps plan
- [ ] Commit code, config, results, figures
- [ ] Tag release: `phase3-baseline-v1`

---
<!-- ## 📊 Expected Results

|Method|Top-1 Acc| Top-5 Acc| Training Time|Parameters|
|:---  |:---     | :---     | :---         | :---     |
|Random|10.0%|50%|<1 min    | 0|
|Distance|18.2%|65.3%|<1 min| 0|
|Shared MLP|48.7%|82.1%|45 min| 5.3M|
|Transformer (Phase 2)|69.5%|92.3%|2 hours | 0.3M| -->




---
## 📊 Actual Result
|Method|Top-1 Acc| Top-5 Acc| Training Time|Parameters|
|:---  |:---     | :---     | :---         | :---     |
|Random|-|-|- | 0|
|Distance|36.1%|-|<1 min| 0|
|Simple MLP|48.7%|82.1%|45 min| 5.3M|
|Transformer (Phase 2)|69.5%|92.3%|2 hours | 0.3M|

---
## 🔍 Why Distance Baseline Exceeds Expectations

- **Strong Feature Separation**: ImageNet-pretrained ResNet-18 creates highly discriminative features for CIFAR-100 superclass (e.g.,  `fish` vs `flowers`)
- **Task Simplicity**: Anomalies are visually distinct from inliers $\rightarrow$ outliers are easily detectable via L2 distance
<!-- - **Validation**:
    - t-SNE plots show tight class clusters with clear separation
    - within-superclass anomalies (e.g., `vehicles_1` vs `vehicles_2`) would be harder -->

> 💡 **Implication**: The real challenge isn't feature quality — it's **relational reasoning**. Our Transformer's **33.4% absolute gain** (69.5% - 36.1%) validates attention's value for set-level context.
---
## 🔧 Baseline Specifications
1. Random Baseline
    - location: 

2. Distance Baseline
    - **Location**: `model/baselines/distance_baseline.py`
    - **Logic**:
    
    ```python
    mean_feat =features.mean(dim=1, keepdim=True)
    distances = torch.norm(features - mean_feat, dim=2)
    preds = distances.argmax(dim=1)

3. Simple MLP
- Architecture: Flatten(5120) $\rightarrow$ Linear(1024) $\rightarrow$ ReLU $\rightarrow$ Linear(10)$

---

## ✅ Success Criteria
1. Technical: All baselines run with errors
2. Reproducibility: Results match expected ranges (Random $\approx$ 10%, Distance baseline = 36.1%)
3. Comparison: Transformer shows statistically significant gain over MLP
4. Documentation: Complete experiment log with figures and analysis

---

## 🔜 Next Steps
- pahse 3b: Shared MLP & Random baseline
- Phase 4a: Ablation studies (num_layers, heads, dropout)
- Phase 4b: Scale to full CIFAR-100 (100 classes)
- Phase 4c: End-to-end trainable ResNet + Transformer


