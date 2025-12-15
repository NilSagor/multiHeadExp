# Experiment Master Log: Set Anomaly Detection

**Project Goal**: Detect anomalous images in sets using attention mechanisms on CIFAR100.

---

<!-- :white_check_mark
:x
:wench
:test_tube
:test_tube 
 -->


## 📁 Experiment Timeline

### Phase 1: Baseline & Pipeline Validation ()
- **Objective**: Validate data pipeline and training loop with dummy model
- **Key Outcome**:
    - ✅ Data sampling works (9 inliers + 1 anomaly, shuffled)
    - ✅ Training loop functional
    - ❌ Dummy model achieves 10% (as expected)
- **Logs**: [`experiments/phase1/experiment_log.md`](experiments/phase1/experiment_log.md)
- **Lessons**:
    > "Need proper feature extractor before phase 2."

---

### Phase 2: Model Implementation & Validate ()
- **Objectives**: Implement and validate custom Transformer-based anomaly predictor.
- **Hypothesis**:
    > "Frozen ResNet + attention achieves >30% top-1 accuracy."
- **Architecture**:
    - ResNet18 (frozen) -> Linear(512->256) -> 4 layer Transformer -> per position classifier
    - Scaled dot-product attention, no positional encoding
<!-- - **Status**: &#1F527; In Progress -->
- **Status**: 🔧 In Progress
- **Log**: []
- **Key Decisions**:
    - Moved ResNet feature extractor into `data.py` (not model)
    - Fixed dimension flow: `512->256->1`
    - Disable masking (fixed-size sets)


## Results Summary

## Key Insights & Lessons

### Iteration Insights
> **"Feature extraction must be part of the data pipeline, not the model."**

> Moving ResNet into `data.py` simplified the model and ensured consistent preprocessing.

> **"Dimension mismatches are the #1 cause of silent failures"**

> Always validate tensor shapes at each major stage

### Failure Analysis
> **"Hardcoding hyperparameters in model init causes config drift."**

> Always pass all args from config.

## Next Steps
1. Complete Phase 2 training (50 epochs)
2. Run baselines (distance-based, MLP)
3. Start Phase 3: Full CIFAR100 scaling

## Related Artifacts
- **Code**: 
- **Configs**:
- **Detailed Logs**:
- **Results**: 




# update

|Section| Phase 1| Phase 2|
|:--- |:--- |:--- |
|Feature Extractor| Under Model Architecture| Under Dataset Configuration|
|Model Input|Raw images [B, 10, 3, 224, 224]| Features `[B, 10, 512]`|
| Data Flow Step 1 | ResNet in model | ResNet in dataset|
---

