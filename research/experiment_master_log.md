# Summary 

|Section| Phase 1| Phase 2|
|:--- |:--- |:--- |
|Feature Extractor| Under Model Architecture| Under Dataset Configuration|
|Model Input|Raw images [B, 10, 3, 224, 224]| Features `[B, 10, 512]`|
| Data Flow Step 1 | ResNet in model | ResNet in dataset|
---