# Aircraft Classification Experiments

## Experiment 1: Baseline ResNet18 (Transfer Learning)
**Date:** 2026-02-01
**Goal:** Establish baseline with pretrained ResNet18

### Configuration
- Model: ResNet18 (pretrained, frozen except FC)
- Learning Rate: 1e-4
- Epochs: 5
- Batch Size: 32
- Optimizer: Adam
- Data Augmentation: RandomCrop, HFlip, Rotation, ColorJitter

### Results
- Train Accuracy: 0.02%
- Val Accuracy: 0.01%
- Best Epoch: 3
- Training Time: 14 minutes

### Observations
- Accuracy was very low
- Model barely learning (frozen layers issue)

### Next Steps
- Try training custom CNN from scratch
- Increase epochs to 25-30

---

## Experiment 2: DeepCNN from Scratch
**Date:** 2026-02-01
**Goal:** Train custom CNN architecture from scratch

### Configuration
- Model: DeepCNN (4 conv blocks)
- Learning Rate: 1e-3
- Epochs: 25
- Batch Size: 32
- Optimizer: Adam

### Code Changes
```python

### Results
- Train Accuracy: 
- Val Accuracy: 
- Best Epoch: 
- Training Time:  minutes

### Observations


### Next Steps
