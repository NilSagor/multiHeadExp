import torch 
import numpy as np


from run_single import load_config

from data import prepare_data

config_path = "config.yaml"
config = load_config(config_path)

def distance_baseline(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            mean_feat = features.mean(dim=1, keepdim=True)
            distances = torch.norm(features, - mean_feat, dim=2)
            preds = distances.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total

_, _, test_loader = prepare_data(config)
acc = distance_baseline(test_loader)
print(f"Distance baseline accuracy:{acc:.4f}")