import torch 
import numpy as np
# from data import prepare_data

# from run_single import load_config


# config_path = "config.yaml"
# config = load_config(config_path)

def distance_baseline(features, labels):    
    with torch.no_grad():       
        mean_feat = features.mean(dim=1, keepdim=True) # [B, 1, 512]
        distances = torch.norm(features - mean_feat, dim=2) # [B, 10]
        preds = distances.argmax(dim=-1) # [B]
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct/total

# _, _, test_loader = prepare_data(config)
# acc = distance_baseline(test_loader)
# print(f"Distance baseline accuracy:{acc:.4f}")