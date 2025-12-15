import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


import yaml

from model.baselines.distance_baseline import distance_baseline
from data import prepare_data

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config_path = "config.yaml"
    config = load_config(config_path)
    _, _, test_loader = prepare_data(config)

    for features, labels in test_loader:
        print("feature shape: ", features.shape) # [B, 10, 512]
        print("Labels range: ", labels.min(), labels.max())
        break
    
    total_correct = 0
    total_samples = 0
    for features, labels in test_loader:
        # print("feature shape: ", features.shape) # [B, 10, 512]
        # print("Labels range: ", labels.min(), labels.max()) # should be 0-9
        
        batch_acc = distance_baseline(features, labels)
        total_correct += batch_acc*labels.size(0)
        total_samples += labels.size(0)

    final_acc = total_correct / total_samples
    print(f"Distance baseline accuracy: {final_acc*100:.4f}")

    result = {"accuracy": final_acc, "method": "distance"}
    with open("results/distance_result.yaml", "w") as f:
        yaml.dump(result, f)

if __name__ == "__main__":
    main()