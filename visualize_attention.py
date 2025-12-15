import torch 
import matplotlib.pyplot as plt
from model.predictor import AnomalyPredictor
from data import prepare_data
from run_single import load_config


def visualize_attention():
    model = AnomalyPredictor.load_from_checkpoint("checkpoints/phase2/best-epoch=41-val_acc=0.7490.ckpt")
    model.eval()
    
    config_path = "config.yaml"
    config = load_config(config_path)

    _,_, test_loader = prepare_data(config)
    features, labels = next(iter(test_loader))

    device = next(model.parameters()).device
    
    # Move the input features tensor to the model's device
    features = features.to(device)

    # Get attention maps
    with torch.no_grad():
        attention_maps = model.get_attention_maps(features) # list [4 layers]
    sample_idx = 0
    layer_idx = 0
    head_idx = 0
    attn = attention_maps[layer_idx][sample_idx, head_idx].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    im = ax.imshow(attn, cmap="viridis", interpolation="nearest")
    ax.set_title(f"Attention Map (Layer {layer_idx+1}, Head {head_idx + 1})")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im)
    plt.savefig(f"results/attention_sample{sample_idx}.png", dpi=300)
    plt.show()

    # Print anomaly position
    print(f"Anomaly at position {labels[sample_idx]}")

if __name__ == "__main__":
    visualize_attention()