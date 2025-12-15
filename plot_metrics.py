import pandas as pd
import matplotlib.pyplot as plt



def plot_training_curves():
    # load metrics
    train_df = pd.read_csv("logs/lightning_logs/version_1/metrics.csv")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Training loss 

    train_loss = train_df.dropna(subset=["train_loss"])
    axes[0,0].plot(train_loss["epoch"], train_loss["train_loss"])
    axes[0,0].set_title("Train Loss")

    val_loss = train_df.dropna(subset=["val_loss"])
    axes[0,1].plot(val_loss["epoch"], val_loss["val_loss"])
    axes[0,1].set_title("validation Loss")

    train_acc = train_df.dropna(subset=["train_acc"])
    axes[1,0].plot(train_acc["epoch"], train_acc["train_acc"])
    axes[1,0].set_title("Train Accuracy")

    val_acc = train_df.dropna(subset=["val_acc"])
    axes[1,1].plot(val_acc["epoch"], val_acc["val_acc"])
    axes[1,1].set_title("validation Accuracy")
    
    plt.tight_layout()
    plt.savefig("results/phase2_training_curves.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_training_curves()