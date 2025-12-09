import yaml
import random
import torch 
import numpy as np
from pathlib import Path 
import lightning as L
# from model import DummyAnomalyModel
from model.predictor import AnomalyPredictor
from data import prepare_data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Ensure results dir exists
    Path("results").mkdir(exist_ok=True)


    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    set_seed(config["seed"])
    # print(config["seed"])
    
    
    print("\n=== Starting Training ===")
    train_loader, val_loader, test_loader = prepare_data(config)
    

    # create model
    print("Creating dummy model ...")
    # model = DummyAnomalyModel(num_classes=config["model"]["num_classes"])
    train_batches_per_epoch = len(train_loader)
    max_iters = config["training"]["max_epochs"]*train_batches_per_epoch
    lr=float(config["training"]["lr"])
    model = AnomalyPredictor(**config['model'], lr= lr, warmup=config["training"]["warmup"], max_iters=max_iters)

    print(" training (5 epochs)")
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        deterministic=True,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )

    
    print(f"Training for {config["training"]['max_epochs']} epochs...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\nFinal Testing")
    test_results = trainer.test(model, test_loader)
    test_acc = test_results[0]["test_acc"]

    result = {
        "config": config,
        "test_accuracy": float(test_acc),
        "epochs_run": trainer.current_epoch
    }

    with open("results/phase2_result.yaml", "w") as f:
        yaml.dump(result, f, default_flow_style=False, indent=2)

    print(f"\n🎉 Phase 1 {'Succeeded' if test_acc > 0.30 else 'Failed'}!")
    print(f"Test Accuracy: {test_acc:.4f} (Target: >0.30)")
    print(f" Result saved: results/phase1_result.yaml")
    print(f" Next: ...")
   
    

if __name__ == "__main__":
    main()