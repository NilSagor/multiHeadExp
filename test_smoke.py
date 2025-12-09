import yaml
import torch 
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from data import (
    prepare_data, 
    SELECTED_CLASSES, 
    get_cifar100_dataset,
    get_selected_class_indices,
    filter_dataset_to_selected_classes
    )


def visualize_sample_sets(img_sets, labels, batch_idx=0, num_sets=2):
    print(f"\n Visualizing {num_sets} samples sets from batch {batch_idx}")

    # fig, axes = plt.subplots(num_sets, 1, figsize=(12,3*num_sets))
    # if num_sets == 1:
    #     axes = [axes]

    for set_idx in range(num_sets):
        # Get one set [10, 3, 32, 32]
        single_set = img_sets[set_idx]
        anomaly_pos = labels[set_idx].item()

        # convert to numpy for plotting [10, 32, 32, 3]
        images_np = single_set.permute(0, 2, 3, 1).numpy()
        # denormalize if need (cifar is 0-1 form totensor)
        images_np = np.clip(images_np, 0, 1)

        # create a grid with anomaly highlighted
        fig_set, ax = plt.subplots(2, 5, figsize=(12,5))
        fig_set.suptitle(f"Set {set_idx} | Anomaly at position {anomaly_pos}", fontsize=14)

        for i in range(10):
            row, col = i//5, i%5
            ax[row, col].imshow(images_np[i])
            ax[row, col].axis('off')

            # highlighted anomaly
            if i==anomaly_pos:
                for spine in ax[row, col].spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(3)

        plt.tight_layout()
        plt.savefig(f"results/smoke_test_set_{batch_idx}_{set_idx}.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_set)

    print(f" Sample sets saved to results/directory")







def test_smoke():
    """Quick test to verify data loading works"""
    
    print("=" * 50)
    print("SMOKE TEST: Data Pipeline")
    print("=" * 50)
    print("🧪 Running comprehensive smoke test...\n")

    # load minimal config
    config = {
        "set_size": 10,
        "seed": 42,
        "batch_size": 8,
        "model": {
            "num_classes": 10
        }
    }

    # Test 1: CIFAR-100 loading
    print(" 1️⃣ Testing CIFAR-100 dataset loading...")
    train_set, test_set = get_cifar100_dataset()
    print(f"   Train set: {len(train_set)} images")
    print(f"   Test set: {len(test_set)} images")
    assert len(train_set) == 50000
    assert len(test_set) == 10000
    print("   ✅ CIFAR-100 loaded correctly\n")

    # Test 2: Class filtering
    print(" 2️⃣ Testing class filtering...")
    train_class_indices = get_selected_class_indices(train_set)
    train_filtered = filter_dataset_to_selected_classes(train_set, train_class_indices)
    print(f"   Selected classes: {len(SELECTED_CLASSES)}")
    print(f"   Filtered train images: {len(train_filtered)}")

    print(f"   Images per class: {len(train_filtered) // len(SELECTED_CLASSES)}")
    # Should be 500 per class (50000/100 * 20 = 10000, but we have 20 classes → 500 each)
    expected_train_filtered = 20*500
    print(f" Expected filtered images: {expected_train_filtered}")

    assert len(train_filtered) >= 9900, f"Expected 10000 images, got {len(train_filtered)}"  # 20 classes × 500 images
    print("   ✅ Class filtering works\n")

    # Test 3: Set anomaly dataset
    print("3️⃣ Testing set anomaly dataset...")
    train_loader, val_loader, test_loader = prepare_data(config, smoke_test=True)
    
    # Check dataset sizes
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}") 
    print(f"   Test batches: {len(test_loader)}")

    print("\n Generating visualizations ... ")
    import os 
    os.makedirs("results", exist_ok=True)
    for batch_idx, (img_sets, labels) in enumerate(train_loader):
        if batch_idx == 0:
            visualize_sample_sets(img_sets, labels, batch_idx=batch_idx, num_sets=2)
            break

    
    # 🔍 DEBUG INFO GOES HERE
    print(f"🔍 Debug Info:")
    print(f"  Selected classes ({len(SELECTED_CLASSES)}): {SELECTED_CLASSES}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Set size: {config['model']['num_classes']}")


    loaders_to_test = [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader)
    ]

    all_train_labels = []

    for loader_name, loader in loaders_to_test:
        print(f"\n Testing {loader_name} loader: {len(loader)} batches")
        batch_count = 0
        max_batches = 5 if loader_name == "train" else 3

        for i, (img_sets, labels) in enumerate(loader):
            if i >= max_batches:
                break
            # handle variable batch sizes
            batch_size_actual = img_sets.shape[0]
            assert batch_size_actual > 0, f"Empty batch in {loader_name}"
            assert img_sets.shape[1:] == (10, 3, 32, 32), f"Wrong img shape {img_sets.shape}"
            assert labels.shape == (batch_size_actual,), f"Wrong labels shape {labels.shape}"
            assert torch.all((labels>= 0) & (labels < 10)), f"Invalid labels:{labels}"

            print(f" Batch {i}: OK| Image set: {img_sets.shape}, Labels: {labels.tolist()}")
            batch_count += 1

            if loader_name == "train":
                all_train_labels.extend(labels.tolist())

    # Test label distribution and shuffling
    if all_train_labels:

        label_counts = Counter(all_train_labels)
        print(f" Label distribution: (first {len(all_train_labels)} labels)")
        print(f" Count: {dict(label_counts)}")
        print(f" Unique positions used: {len(set(all_train_labels))}/10")
        assert len(set(all_train_labels)) > 1, " Labels are not shuffled property!"
        assert set(range(10)).issubset(set(all_train_labels)), "Not all positions (0-9) are being used!"
        print("Shuffling and label generation working correctly!")
    # Test that labels aren't all the same (shuffling works)
    # assert len(set(all_labels))>1, "Labels are not shuffled properly!"

    print("\n All smoke tests passed")
    print(" Ready for full training")

if __name__ == "__main__":
    test_smoke()




    # print(f"\n 🧪 Testing train loader: checking {len(train_loader)} batches")
    # for i, (img_sets, labels) in enumerate(train_loader):
    #     if i>=12:
    #         break
    #     assert img_sets.shape == (config["batch_size"], 10, 3, 32, 32), f"Wrong img shape {img_sets.shape}"
    #     assert labels.shape == (config["batch_size"],), f"Wrong labels shape {labels.shape}"
    #     assert torch.all(labels >= 0) and torch.all(labels < 10), f"Invalid labels: {labels}"
    #     print(f"Batch {i}:OK Image set shape {img_sets.shape}, Labels shape {labels[:10]}")

    # print(f"\n 🧪 Testing val loader: checking {len(val_loader)} batches")
    # for i, (img_sets, labels) in enumerate(val_loader):
    #     if i>=3:
    #         break
    #     assert img_sets.shape == (config["batch_size"], 10, 3, 32, 32), f"Wrong img shape {img_sets.shape}"
    #     assert labels.shape == (config["batch_size"],), f"Wrong labels shape {labels.shape}"
    #     assert torch.all(labels >= 0) and torch.all(labels < 10), f"Invalid labels: {labels}"
    #     print(f"Batch {i}:OK Image set shape {img_sets.shape}, Labels shape {labels[:10]}")
    
    # print(f"\n 🧪 Testing test loader: checking {len(test_loader)} batches")
    # for i, (img_sets, labels) in enumerate(test_loader):
    #     if i>=3:
    #         break
    #     assert img_sets.shape == (config["batch_size"], 10, 3, 32, 32), f"Wrong img shape {img_sets.shape}"
    #     assert labels.shape == (config["batch_size"],), f"Wrong labels shape {labels.shape}"
    #     assert torch.all(labels >= 0) and torch.all(labels < 10), f"Invalid labels: {labels}"
    #     print(f"Batch {i}:OK Image set shape {img_sets.shape}, Labels shape {labels[:10]}")

    #     if i>=5:
    #         break
    #     assert img_sets.shape == (8, 10, 3, 32, 32)
    #     assert labels.shape == (8,)
    #     assert torch.all((labels>=0) & (labels<10))
    #     all_labels.extend(labels.tolist())
    #     print(f" Batch {i}: labels={labels.tolist()}")