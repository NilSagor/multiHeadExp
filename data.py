import random
import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn



CIFAR100_SUPERCLASS_CLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bowl', 'can', 'cup', 'plate', 'bottle'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_sized_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

# Pick first class from each superclass → 20 diverse classes
SELECTED_CLASSES = [classes[0] for classes in CIFAR100_SUPERCLASS_CLASSES.values()]
# SELECTED_CLASSES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

print(len(SELECTED_CLASSES))


def get_selected_class_indices(dataset):
    selected_set = set(SELECTED_CLASSES)
    dataset_classes = set(dataset.classes)
    missing_classes = selected_set - dataset_classes
    if missing_classes:
        print(f"Warning these classes not found in CIFAR-100: {missing_classes}")
        print(f"Available classes: {sorted(dataset.classes)}")
    class_to_idx = {cls:idx for cls, idx in dataset.class_to_idx.items() if cls in selected_set}
    return class_to_idx



def get_cifar100_dataset():
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_set = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    return train_set, test_set

def filter_dataset_to_selected_classes(dataset, selected_class_indices):
    """Return subset of dataset with only selected classes"""
    selected_indices = []
    new_targets = []
    new_class_to_idx = {cls: i for i, cls in enumerate(selected_class_indices.keys())}
    
    for i, (img, target) in enumerate(dataset):
        class_name = dataset.classes[target]
        if class_name in selected_class_indices:
            selected_indices.append(i)
            new_targets.append(new_class_to_idx[class_name])
    
    subset = torch.utils.data.Subset(dataset, selected_indices)
    subset.targets = new_targets  # Add targets attribute
    subset.classes = list(new_class_to_idx.keys())
    return subset



class SetAnomalyDataset(Dataset):
    def __init__(self, dataset,  set_size=10, num_sets=10000, seed=42):
        self.dataset = dataset
        self.set_size = set_size
        self.num_sets = num_sets
        self.seed = seed
        # self.rng = random.Random(seed)
        
        # Group images by class
        self.class_to_indices = {}
        for i, target in enumerate(dataset.targets):
            if target not in self.class_to_indices:
                self.class_to_indices[target] = []
            self.class_to_indices[target].append(i)

        # Initialize ResNet18 feature extractor (frozen)
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.device = torch.device("cpu")
    
    def __len__(self):
        return self.num_sets
    
    def __getitem__(self, idx):
        # randomly select two different classes
        rng = np.random.RandomState(self.seed + idx)
        
        # get available classes 
        classes = list(self.class_to_indices.keys())
        
        # get two different classes
        c1 = rng.choice(classes)
        c2 = rng.choice([c for c in classes if c != c1])

        # sample 9 from c1, 1 from c2
        # Get indices 
        c1_indices = self.class_to_indices[c1]
        c2_index = self.class_to_indices[c2]

        # sample with replacement (use copy to avoid modifying original)
        c1_samples = rng.choice(c1_indices, size=9, replace=False)
        c2_samples = rng.choice(c2_index, size=1, replace=False)[0]

        # Get images
        images = []
        for i in  c1_samples:
            img, _ = self.dataset[i]
            images.append(img)
        img, _ = self.dataset[c2_samples]
        images.append(img)

        # create list of indices for shuffling
        indices = list(range(len(images)))
        rng.shuffle(indices)

        # Find new position of anomaly
        gt_index = indices.index(9) # original anomaly was at 9

        #
        
        # # shuffle and record anomaly index
        # anomaly_idx = len(images) - 1 # c2 image is last before shuffle
        # combined = list(zip(images, range(len(images))))
        # np.random.shuffle(combined) # shuffle with fixed seed per set
        # shuffled_images, original_indices = zip(*combined)
        

        # # find new anomaly position
        # gt_index = original_indices.index(anomaly_idx)
        
        shuffled_images = [images[i] for i in indices]

        # img_set = torch.stack(shuffled_images, dim=0) # [10, 3, 32, 32]
        
        # 2 stack and resize to 224x224
        img_tensor = torch.stack(shuffled_images) # [10, 3, 32, 32]
        img_tensor = F.interpolate(
            img_tensor, size=(224,224), mode="bilinear", align_corners=False
        )

        # 3. Extract features (move to GPU)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor) # [10, 512, 1, 1]
            features = features.view(10, -1) # [10, 512]
        
        return features, gt_index


def prepare_data(config, smoke_test=False):
    
    # set_size = config['data']['set_size']  # Should be 10
    set_size = config["data"]['set_size']  # Should be 10
    # seed = config['experiment']['seed']
    seed = config['seed']
    # batch_size = config['data']['batch_size']
    batch_size = config["data"]['batch_size']
    
    # load and filter datasets
    train_set, test_set = get_cifar100_dataset()
    
    train_class_indices = get_selected_class_indices(train_set)
    test_class_indices = get_selected_class_indices(test_set)

    train_filtered = filter_dataset_to_selected_classes(train_set, train_class_indices)
    test_filtered = filter_dataset_to_selected_classes(test_set, test_class_indices)

    if smoke_test:
        train_sets = 100
        val_sets = 50
        test_sets = 50
        batch_size = 4
    else:
        train_sets = 8000
        val_sets = 2000
        test_sets = 2000

    
    # calculate maximum possible sets
    max_train_sets = len(train_filtered) // set_size
    max_test_sets = len(train_filtered) // set_size

    train_sets = min(train_sets, max_train_sets)
    val_sets = min(val_sets, max_train_sets) # use train data for val
    test_sets = min(test_sets, max_test_sets)
    # if smoke_test:
    #     train_sets = min(100, max_train_sets)
    #     test_sets = min(50, max_test_sets)
    # else:
    #     train_sets = min(8000, max_train_sets)
    #     test_sets = min(2000, max_test_sets)

    train_anom = SetAnomalyDataset(
        train_filtered, 
        set_size=set_size, 
        num_sets=train_sets, 
        seed=seed
    )
    # val_sets = min(2000, len(train_anom)//4)
    
    val_anom = SetAnomalyDataset(
        train_filtered, 
        set_size=set_size, 
        num_sets=val_sets, 
        seed=seed+1000
    )

    test_anom = SetAnomalyDataset(
        test_filtered, 
        set_size=set_size, 
        num_sets=test_sets, 
        seed=seed + 2000
    )
    # test_anom = SetAnomalyDataset(test_set, set_size=set_size)
    
    # val_size = len(train_anom) // 10
    # train_size = len(train_anom) - val_size

    # train_subset, val_subset = random_split(
    #     train_anom, [train_size, val_size]
    # )
    
    # Create dataloaders
    num_workers = 0 if smoke_test else 4  # Use 0 workers for debugging
    train_loader = DataLoader(
        train_anom,
        batch_size = batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_anom,
        batch_size = batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_anom,
        batch_size = batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, val_loader, test_loader