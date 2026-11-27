from torchvision import transforms
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

transfer_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) #based on ResNet18 pretrained on RGB images (Red, Green, Blue)
])

IMG_SIZE = 224

# Training: preprocessing + augmentation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation and testing: no random augmentation
eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

VAL_FRACTION = 0.15
SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

def get_transfer_loaders(batch_size=32, num_workers=2):

    # Load the training images twice w. different transformations
    train_full = datasets.ImageFolder(
        DATA_DIR / "Training",
        transform=train_transforms
    )

    train_full_eval = datasets.ImageFolder(
        DATA_DIR / "Training",
        transform=eval_transforms
    )

    # Load test images
    test_set = datasets.ImageFolder(
        DATA_DIR / "Testing",
        transform=eval_transforms
    )

    # make sure both datasets have same image ordering
    assert train_full.samples == train_full_eval.samples

    # Create same train/validation split as CNN
    indices = list(range(len(train_full)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_FRACTION,
        stratify=train_full.targets,
        random_state=SEED
    )

    # Create subsets to train (on augmented photos) and test (on non-augmented photos)
    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full_eval, val_idx)

    # Create Dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_full.classes