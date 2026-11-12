from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 224
VAL_FRACTION = 0.15
SEED = 42

# Note that this mean and standard deviation are just guesses for now.
# TODO: Improve these values by calculating the mean and standard deviation of the training dataset.
MEAN, STD = [0.5], [0.5]

# training images: preprocessing + random augmentation
# The compose here does it so that the transformations are applied in the order they are listed.
train_transform = transforms.Compose([
    # Grayscale turns the image into a single channel (our model expects in_channels=1).
    # Resize makes every image exactly 224x224, so the images in a batch can be stacked into one tensor.
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # We apply random horizontal flip, random rotation, and color jitter to augment the data. This helps the model generalize better by seeing different variations of the images during training.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),

    # Finally, we convert the image to a tensor and normalize it. Normalization helps as networks learn better when the input data is centered around zero and has a standard deviation of one.
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# validation and test images: preprocessing only, no randomness
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# path to dataset/, worked out from this file's location (cnn/data_preprocessing.py -> repo root -> dataset)
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


def get_loaders(batch_size=32, val_fraction=VAL_FRACTION, seed=SEED, num_workers=2):
    """Returns train_loader, val_loader, test_loader, class_names."""
    # Here we load the datasets using ImageFolder, which assumes that the images are organized in subdirectories named after their classes.
    # The Training folder is loaded twice: train_full applies train_transform (augmented) and train_full_eval applies eval_transform (not augmented).
    # The test_set applies eval_transform.
    train_full      = datasets.ImageFolder(DATA_DIR / "Training", transform=train_transform)
    train_full_eval = datasets.ImageFolder(DATA_DIR / "Training", transform=eval_transform)
    test_set        = datasets.ImageFolder(DATA_DIR / "Testing",  transform=eval_transform)

    # both copies must list the files in the same order, otherwise the validation labels would be wrong
    assert train_full.samples == train_full_eval.samples, "the two copies list files in different orders"

    # split the image positions 0 ... 5599 into two groups, keeping the class balance
    indices = list(range(len(train_full)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=train_full.targets,   # same class proportions in both groups
        random_state=seed,             # same split every time it runs
    )

    train_set = Subset(train_full, train_idx)        # 85%, augmented
    val_set   = Subset(train_full_eval, val_idx)     # 15%, not augmented

    # Create data loaders for training, validation, and testing.
    # shuffle=True for training so every batch mixes the classes and the order changes each epoch.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_full.classes