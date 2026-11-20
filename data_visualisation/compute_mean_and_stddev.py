"""
Computes the pixel mean and standard deviation of the training split, for transforms.Normalize.

Uses the same preprocessing as the pipeline (grayscale, resize) but no augmentation and no Normalize,
and only the training part of the split (not the validation images, not the test set).

Run from the repo root:
    python data_visualisation/compute_mean_and_stddev.py
or from inside data_visualisation/:
    python compute_mean_and_stddev.py
Then copy the printed MEAN, STD line into cnn/data_preprocessing.py.
"""
import sys
from pathlib import Path

# This script lives in data_visualisation/, but data_preprocessing.py lives in cnn/.
# Python only looks for imports in the script's own folder (and installed packages),
# so we add cnn/ to the list of folders it searches.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cnn"))

from sklearn.model_selection import train_test_split  
from torch.utils.data import DataLoader, Subset       
from torchvision import datasets, transforms          

# settings shared with the training pipeline, so the split and the image size are identical
from data_preprocessing import DATA_DIR, IMG_SIZE, SEED, VAL_FRACTION, get_loaders 

# same preprocessing as the pipeline, but no augmentation and no Normalize: we're measuring raw 0-1 values
stats_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_full = datasets.ImageFolder(DATA_DIR / "Training", transform=stats_transform)

# the same stratified split as get_loaders(): same arguments, same seed, so the same images
indices = list(range(len(train_full)))
train_idx, _ = train_test_split(
    indices,
    test_size=VAL_FRACTION,
    stratify=train_full.targets,   # same class proportions in train and validation
    random_state=SEED,             # same split every time it runs
)

# safety check: confirm these really are the images get_loaders() trains on
train_loader, _, _, _ = get_loaders()
assert list(train_loader.dataset.indices) == list(train_idx), "split differs from get_loaders()"

loader = DataLoader(Subset(train_full, train_idx), batch_size=64)

# running totals: sum of pixel values, sum of squared pixel values, number of pixels
total, total_sq, n = 0.0, 0.0, 0
for images, _ in loader:
    images = images.double()          # extra precision for summing millions of values
    total += images.sum().item()
    total_sq += (images ** 2).sum().item()
    n += images.numel()

mean = total / n
std = (total_sq / n - mean ** 2) ** 0.5     # variance = mean of squares - square of mean

print(f"training images used: {len(train_idx)} (same split as get_loaders)")
print(f"MEAN, STD = [{mean:.4f}], [{std:.4f}]")