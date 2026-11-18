from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import laplace

# Set root so that the script can run from anywhere in the repo
ROOT = Path(__file__).resolve().parent.parent
SPLIT = "Training"
DATA_DIR = ROOT / "dataset"
FIG_DIR = ROOT / "figures"
# Creates figure folder if not already existing
FIG_DIR.mkdir(exist_ok=True)

# Set constants for analysis. 
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
SIZE = 224
BACKGROUND_THRESHOLD = 0.05   # pixels darker than 5% of full brightness count as background


# Function to load an image and preprocess it for analysis. 
def load_image(path):
    """Load an image the way the model sees it: grayscale, 224x224, values 0-1."""
    with Image.open(path) as img:
        img = img.convert("RGB").convert("L").resize((SIZE, SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


rows = []            # one dict of measurements per image (item 5)
class_means = {}     # the average image of each class (item 6)

# Loop through each class, load the images, and compute statistics for each image. We also compute the mean image for each class.
for cls in CLASSES:
    folder = DATA_DIR / "Testing" / cls
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_TYPES]
    total = np.zeros((SIZE, SIZE), dtype=np.float64)

    # Loop through each file in the class folder
    for f in files:
        # Load the image and preprocess it
        arr = load_image(f)
        total += arr
        # Update our rows list with the dictionary of measurements for this image
        rows.append({
            "class": cls,
            "brightness": arr.mean(),
            "contrast": arr.std(),
            "background": (arr < BACKGROUND_THRESHOLD).mean(),
            "sharpness": laplace(arr).var(),
        })
    # Compute the mean image for this class and store it in the class_means dictionary
    class_means[cls] = total / len(files)
    print(f"{cls}: {len(files)} images")

# Here we convert our rows list of dictionaries into a pandas DataFrame for easier analysis and visualization.
stats = pd.DataFrame(rows)
measures = ["brightness", "contrast", "background", "sharpness"]

# Here we print the median values of our measures for each class, rounded to 3 decimal places. 
print(stats.groupby("class")[measures].median().round(3))

# Create boxplots for each measure, grouped by class. We use subplots to create a 1x4 grid of plots, one for each measure. We also remove the automatic title added by pandas and save the figure to the figures directory.
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, m in zip(axes, measures):
    stats.boxplot(column=m, by="class", ax=ax)
    ax.set_title(m)
    ax.set_xlabel("")
fig.suptitle("")                      # removes pandas' automatic title
plt.tight_layout()
fig.savefig(FIG_DIR / "per_class_image_stats.png", dpi=150)

# Again, here we create a boxplot for each different class
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), layout="constrained")
for ax, cls in zip(axes, CLASSES):
    ax.imshow(class_means[cls], cmap="gray", vmin=0, vmax=1)
    ax.set_title(cls)
    ax.axis("off")
fig.savefig(FIG_DIR / f"average_image_per_class_{SPLIT.lower()}.png", dpi=150)

overall_mean = np.mean([class_means[cls] for cls in CLASSES], axis=0)
diffs = {cls: class_means[cls] - overall_mean for cls in CLASSES}
limit = max(np.abs(d).max() for d in diffs.values())   # same color scale for all four panels

fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), layout="constrained")
for ax, cls in zip(axes, CLASSES):
    im = ax.imshow(diffs[cls], cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(f"{cls} − overall average")
    ax.axis("off")
fig.colorbar(im, ax=axes, shrink=0.8, label="difference in brightness")
fig.savefig(FIG_DIR / f"average_image_difference_{SPLIT.lower()}.png", dpi=150)

plt.show()

'''
Across both the training and test sets, no-tumor images are on average brighter, higher in contrast 
and richer in high-frequency detail than the tumor images, and they vary considerably more. This is 
consistent with them coming from a different source dataset than the tumor classes. Because the test 
set has the same source mix, a CNN could learn to recognize no-tumor scans from these acquisition
related properties rather than from the absence of a tumor, so a high test accuracy, particularly 
for the no-tumor class, may overstate how well the model would generalize to scans from other scanners or hospitals.
'''
