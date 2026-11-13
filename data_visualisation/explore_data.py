'''
This file is used to explore the dataset and get a sense of the number of images, their sizes, and their modes (e.g., RGB, grayscale) for each class in both the training and testing sets. 
It helps to understand the dataset's structure and characteristics before training the CNN model.
'''

from pathlib import Path
from collections import Counter
from PIL import Image

# Let's set up the path to the dataset and define the image file types we want to consider. We'll then iterate through the "Training" and "Testing" directories, counting the number of images, their modes, and their sizes for each class.
DATA_DIR = Path("dataset")
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

# Iterate through the "Training" and "Testing" strings, as we know these are the names of the directories containing the data.
for split in ["Training", "Testing"]:
    print(f"\n{split}")

    # Here we iterate through the glioma, meningioma, no tumor and pituitary tumor directories. 
    for class_dir in sorted((DATA_DIR / split).iterdir()):

        # We check if the current path is a directory. If it's not (for example, if it's a file like .gitignore), we skip it.
        if not class_dir.is_dir():         
            continue

        # Can do this step as list comprehension, but then it gets a bit complicated. Essentially we create a list of all elements with JPEG, JPG, or PNG extensions in the current class directory.
        files = []
        for f in class_dir.iterdir():                  
            if f.suffix.lower() in IMAGE_TYPES:        
                files.append(f)

        # Counter here is used to count the occurrences of each image size and mode. 
        sizes, modes = Counter(), Counter()

        # Loop through files, open each image, and update the counters for sizes and modes.
        for f in files:
            with Image.open(f) as img:     
                sizes[img.size] += 1
                modes[img.mode] += 1
        # Finally, we print out the results for each class, including the number of images, the modes of the images, and the most common sizes.        
        print(f"  {class_dir.name:<11} {len(files):>5} images | "
              f"modes: {dict(modes)} | most common sizes: {sizes.most_common(5)}")
        print(len(sizes))