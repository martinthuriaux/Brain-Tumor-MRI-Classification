from torchvision import transforms

IMG_SIZE = 224
MEAN, STD = [0.5], [0.5]

# training images: preprocessing + random augmentation
# The compose here does it so that the transformations are applied in the order they are listed. 
train_transform = transforms.Compose([
    # Grayscale turns the image into a single channel and resize makes sure all images are the same size. This is important because CNNs require a fixed input size.
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), #this is good but transforms every image into a square can distort the original image if it is rectangular. We can also use transforms.CenterCrop to crop the image to a square instead of resizing it.

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