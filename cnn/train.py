from torch import nn  # Provides the loss function.
import torch
from torch.utils.data import DataLoader  # Groups images into batches.
from torchvision.datasets import ImageFolder  # Reads images from class folders.

from data_preprocessing import train_transform  # Prepares each image for the CNN.
from model import CNN  # Imports our model.

dataset = ImageFolder("dataset/Training", transform=train_transform)  # Locate training images and assign labels.
loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Make shuffled batches of eight images.

model = CNN(num_classes=len(dataset.classes))  # Create a model with one output per class.
criterion = nn.CrossEntropyLoss()  # Define how to measure prediction errors.

images, labels = next(iter(loader))  # Load one batch of images and correct labels.
outputs = model(images)  # Calculate class scores for the batch.
loss = criterion(outputs, labels)  # Compare the scores with the correct labels.

print("Classes:", dataset.class_to_idx)  # Show which number represents each class.
print("Loss:", loss.item())  # Display the batch's error as a number.


# Create Adam optimizer to update weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    total_loss = 0.0
    total_images = 0

    for images, labels in loader:
        optimizer.zero_grad()            # Clear gradients from the previous batch.
        outputs = model(images)           # Produce four scores for each MRI.
        loss = criterion(outputs, labels) # Measure the batch's average error.
        loss.backward()                  # Calculate gradients for this batch.
        optimizer.step()                 # Use those gradients to update the weights.

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    print(f"Epoch {epoch + 1}: average training loss = {total_loss / total_images:.4f}")
