import torch
from torch import nn
from pathlib import Path

from data_preprocessing import get_loaders
from model import CNN

# Create the folder where model weights will be saved
Path("checkpoints_2").mkdir(exist_ok=True)

# Load training, validation and test data from data_preprocessing.py
train_loader, val_loader, test_loader, class_names = get_loaders()

print("Classes:", class_names)
print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and validation loop
for epoch in range(20):

    model.train()
    total_loss = 0.0
    total_images = 0
    correct = 0

    for images, labels in train_loader: # Loop through each batch of images and labels in the training data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear gradients from the previous batch.

        outputs = model(images) # Produce four scores for each MRI.
        predictions = outputs.argmax(dim=1) # Find the predicted class (highest score) for each image
        loss = criterion(outputs, labels) # Measure the batch's average error.

        loss.backward()  # Calculate gradients for this batch.
        optimizer.step()  # Use those gradients to update the weights.


        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        correct += (predictions == labels).sum().item()

    # Calculate average loss and training accuracy
    train_loss = total_loss / total_images
    train_accuracy = 100 * correct / total_images

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_images = 0
    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_images += batch_size
            val_correct += (predictions == labels).sum().item()

    val_loss = val_loss / val_images
    val_accuracy = 100 * val_correct / val_images

    # Print the results 
    print(
        f"Epoch {epoch + 1}: "
        f"Train loss = {train_loss:.4f} | "
        f"Train accuracy = {train_accuracy:.2f}% | "
        f"Val loss = {val_loss:.4f} | "
        f"Val accuracy = {val_accuracy:.2f}%"
    )

    # Save model weights after each epoch
    torch.save(
        model.state_dict(),
        f"checkpoints_2/cnn_epoch_{epoch + 1}.pth"
    )

print("Finished Training")
