import csv
import re
import time
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm

from data_preprocessing import get_loaders
from model import CNN

# checkpoints/ at the repo root, worked out from this file's location (cnn/train.py -> repo root)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
EPOCHS = 2

def next_run_number(folder):
    """Return the next free run number N, based on the cnnN_epoch<E>_best_model.pth files already in folder."""
    numbers = []
    for path in folder.glob("cnn*_best_model.pth"):
        match = re.fullmatch(r"cnn(\d+)_epoch\d+_best_model\.pth", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def get_device():
    """Use an NVIDIA GPU or an Apple Silicon GPU if there is one, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # Create the folder where model weights will be saved, and pick this run's number
    # so that each run writes its own cnnN_... file and never overwrites an earlier run
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_number = next_run_number(CHECKPOINT_DIR)
    print(f"Run number: {run_number} (best model will be saved as {CHECKPOINT_DIR.name}/cnn{run_number}_epoch<E>_best_model.pth)")

    # Training history: one row per epoch, used by evaluation.py to plot the loss and accuracy curves.
    # It is written after every epoch, so the curves are kept even if training stops early.
    history_path = CHECKPOINT_DIR / f"cnn{run_number}_history.csv"
    history_columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_seconds"]
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(history_columns)
    print(f"Training history will be saved to {history_path.relative_to(CHECKPOINT_DIR.parent)}")

    # Load training, validation and test data from data_preprocessing.py
    train_loader, val_loader, test_loader, class_names = get_loaders()

    print("Classes:", class_names)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    device = get_device()
    print("Using device:", device)
    model = CNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Cosine schedule: the learning rate starts at 0.001 and smoothly decreases to almost 0 by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*EPOCHS)
    
    # Best model so far: highest validation accuracy, ties broken by the lower validation loss
    # (the same rule evaluation.py uses to pick the best epoch)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_path = None

    # Training and validation loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        total_images = 0
        correct = 0

        # Progress bar over the training batches, showing the running loss and accuracy
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", unit="batch")

        for images, labels in train_bar: # Loop through each batch of images and labels in the training data
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

            train_bar.set_postfix(loss=f"{total_loss / total_images:.4f}",
                                  acc=f"{100 * correct / total_images:.2f}%")

        # Calculate average loss and training accuracy
        train_loss = total_loss / total_images
        train_accuracy = 100 * correct / total_images

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_images = 0
        val_correct = 0

        with torch.no_grad():

            # leave=False removes the validation bar once it finishes, to keep the output short
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]",
                                       unit="batch", leave=False):
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
        
        scheduler.step()  # lower the learning rate for the next epoch

        # Add this epoch to the training history 
        epoch_seconds = time.time() - epoch_start
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(train_loss, 6), round(train_accuracy, 4),
                                    round(val_loss, 6), round(val_accuracy, 4), round(epoch_seconds, 1)])

        # Save the model weights only if this epoch is the best one so far
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy = val_accuracy
            best_loss = val_loss

            new_path = CHECKPOINT_DIR / f"cnn{run_number}_epoch{epoch + 1}_best_model.pth"
            torch.save(model.state_dict(), new_path)

            # Delete this run's previous best, so each run keeps exactly one file
            if best_path is not None:
                best_path.unlink(missing_ok=True)
            best_path = new_path

            print(f"  New best model saved to {best_path.relative_to(CHECKPOINT_DIR.parent)}")

    print("Finished Training")
    print(f"Best model: {best_path.relative_to(CHECKPOINT_DIR.parent)} "
          f"(Val accuracy = {best_accuracy:.2f}% | Val loss = {best_loss:.4f})")


if __name__ == "__main__":
    main()
