import torch.optim.adam
from torchvision import models, transforms
import torch.nn as nn
import torch
from tt.transfer_learning_preprocessing import get_transfer_loaders
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                             precision_score, confusion_matrix, 
                             classification_report
                             )


def main():
    # load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze pretrained parameter, so that the pre-trained parameters
    # are not updated during training, we use the ReseNet18 already trained parameters
    for param in model.parameters():
        param.requires_grad = False


    model.fc = nn.Linear(model.fc.in_features, 4) # new layer w. random weights/biases

    # 3 loaders that have been preprocessed for ResNet18
    train_loader, val_loader, test_loader, classes = get_transfer_loaders()

    # retrieve first batch
    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

    # keep the frozen feature extraction in evaluation mode
    model.eval()

    # without tracking gradients, just checking predictions
    with torch.no_grad():
        outputs = model(images)

    # convert logits into probabilites, then predict class w. highest probability
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)

    criterion = nn.CrossEntropyLoss()

    optimser = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.001
    )

    def validate (model, val_loader, criterion):

        model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_loader:

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss * images.size(0)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(
            all_labels, all_predictions, 
            average="macro", zero_division=0
        )
        f1 = f1_score(
            all_labels, all_predictions,
            average="macro", zero_division=0
        )
    
        return avg_loss, accuracy, recall, f1
    

    

    num_epochs = 10

    for epoch in range(num_epochs):

        # training
        model.eval()
        model.fc.train()


        running_loss = 0.0

        for images, labels in train_loader:

            # reset gradient from previous batch
            optimser.zero_grad()

            # forward pass
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)

            # backpropagation
            loss.backward()

            # update the classifier parameters
            optimser.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc, val_recall, val_f1 = validate(model, val_loader, criterion)            


        # Display results for this epoch
        print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Accuracy: {val_acc:.2%} | "
                f"Recall: {val_recall:.2%} | "
                f"F1: {val_f1:.2%}"
            )



if __name__ == "__main__":
    main()
