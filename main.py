import torch.optim as optim
import torch.nn as nn

# Import functions and classes from train.py
from train import MyCNN, count_parameters, get_data_loaders, train_model, validate_model, evaluate_accuracy

def main():
    batch_size = 64
    num_epochs = 1  # Set the number of epochs to 1 for now
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Initialize the model
    model = MyCNN()
    print(f'Total number of parameters: {count_parameters(model)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_accuracy = train_model(model, train_loader, criterion, optimizer, num_epochs)
    print(f'Training Accuracy: {train_accuracy:.2f}%')

    # Validate the model
    val_loss, val_accuracy = validate_model(model, val_loader, criterion)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Evaluate the accuracy
    correct = int(train_accuracy * len(train_loader.dataset) / 100)
    total = len(train_loader.dataset)
    evaluate_accuracy(correct, total)

if __name__ == '__main__':
    main()
