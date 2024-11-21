import torch.optim as optim
import torch.nn as nn
from train import MyCNN, count_parameters, get_data_loaders, train_model, show_augmented_images

# Set the seed for reproducibility
# set_seed()

def test_model():
    # Configuration
    batch_size = 64
    num_epochs = 1
    accuracy_threshold = 95
    parameter_threshold = 25000

   # Load data
    train_loader, val_loader, original_dataset = get_data_loaders(batch_size=batch_size)

    # # Show augmented images
    # show_augmented_images(original_dataset, train_loader.dataset)

    # Initialize the model
    model = MyCNN()
    total_parameters = count_parameters(model)

    # Check if the model has less than 25000 parameters
    assert total_parameters < parameter_threshold, f"Model has {total_parameters} parameters, which is not less than {parameter_threshold}"
    print(f"Model has {total_parameters} parameters, which is within the threshold of {parameter_threshold}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train the model
    train_accuracy = train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Check if the model has an accuracy of more than 95% in 1 epoch
    assert train_accuracy >= accuracy_threshold, f"Model accuracy is {train_accuracy}%, which is not greater than {accuracy_threshold}%"
    print(f"Model achieved an accuracy of {train_accuracy}%, which meets the threshold of {accuracy_threshold}%")

    # return train_accuracy

if __name__ == "__main__":
    test_model()
