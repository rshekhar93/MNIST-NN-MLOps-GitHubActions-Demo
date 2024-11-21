import torch.optim as optim
import torch.nn as nn
import time
from train import MyCNN, count_parameters, get_data_loaders, train_model, show_augmented_images, validate_model

def test_model():
    # Configuration
    batch_size = 64
    num_epochs = 1  # Run for more epochs for better learning
    accuracy_threshold = 95
    parameter_threshold = 25000
    loss_threshold = 0.2
    max_inference_time = 0.1  # seconds

    # Load data
    train_loader, val_loader, original_dataset = get_data_loaders(batch_size=batch_size)

    # Show augmented images
    show_augmented_images(original_dataset, train_loader.dataset)

    # Initialize the model
    model = MyCNN()
    total_parameters = count_parameters(model)

    # Check if the model has less than 25000 parameters
    assert total_parameters < parameter_threshold, f"Model has {total_parameters} parameters, which is not less than {parameter_threshold}"
    print(f"Model has {total_parameters} parameters, which is within the threshold of {parameter_threshold}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train the model and collect training loss
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, num_epochs)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validate the model
    val_loss, val_accuracy = validate_model(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Check if the training loss is within the threshold
    assert train_loss <= loss_threshold, f"Training loss is {train_loss}, which is not within the threshold of {loss_threshold}"

    # Check if the validation loss is within the threshold
    assert val_loss <= loss_threshold, f"Validation loss is {val_loss}, which is not within the threshold of {loss_threshold}"

    # Check if the model has an accuracy of more than 95% in 1 epoch
    assert train_accuracy >= accuracy_threshold, f"Model accuracy is {train_accuracy}%, which is not greater than {accuracy_threshold}%"
    print(f"Model achieved an accuracy of {train_accuracy}%, which meets the threshold of {accuracy_threshold}%")

    # Test inference time
    input_data, _ = next(iter(val_loader))
    start_time = time.time()
    _ = model(input_data)
    inference_time = time.time() - start_time

    assert inference_time / len(input_data) <= max_inference_time, f"Inference time per sample is {inference_time / len(input_data):.4f} seconds, which exceeds {max_inference_time} seconds"
    print(f"Inference time per sample is {inference_time / len(input_data):.4f} seconds, which is within the threshold")

if __name__ == "__main__":
    test_model()
