import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Define the neural network model with convolution and max pooling layers
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 1 input channel (grayscale), 8 output channels, 3x3 convolution
        self.conv2 = nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1) # 8 input channels, 16 output channels, 3x3 convolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 2x2 max pooling layer
        self.fc1 = nn.Linear(12 * 7 * 7, 32) # 16 channels, 7x7 after pooling
        self.fc2 = nn.Linear(32, 10) # Fully connected layer to output 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to count the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to load and preprocess the data
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    return train_loader, val_loader

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        print(f"Training started for epoch {epoch + 1}...")
        # Iterate over the training data
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    # Calculate the final training accuracy for the last epoch
    accuracy = 100 * correct / total
    return accuracy

# Function to validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    print("Validation started...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Validation Batch {batch_idx}/{len(val_loader)} - Loss: {val_loss/(batch_idx+1):.4f}, Accuracy: {100 * correct / total:.2f}%')

    accuracy = 100 * correct / total
    val_loss /= len(val_loader)
    return val_loss, accuracy

# Function to evaluate the model's performance
def evaluate_accuracy(correct, total):
    accuracy = 100 * correct / total
    if accuracy >= 95:
        print("The model achieved the required training accuracy in 1 epoch")
    else:
        print("The model did not achieve the required training accuracy in 1 epoch")
    return accuracy
