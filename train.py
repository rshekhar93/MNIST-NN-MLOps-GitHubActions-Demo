import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import numpy as np

# Define the neural network model with convolution and max pooling layers
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 8x28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x14x14
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 8x14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 16x7x7
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# Function to count the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to load and preprocess the data with augmentation
# Data Augmentation: Added random rotation and affine transformations to the data loading process.
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        # transforms.RandomRotation(10),
        # transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced intensity
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(28, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    original_dataset = datasets.MNIST('./data', train=True, download=True, transform=original_transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=original_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    return train_loader, val_loader, original_dataset

# Function to display original and augmented images side by side
def show_augmented_images(original_dataset, augmented_dataset, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    for i in range(num_images):
        orig_img, _ = original_dataset[i]
        aug_img, _ = augmented_dataset[i]
        
        orig_img = orig_img.numpy().transpose((1, 2, 0))
        aug_img = aug_img.numpy().transpose((1, 2, 0))
        
        mean = np.array([0.1307])
        std = np.array([0.3081])
        
        orig_img = std * orig_img + mean
        aug_img = std * aug_img + mean
        
        orig_img = np.clip(orig_img, 0, 1)
        aug_img = np.clip(aug_img, 0, 1)
        
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_img, cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title('Original Images', fontsize=16)
    axes[1, 0].set_title('Augmented Images', fontsize=16)
    plt.show()


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
