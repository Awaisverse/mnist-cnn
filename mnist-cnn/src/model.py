"""
MNIST CNN Model Architecture

A Convolutional Neural Network for MNIST digit classification.
Input: 28x28 grayscale images (1 channel)
Output: 10 classes (digits 0-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    CNN model for MNIST digit classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activations
    - MaxPooling layers
    - Fully connected layers
    - Output: 10 classes
    """
    
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # First convolutional block
        # Input: 1 channel (grayscale), Output: 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # After conv1: 28x28x32
        
        # MaxPooling: reduces size by half
        # After pool1: 14x14x32
        
        # Second convolutional block
        # Input: 32 channels, Output: 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # After conv2: 14x14x64
        
        # MaxPooling: reduces size by half again
        # After pool2: 7x7x64 = 3136 features
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for digits 0-9
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        # First conv block: conv -> relu -> maxpool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second conv block: conv -> relu -> maxpool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten for fully connected layers
        x = x.view(-1, 7 * 7 * 64)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final output layer (no activation - will use CrossEntropyLoss)
        x = self.fc2(x)
        
        return x


def get_model(device=None):
    """
    Helper function to create and return the model.
    
    Args:
        device: torch.device to move model to (e.g., 'cuda' or 'cpu')
        
    Returns:
        MNISTCNN model instance
    """
    model = MNISTCNN()
    
    if device is not None:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model architecture
    print("Testing MNIST CNN Model Architecture")
    print("-" * 50)
    
    # Create model instance
    model = MNISTCNN()
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Model architecture verified!")
    print(f"✓ Expected output shape: (batch_size, 10)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
