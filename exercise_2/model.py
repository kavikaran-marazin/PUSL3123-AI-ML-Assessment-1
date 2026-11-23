"""
Exercise 2: Convolutional Neural Network (CNN) Classifier
PUSL3123 - AI and Machine Learning

This module defines:
1. ThreeLayerCNN: A CNN with exactly three trainable layers (Conv2d, Conv2d, Linear).
2. ThreeLayerMLP: A baseline MLP with three trainable layers for comparative evaluation.
3. Utility function to calculate and display parameter counts.
"""

import torch
import torch.nn as nn


class ThreeLayerCNN(nn.Module):
    """
    Convolutional Neural Network with exactly THREE trainable layers.
    
    Structure:
      - Trainable Layer 1: Conv2d(3, 16, kernel_size=3, padding=1)
      - Activation: ReLU (non-trainable)
      - Pooling: MaxPool2d(2, 2) (non-trainable)
      - Trainable Layer 2: Conv2d(16, 32, kernel_size=3, padding=1)
      - Activation: ReLU (non-trainable)
      - Pooling: MaxPool2d(2, 2) (non-trainable)
      - Flatten: Converts 32 x 8 x 8 -> 2048 (non-trainable)
      - Trainable Layer 3: Linear(2048, 10) (Output layer)
    """
    def __init__(self, num_classes: int = 10):
        super(ThreeLayerCNN, self).__init__()
        
        # Trainable Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Trainable Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Trainable Layer 3 (Output classification layer)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ThreeLayerMLP(nn.Module):
    """
    Baseline Multi-Layer Perceptron (MLP) with exactly THREE trainable layers.
    Used for direct comparative analysis as required by Exercise 2.
    
    Structure:
      - Flatten: 3 x 32 x 32 = 3072 input features
      - Trainable Layer 1: Linear(3072, 256)
      - Activation: ReLU
      - Trainable Layer 2: Linear(256, 128)
      - Activation: ReLU
      - Trainable Layer 3: Linear(128, 10)
    """
    def __init__(self, num_classes: int = 10):
        super(ThreeLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def count_trainable_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print layer-by-layer parameter summary."""
    print(f"=== {model_name} Summary ===")
    total = 0
    trainable_layers_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total += num
            if "weight" in name:
                trainable_layers_count += 1
            print(f"  {name:25s} | Shape: {str(list(param.shape)):15s} | Params: {num:,}")
    print(f"Total Trainable Parameters : {total:,}")
    print(f"Trainable Weight Layers    : {trainable_layers_count}")
    print("-" * 45)
