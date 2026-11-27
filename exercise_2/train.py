"""
Exercise 2: Training and Comparative Evaluation Script
PUSL3123 - AI and Machine Learning

This script performs:
1. Loading CIFAR-10 dataset and creating reproducible Train / Validation splits (preserving Test partition).
2. Training the 3-Trainable-Layer CNN.
3. Training the 3-Trainable-Layer baseline MLP under comparable training settings.
4. Comparing classification accuracy, training/validation loss, and parameter counts.
5. Saving plots and results.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import ThreeLayerCNN, ThreeLayerMLP, count_trainable_parameters, print_model_summary


def set_seed(seed: int = 42):
    """Ensure reproducibility across random splits, initializations, and data loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cifar10(data_dir: str = "./data"):
    """
    Ensure CIFAR-10 is cleanly downloaded.
    If a previous download was interrupted (e.g. stopped at 97.5%), torchvision leaves a
    corrupted .tar.gz file which triggers RuntimeError("File not found or corrupted.").
    This function automatically detects and removes corrupted archives.
    """
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_folder = os.path.join(data_dir, "cifar-10-batches-py")
    
    # If already extracted and batches exist, no need to touch the archive
    if os.path.exists(extracted_folder):
        batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
        if all(os.path.exists(os.path.join(extracted_folder, b)) for b in batches):
            return
            
    # If tar.gz exists, verify integrity
    if os.path.exists(tar_path):
        import hashlib
        hash_md5 = hashlib.md5()
        with open(tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096 * 1024), b""):
                hash_md5.update(chunk)
        if hash_md5.hexdigest() != "c58f30108f718f92721af3b95e74349a":
            print(f"[Auto-Repair] Found incomplete/corrupted '{tar_path}'. Removing to allow clean download...")
            os.remove(tar_path)


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64, val_size: int = 5000):
    """
    Load CIFAR-10 and create reproducible Train/Val splits.
    The official 10,000 image test set is kept strictly separate.
    """
    prepare_cifar10(data_dir)
    
    # Standard normalization for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Official partitions: 50,000 train, 10,000 test
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_size = len(full_train_dataset) - val_size  # 45,000
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total
    return eval_loss, eval_acc


def run_experiment(model, train_loader, val_loader, test_loader, epochs=15, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")
              
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    
    return history, test_loss, test_acc


def plot_comparison(cnn_hist, mlp_hist, save_path="comparison_plot.png"):
    epochs = range(1, len(cnn_hist['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, cnn_hist['train_loss'], 'b-', label='CNN Train Loss')
    axes[0].plot(epochs, cnn_hist['val_loss'], 'b--', label='CNN Val Loss')
    axes[0].plot(epochs, mlp_hist['train_loss'], 'r-', label='MLP Train Loss')
    axes[0].plot(epochs, mlp_hist['val_loss'], 'r--', label='MLP Val Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, cnn_hist['train_acc'], 'b-', label='CNN Train Acc')
    axes[1].plot(epochs, cnn_hist['val_acc'], 'b--', label='CNN Val Acc')
    axes[1].plot(epochs, mlp_hist['train_acc'], 'r-', label='MLP Train Acc')
    axes[1].plot(epochs, mlp_hist['val_acc'], 'r--', label='MLP Val Acc')
    axes[1].set_title('Training & Validation Accuracy (%)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Models
    cnn_model = ThreeLayerCNN().to(device)
    mlp_model = ThreeLayerMLP().to(device)
    
    print_model_summary(cnn_model, "Exercise 2: 3-Layer CNN")
    print_model_summary(mlp_model, "Exercise 1 Baseline: 3-Layer MLP")
    
    # 2. Data Loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    
    # 3. Train CNN
    print("\n--- Training 3-Layer CNN ---")
    cnn_hist, cnn_test_loss, cnn_test_acc = run_experiment(
        cnn_model, train_loader, val_loader, test_loader, epochs=15, lr=0.001, device=device
    )
    
    # 4. Train MLP
    print("\n--- Training 3-Layer MLP (Comparable Settings) ---")
    mlp_hist, mlp_test_loss, mlp_test_acc = run_experiment(
        mlp_model, train_loader, val_loader, test_loader, epochs=15, lr=0.001, device=device
    )
    
    # 5. Save Comparison Plot
    plot_comparison(cnn_hist, mlp_hist, save_path="exercise_2_comparison.png")
    
    # 6. Summary Comparison Table
    print("\n================ COMPARATIVE SUMMARY ================")
    print(f"{'Metric':<30} | {'3-Layer MLP':<15} | {'3-Layer CNN':<15}")
    print("-" * 65)
    print(f"{'Trainable Parameters':<30} | {count_trainable_parameters(mlp_model):<15,} | {count_trainable_parameters(cnn_model):<15,}")
    print(f"{'Final Train Accuracy (%)':<30} | {mlp_hist['train_acc'][-1]:<15.2f} | {cnn_hist['train_acc'][-1]:<15.2f}")
    print(f"{'Final Val Accuracy (%)':<30} | {mlp_hist['val_acc'][-1]:<15.2f} | {cnn_hist['val_acc'][-1]:<15.2f}")
    print(f"{'Test Accuracy (%)':<30} | {mlp_test_acc:<15.2f} | {cnn_test_acc:<15.2f}")
    print(f"{'Final Validation Loss':<30} | {mlp_hist['val_loss'][-1]:<15.4f} | {cnn_hist['val_loss'][-1]:<15.4f}")
    print("=====================================================")
