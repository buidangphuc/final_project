from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.1,
    ckpt_path: Optional[str] = None,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # LR scheduler: reduce at 60%, 80% of total epochs
    milestone1 = int(epochs * 0.6)
    milestone2 = int(epochs * 0.8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    model = model.to(device)
    
    prev_loss = None
    loss_threshold = 0.01  # Skip logging if loss change is less than this
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate accuracy after epoch 10
            if epoch >= 10:
                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        
        scheduler.step()
        if ckpt_path and ((epoch + 1) % 20 == 0 or epoch == epochs - 1):
            torch.save(model.state_dict(), ckpt_path)
        
        avg_loss = running_loss / len(train_loader)
        
        # Check if we should log
        should_log = True
        if prev_loss is not None and epoch >= 10:
            loss_change = abs(avg_loss - prev_loss)
            if loss_change < loss_threshold:
                should_log = False
        
        if should_log:
            if epoch >= 10:
                train_acc = correct / total if total > 0 else 0.0
                print(f"Epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} train_acc {train_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} loss {avg_loss:.4f}")
        
        prev_loss = avg_loss
    return model


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    return correct / total if total > 0 else 0.0
