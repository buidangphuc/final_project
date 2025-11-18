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
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if ckpt_path and ((epoch + 1) % 20 == 0 or epoch == epochs - 1):
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch + 1}/{epochs} loss avg {running_loss / len(train_loader):.4f}")
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
