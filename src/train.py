import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TinyCNN


BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

torch.manual_seed(42)
torch.set_num_threads(1)


def load_dataset(path):
    data = np.load(path)

    images = data["images"].astype(np.float32) / 255.0
    labels = data["labels"].astype(np.int64)

    images = torch.from_numpy(images).unsqueeze(1)
    labels = torch.from_numpy(labels)

    return TensorDataset(images, labels)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += len(labels)

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


train_dataset = load_dataset("data/processed/train.npz")
val_dataset = load_dataset("data/processed/val.npz")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

with open("data/processed/classes.json", encoding="utf-8") as f:
    class_names = json.load(f)

model = TinyCNN(num_classes=len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

Path("models").mkdir(exist_ok=True)
model_path = "models/drawguess_cnn.pth"
best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    start_time = time.perf_counter()

    train_loss, train_accuracy = run_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_loss, val_accuracy = run_epoch(
        model,
        val_loader,
        criterion,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            },
            model_path,
        )

    elapsed = time.perf_counter() - start_time

    print(
        f"Époque {epoch}/{EPOCHS} | "
        f"train loss: {train_loss:.4f} | "
        f"train accuracy: {train_accuracy:.1%} | "
        f"val loss: {val_loss:.4f} | "
        f"val accuracy: {val_accuracy:.1%} | "
        f"temps: {elapsed:.1f} s"
    )

print(f"Meilleure précision validation : {best_val_accuracy:.1%}")
print(f"Modèle sauvegardé dans : {model_path}")