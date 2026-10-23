import json
import time
from pathlib import Path
import csv
import copy
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TinyCNN


BATCH_SIZE = 32
EPOCHS = 20
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

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("outputs") / "runs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

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
best_epoch = 0
best_model_state = None
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

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())


    elapsed = time.perf_counter() - start_time

    print(
        f"Époque {epoch}/{EPOCHS} | "
        f"train loss: {train_loss:.4f} | "
        f"train accuracy: {train_accuracy:.1%} | "
        f"val loss: {val_loss:.4f} | "
        f"val accuracy: {val_accuracy:.1%} | "
        f"temps: {elapsed:.1f} s"
    )



history_path = run_dir / "history.csv"

with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(history.keys())

    for row in zip(*history.values()):
        writer.writerow(row)

config = {
    "architecture": "TinyCNN",
    "class_names": class_names,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "seed": 42,
}
with (run_dir / "config.json").open("w", encoding="utf-8") as file:
    json.dump(config, file, indent=2, ensure_ascii=False)

epochs = history["epoch"]

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], label="Train")
plt.plot(epochs, history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss pendant l'entraînement")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "training_curves.png", dpi=150)
plt.close()

checkpoint = {
    "model_state_dict": best_model_state,
    "class_names": class_names,
    "config": config,
    "history": history,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
}

run_model_path = run_dir / "checkpoint_best.pth"
torch.save(checkpoint, run_model_path)
torch.save(checkpoint, model_path)
print(f"Meilleur modèle : époque {best_epoch}")
print(f"Meilleure précision validation : {best_val_accuracy:.1%}")
print(f"Checkpoint de cette run : {run_model_path}")
print(f"Modèle courant pour l'appli sauvegardé dans : {model_path}")