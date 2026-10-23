import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.model import TinyCNN


data = np.load("data/processed/test.npz")

images = torch.from_numpy(data["images"].astype(np.float32)).unsqueeze(1) / 255.0
labels = torch.from_numpy(data["labels"].astype(np.int64))

test_loader = DataLoader(
    TensorDataset(images, labels),
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

with open("data/processed/classes.json", encoding="utf-8") as f:
    class_names = json.load(f)

checkpoint = torch.load(
    "models/drawguess_cnn.pth",
    map_location="cpu",
)

model = TinyCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

accuracy = accuracy_score(all_labels, all_predictions)

print(f"Précision sur le test : {accuracy:.1%}")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0,
    )
)