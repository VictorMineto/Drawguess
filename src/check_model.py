import numpy as np
import torch

from src.model import TinyCNN


data = np.load("data/processed/val.npz")
images = data["images"]
labels = data["labels"]

checkpoint = torch.load(
    "models/drawguess_cnn.pth",
    map_location="cpu",
)

class_names = checkpoint["class_names"]
model = TinyCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

for label, class_name in enumerate(class_names):
    index = np.flatnonzero(labels == label)[0]
    image = images[index]

    x = torch.from_numpy(image.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        scores = model(x)
        probabilities = torch.softmax(scores, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities[0], k=3)

    print(f"\nDessin attendu : {class_name}")
    for probability, prediction in zip(top_probabilities, top_indices):
        print(f"  {class_names[prediction.item()]} : {probability.item():.1%}")