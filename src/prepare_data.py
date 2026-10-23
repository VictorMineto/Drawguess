from pathlib import Path
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ["airplane", "apple", "cat", "star", "tree"]
SAMPLES_PER_CLASS = 500
RNG = np.random.default_rng(42)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": [[], []],
    "val": [[], []],
    "test": [[], []],
}

sample_images = []

for label, class_name in enumerate(CLASSES):
    file_path = RAW_DIR / f"{class_name}.npy"

    if not file_path.exists():
        url = (
            "https://storage.googleapis.com/quickdraw_dataset/"
            f"full/numpy_bitmap/{class_name}.npy"
        )
        print(f"Téléchargement de {class_name}...")
        urllib.request.urlretrieve(url, file_path)

    images = np.load(file_path, mmap_mode="r")
    count = min(SAMPLES_PER_CLASS, len(images))
    indices = RNG.choice(len(images), size=count, replace=False)
    images = np.asarray(images[indices], dtype=np.uint8)

    #sample_images.append(images[0])
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    elif images.shape[1:] != (28, 28):
        raise ValueError(f"Forme inattendue pour les images : {images.shape}")

    sample_images.append(images[0])

    indices = RNG.permutation(count)
    train_end = int(0.70 * count)
    val_end = int(0.85 * count)

    split_indices = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, split_ids in split_indices.items():
        splits[split_name][0].append(images[split_ids])
        splits[split_name][1].append(
            np.full(len(split_ids), label, dtype=np.int64)
        )

for split_name, (image_groups, label_groups) in splits.items():
    images = np.concatenate(image_groups)
    labels = np.concatenate(label_groups)
    order = RNG.permutation(len(images))

    np.savez_compressed(
        PROCESSED_DIR / f"{split_name}.npz",
        images=images[order],
        labels=labels[order],
    )

    print(f"{split_name}: {len(images)} dessins")

fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 3))

for ax, image, class_name in zip(axes, sample_images, CLASSES):
    ax.imshow(image, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")

fig.tight_layout()
fig.savefig(PROCESSED_DIR / "sample_grid.png")
plt.close(fig)

with open(PROCESSED_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(CLASSES, f, ensure_ascii=False, indent=2)

print("Préparation terminée.")
print("Vérifie l'image : data/processed/sample_grid.png")