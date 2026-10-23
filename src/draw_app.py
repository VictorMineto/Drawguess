import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.model import TinyCNN


torch.set_num_threads(1)

checkpoint = torch.load(
    "models/drawguess_cnn.pth",
    map_location="cpu",
)

class_names = checkpoint["class_names"]
model = TinyCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

window = tk.Tk()
window.title("DrawGuess")


canvas_size = 300
stroke_width = 10
radius = stroke_width // 2

canvas_widget = tk.Canvas(
    window,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    highlightthickness=0,
)
canvas_widget.pack(padx=10, pady=10)

drawing = Image.new("L", (canvas_size, canvas_size), 0)
image_draw = ImageDraw.Draw(drawing)
last_point: list[tuple[int, int] | None] = [None]

result = tk.Label(window, text="Dessine un objet, puis clique sur Deviner.")
result.pack(padx=10, pady=5)


def start_draw(event):
    last_point[0] = (event.x, event.y)
    canvas_widget.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white",
    )
    image_draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        ),
        fill=255,
    )


def continue_draw(event):
    if last_point[0] is None:
        start_draw(event)
        return

    x0, y0 = last_point[0]
    x1, y1 = event.x, event.y

    canvas_widget.create_line(
        x0,
        y0,
        x1,
        y1,
        fill="white",
        width=stroke_width,
        capstyle=tk.ROUND,
        smooth=True,
    )
    image_draw.line(
        (x0, y0, x1, y1),
        fill=255,
        width=stroke_width,
    )
    image_draw.ellipse(
        (
            x1 - radius,
            y1 - radius,
            x1 + radius,
            y1 + radius,
        ),
        fill=255,
    )

    last_point[0] = (x1, y1)


def prepare_drawing(image):
    bounding_box = image.getbbox()

    if bounding_box is None:
        return None

    cropped = image.crop(bounding_box)
    width, height = cropped.size
    scale = 20 / max(width, height)
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
    prepared = Image.new("L", (28, 28), 0)
    prepared.paste(
        resized,
        ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2),
    )

    return prepared


def guess():
    prepared = prepare_drawing(drawing)

    if prepared is None:
        result.config(text="Dessine d'abord quelque chose.")
        return

    Path("outputs").mkdir(exist_ok=True)
    prepared.save("outputs/gui_preprocessed.png")

    pixels = np.asarray(prepared, dtype=np.float32).copy()
    x = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=3)

    lines = ["Mes trois propositions :"]
    for probability, index in zip(top_probabilities, top_indices):
        lines.append(
            f"{class_names[index.item()]} : {probability.item():.1%}"
        )

    result.config(text="\n".join(lines))


def clear():
    canvas_widget.delete("all")
    drawing.paste(0, (0, 0, canvas_size, canvas_size))
    last_point[0] = None
    result.config(text="Dessine un objet, puis clique sur Deviner.")


canvas_widget.bind("<Button-1>", start_draw)
canvas_widget.bind("<B1-Motion>", continue_draw)

buttons = tk.Frame(window)
buttons.pack(pady=5)

tk.Button(buttons, text="Deviner", command=guess).pack(side="left", padx=5)
tk.Button(buttons, text="Effacer", command=clear).pack(side="left", padx=5)

window.mainloop()