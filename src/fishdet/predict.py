"""Using a detector on one photo, and drawing the result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.v2 import functional as F

from .species import SPECIES, species_of


@dataclass(frozen=True)
class Detection:
    species: str
    score: float  # confidence between 0 and 1
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the original photo


def detect(model, image: Image.Image | str | Path, score_thresh: float = 0.5) -> list[Detection]:
    """Find the fish of a photo of any resolution, most confident first.

    The detector resizes photos itself and returns boxes in the original photo's pixels. It only knows
    the 13 species it was trained on: a fish of another species gets the name of the closest one.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    batch = F.to_dtype(F.pil_to_tensor(image.convert("RGB")), torch.float32, scale=True).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(batch)[0]
    keep = output["scores"] > score_thresh
    detections = [
        Detection(species_of(int(label)), float(score), tuple(float(v) for v in box))
        for box, label, score in zip(output["boxes"][keep], output["labels"][keep], output["scores"][keep])
    ]
    return sorted(detections, key=lambda d: d.score, reverse=True)


PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000",
           "#808000", "#000075", "#f032e6", "#469990", "#bfa100"]


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """A copy of the photo with a box and a label around each detected fish."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    biggest_side = max(canvas.size)
    line_width = max(2, round(biggest_side / 250))
    font = ImageFont.load_default(size=max(14, round(biggest_side / 40)))  # readable on a 4000 px phone photo too
    for d in detections:
        color = PALETTE[SPECIES.index(d.species) % len(PALETTE)]
        draw.rectangle(d.box, outline=color, width=line_width)
        text = f"{d.species} {d.score:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        pad = max(2, (bottom - top) // 5)
        label_width, label_height = right - left + 2 * pad, bottom - top + 2 * pad
        x = min(max(d.box[0], 0), canvas.width - label_width)
        y = d.box[1] - label_height if d.box[1] >= label_height else d.box[1]  # above the box, or inside if no room
        draw.rectangle((x, y, x + label_width, y + label_height), fill=color)
        draw.text((x + pad - left, y + pad - top), text, fill="white", font=font)
    return canvas
