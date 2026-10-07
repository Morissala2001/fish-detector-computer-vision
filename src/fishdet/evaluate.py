"""Measuring a detector instead of judging it by a few photos."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from .data import FishDataset, collate_fn
from .species import species_of


def evaluate(model, image_paths: list[Path], batch_size: int = 4, workers: int = 0, score_thresh: float = 0.5) -> dict:
    """Score a detector on labelled photos.

    - `map`: mean Average Precision, the standard metric of detection. A predicted box counts as
      right when it overlaps the true box enough (IoU) and names the right species; `map` averages
      over several overlap thresholds from 0.5 to 0.95, `map_50` only asks for 0.5.
    - `photo_species_accuracy`: on photos where the detector found a fish (confidence above
      `score_thresh`), how often its most confident box names the photo's main species.
    - `photo_detection_rate`: share of photos with at least one confident box.
    """
    loader = DataLoader(FishDataset(image_paths), batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=workers)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    model.eval()
    photos = detected = correct = 0
    start = time.time()
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(list(images))
            truths = [{"boxes": t["boxes"].as_subclass(torch.Tensor), "labels": t["labels"]} for t in targets]
            metric.update(list(predictions), truths)
            for prediction, truth in zip(predictions, truths):
                photos += 1
                confident = prediction["scores"] >= score_thresh
                if not confident.any() or len(truth["labels"]) == 0:
                    continue
                detected += 1
                best = prediction["labels"][confident][prediction["scores"][confident].argmax()].item()
                correct += best == Counter(truth["labels"].tolist()).most_common(1)[0][0]
    seconds = time.time() - start
    result = metric.compute()
    return {
        "photos": photos,
        "map": result["map"].item(),
        "map_50": result["map_50"].item(),
        "map_75": result["map_75"].item(),
        "ap_per_species": {species_of(c): round(ap, 4) for c, ap in
                           zip(result["classes"].tolist(), result["map_per_class"].tolist())},
        "photo_species_accuracy": correct / detected if detected else 0.0,
        "photo_detection_rate": detected / photos if photos else 0.0,
        "ms_per_photo": seconds / photos * 1000 if photos else 0.0,
    }
