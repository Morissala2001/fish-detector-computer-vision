"""The detector: a COCO-pre-trained Faster R-CNN (MobileNetV3, 320 px) with a new species head."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_320_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .species import SPECIES


def build_model(num_species: int = len(SPECIES), pretrained: bool = True, freeze_backbone: bool = True):
    """A detector ready to be fine-tuned on `num_species` species.

    The backbone ("the part that learned to see") comes pre-trained on COCO. By default it is frozen
    and only the heads are trained. The box-classification head is replaced by a new one with one
    output per species, plus one for the background.

    `pretrained=False` builds the same architecture with random weights and downloads nothing: used
    by the tests and to load a fine-tuned checkpoint.
    """
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, weights_backbone=None)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not freeze_backbone
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_species + 1)
    return model


def save_weights(model: torch.nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_weights(path: str | Path, num_species: int = len(SPECIES), device: str = "cpu"):
    """Rebuild the detector and load fine-tuned weights, in evaluation mode (no download needed)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}. Train them with `fishdet train`.")
    model = build_model(num_species, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
