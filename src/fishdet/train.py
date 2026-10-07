"""Fine-tuning loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .data import FishDataset, collate_fn, list_images
from .model import build_model

LOSS_NAMES = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


@dataclass
class TrainConfig:
    n_train: int | None = 1000  # random training photos (None = all of them)
    epochs: int = 3
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    unfreeze_backbone: bool = False
    lr_backbone: float = 0.001  # learning rate of the backbone when it is unfrozen
    flip: bool = False  # mirror photos at random (boxes follow)
    seed: int = 42
    workers: int = 2
    pretrained: bool = True  # False: random weights, no download (tests)


def train(dataset_root: str | Path, config: TrainConfig | None = None,
          progress: Callable[[int, int, float], None] | None = None):
    """Fine-tune the detector on the training split. Returns `(model, loss_history)`.

    In training mode the detector returns four losses instead of predictions: how well it classifies
    boxes, how well it places them, and the same two things for the region proposals that feed it.
    They are summed and back-propagated. `loss_history` holds the average of each, per epoch.
    """
    config = config or TrainConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    paths = list_images(dataset_root, "train")
    if config.n_train is not None:
        paths = random.sample(paths, min(config.n_train, len(paths)))
    loader = DataLoader(
        FishDataset(paths, flip=config.flip), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=config.workers, persistent_workers=config.workers > 0,
    )

    model = build_model(pretrained=config.pretrained, freeze_backbone=not config.unfreeze_backbone)
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    groups = [{"params": heads, "lr": config.lr}]
    if config.unfreeze_backbone:
        groups.append({"params": list(model.backbone.parameters()), "lr": config.lr_backbone})
    optimizer = torch.optim.SGD(groups, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    history = {name: [] for name in LOSS_NAMES}
    for epoch in range(config.epochs):
        model.train()
        running, batches = {name: 0.0 for name in LOSS_NAMES}, 0
        for images, targets in loader:
            losses = model(list(images), list(targets))
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name in LOSS_NAMES:
                running[name] += losses[name].item()
            batches += 1
        for name in LOSS_NAMES:
            history[name].append(running[name] / batches)
        if progress:
            progress(epoch + 1, config.epochs, sum(history[name][-1] for name in LOSS_NAMES))
    return model.eval(), history
