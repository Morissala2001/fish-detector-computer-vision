"""The dataset: YOLO-format photos and labels, as a PyTorch `Dataset`."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2 as T

from .species import SPECIES

KAGGLE_DATASET = "mahmoodyousaf/fish-dataset"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download_dataset() -> Path:
    """Download the Kaggle dataset (about 340 MB) into the local Kaggle cache, once. Returns its root."""
    import kagglehub

    return find_dataset_root(kagglehub.dataset_download(KAGGLE_DATASET))


def find_dataset_root(path: str | Path) -> Path:
    """The folder that contains `train/images`, `valid/images`... (the path itself or one level down)."""
    path = Path(path)
    for candidate in [path, *sorted(p for p in path.iterdir() if p.is_dir())]:
        if (candidate / "train" / "images").is_dir():
            return candidate
    raise FileNotFoundError(f"No train/images folder found in {path}")


def list_images(root: str | Path, split: str) -> list[Path]:
    """Photos of a split, sorted by name."""
    folder = Path(root) / split / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found (splits are {SPLITS})")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(image_path: str | Path) -> Path:
    """`.../images/photo.jpg` -> `.../labels/photo.txt`."""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_yolo_boxes(path: str | Path, width: int, height: int) -> tuple[list[list[float]], list[int]]:
    """Parse a YOLO label file into `[x1, y1, x2, y2]` pixel boxes and species ids (0 to 12).

    A YOLO line is `class x_center y_center width height`, the last four numbers being fractions
    of the image size. Missing files and malformed lines give no box.
    """
    path = Path(path)
    boxes, classes = [], []
    if not path.exists():
        return boxes, classes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append([(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height])
        classes.append(class_id)
    return boxes, classes


def dominant_species(path: str | Path) -> str | None:
    """The most frequent species of a label file, or None for a photo without fish."""
    _, classes = read_yolo_boxes(path, 1, 1)
    return SPECIES[Counter(classes).most_common(1)[0][0]] if classes else None


class FishDataset(Dataset):
    """A recipe that builds sample `i` on demand: a photo and everything known about it.

    Each item is `(image, target)`: a float tensor in [0, 1] and a dict with the `boxes` (pixels,
    XYXY) and `labels` (species id + 1, because label 0 is the background). With `flip=True` a
    photo is mirrored half of the time, and its boxes move with it.
    """

    def __init__(self, image_paths: list[Path], flip: bool = False):
        self.image_paths = [Path(p) for p in image_paths]
        steps = ([T.RandomHorizontalFlip(0.5)] if flip else []) + [T.ToDtype(torch.float32, scale=True)]
        self.transforms = T.Compose(steps)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int):
        path = self.image_paths[i]
        image = read_image(str(path))[:3]  # drop a possible alpha channel
        height, width = image.shape[-2:]
        boxes, classes = read_yolo_boxes(label_path(path), width, height)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), format="XYXY", canvas_size=(height, width)
            ),
            # int64 is required: a photo without fish gives an empty list, which would otherwise be a float tensor
            "labels": torch.tensor([c + 1 for c in classes], dtype=torch.int64),
        }
        return self.transforms(tv_tensors.Image(image), target)


def collate_fn(batch):
    """Detection photos differ in size and in number of boxes, so a batch stays a pair of tuples."""
    return tuple(zip(*batch))
