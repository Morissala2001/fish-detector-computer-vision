"""Shared helpers. No test downloads anything: models are built without pre-trained weights and the
dataset is a handful of tiny synthetic photos."""

import numpy as np
import pytest
import torch
from PIL import Image


def make_dataset(root, n_train=4, n_valid=2, n_test=2):
    """A miniature YOLO dataset: noise photos of 160x120 pixels, one fish in the middle of each.

    Photo i of a split shows species `i % 13`. The last training photo has an empty label file.
    """
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(root / split / "images" / f"{split}_{i}.jpg")
            empty = split == "train" and i == n - 1
            (root / split / "labels" / f"{split}_{i}.txt").write_text("" if empty else f"{i % 13} 0.5 0.5 0.4 0.3\n")
    return root


class FakeDetector(torch.nn.Module):
    """Returns prepared predictions, one dict per photo, in the order the photos are given."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.seen = 0

    def forward(self, images, targets=None):
        out = self.predictions[self.seen:self.seen + len(images)]
        self.seen += len(images)
        return out


@pytest.fixture
def dataset_root(tmp_path):
    return make_dataset(tmp_path / "fish")
