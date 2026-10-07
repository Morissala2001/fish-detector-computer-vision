import pytest
import torch
from torchvision.transforms import v2 as T

from fishdet.data import (FishDataset, collate_fn, dominant_species, find_dataset_root, label_path, list_images,
                          read_yolo_boxes)
from fishdet.species import LABEL_NAMES, SPECIES, species_of


def test_species_labels_are_shifted_by_one_for_the_background():
    assert len(SPECIES) == 13 and len(LABEL_NAMES) == 14
    assert species_of(1) == "AngelFish" and species_of(13) == "ZebraFish" and species_of(4) == "ClownFish"
    for bad in (0, 14, -1):
        with pytest.raises(ValueError):
            species_of(bad)


def test_yolo_line_becomes_a_pixel_box(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("3 0.5 0.5 0.4 0.3\n")
    boxes, classes = read_yolo_boxes(label, width=200, height=100)
    assert boxes == [[60.0, 35.0, 140.0, 65.0]] and classes == [3]


def test_malformed_lines_and_missing_files_give_no_box(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("not a label\n1 0.5 0.5 0.2\n\n2 0.5 0.5 0.2 0.2\n")
    assert read_yolo_boxes(label, 100, 100)[1] == [2]
    assert read_yolo_boxes(tmp_path / "missing.txt", 100, 100) == ([], [])


def test_dominant_species(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("4 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.1 0.1\n1 0.7 0.7 0.1 0.1\n")
    assert dominant_species(label) == "GoldFish"
    label.write_text("")
    assert dominant_species(label) is None


def test_listing_and_label_paths(dataset_root):
    images = list_images(dataset_root, "train")
    assert [p.name for p in images] == [f"train_{i}.jpg" for i in range(4)]
    assert label_path(images[0]) == dataset_root / "train" / "labels" / "train_0.txt"
    with pytest.raises(FileNotFoundError):
        list_images(dataset_root, "nope")


def test_dataset_root_is_found_one_level_down(dataset_root):
    assert find_dataset_root(dataset_root) == dataset_root
    assert find_dataset_root(dataset_root.parent) == dataset_root
    with pytest.raises(FileNotFoundError):
        find_dataset_root(dataset_root / "train")


def test_dataset_item(dataset_root):
    image, target = FishDataset(list_images(dataset_root, "train"))[1]
    assert image.shape == (3, 120, 160) and image.dtype == torch.float32 and 0 <= image.min() <= image.max() <= 1
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [2]  # species id 1 (BlueTang) + 1
    assert [round(v) for v in target["boxes"][0].tolist()] == [48, 42, 112, 78]


def test_a_photo_without_fish_has_empty_but_well_typed_targets(dataset_root):
    _, target = FishDataset(list_images(dataset_root, "train"))[3]
    assert target["boxes"].shape == (0, 4) and target["labels"].dtype == torch.int64 and len(target["labels"]) == 0


def test_flipping_moves_the_boxes_with_the_photo(dataset_root):
    dataset = FishDataset(list_images(dataset_root, "train"), flip=True)
    dataset.transforms = T.Compose([T.RandomHorizontalFlip(1.0), T.ToDtype(torch.float32, scale=True)])
    _, flipped = dataset[1]
    _, original = FishDataset(list_images(dataset_root, "train"))[1]
    x1, _, x2, _ = original["boxes"][0].tolist()
    assert flipped["boxes"][0, 0].item() == pytest.approx(160 - x2)
    assert flipped["boxes"][0, 2].item() == pytest.approx(160 - x1)


def test_collate_keeps_photos_as_tuples(dataset_root):
    dataset = FishDataset(list_images(dataset_root, "train"))
    images, targets = collate_fn([dataset[0], dataset[1]])
    assert len(images) == len(targets) == 2
