import pytest
import torch
from PIL import Image
from conftest import FakeDetector

from fishdet.evaluate import evaluate
from fishdet.data import FishDataset, list_images
from fishdet.model import build_model, load_weights, save_weights
from fishdet.predict import Detection, detect, draw_detections
from fishdet.train import LOSS_NAMES, TrainConfig, train


def test_new_head_has_one_output_per_species_plus_background():
    model = build_model(pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 14


def test_the_backbone_is_frozen_by_default_and_can_be_unfrozen():
    frozen = build_model(pretrained=False)
    assert all(not p.requires_grad for p in frozen.backbone.parameters())
    assert any(p.requires_grad for p in frozen.roi_heads.parameters())
    assert all(p.requires_grad for p in build_model(pretrained=False, freeze_backbone=False).backbone.parameters())


def test_saved_weights_reload_to_the_same_detector(tmp_path):
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    path = save_weights(model, tmp_path / "w" / "detector.pt")
    reloaded = load_weights(path)
    photo = [torch.rand(3, 120, 160)]
    with torch.no_grad():
        a, b = model(photo)[0], reloaded(photo)[0]
    assert torch.equal(a["boxes"], b["boxes"]) and torch.equal(a["scores"], b["scores"])
    with pytest.raises(FileNotFoundError, match="fishdet train"):
        load_weights(tmp_path / "missing.pt")


def test_training_runs_and_returns_finite_losses(dataset_root):
    config = TrainConfig(n_train=None, epochs=2, batch_size=2, workers=0, pretrained=False)
    seen = []
    model, history = train(dataset_root, config, progress=lambda epoch, n, loss: seen.append((epoch, n)))
    assert not model.training
    assert set(history) == set(LOSS_NAMES) and all(len(v) == 2 for v in history.values())
    assert all(torch.isfinite(torch.tensor(v)).all() for v in history.values())
    assert seen == [(1, 2), (2, 2)]


def test_n_train_limits_the_photos_used(dataset_root, monkeypatch):
    used = []
    import fishdet.train as train_module

    original = train_module.FishDataset
    monkeypatch.setattr(train_module, "FishDataset", lambda paths, flip=False: used.append(len(paths)) or original(paths, flip))
    train(dataset_root, TrainConfig(n_train=2, epochs=1, batch_size=2, workers=0, pretrained=False))
    assert used == [2]


def test_detections_are_named_sorted_and_filtered():
    prediction = {"boxes": torch.tensor([[1., 2., 30., 40.], [5., 5., 20., 20.], [0., 0., 9., 9.]]),
                  "labels": torch.tensor([4, 13, 1]), "scores": torch.tensor([0.6, 0.9, 0.3])}
    found = detect(FakeDetector([prediction]), Image.new("RGB", (64, 48)), score_thresh=0.5)
    assert [(d.species, d.score) for d in found] == [("ZebraFish", pytest.approx(0.9)), ("ClownFish", pytest.approx(0.6))]
    assert found[1].box == (1.0, 2.0, 30.0, 40.0)


def test_drawing_keeps_the_photo_size():
    photo = Image.new("RGB", (160, 120), "white")
    drawn = draw_detections(photo, [Detection("ClownFish", 0.9, (10, 10, 80, 60))])
    assert drawn.size == photo.size and drawn.getpixel((10, 30)) != (255, 255, 255)  # a box edge was drawn
    assert photo.getpixel((10, 30)) == (255, 255, 255)  # the original is untouched


def test_the_label_is_readable_on_a_large_photo():
    """Regression test: with a fixed small font the label was invisible on a 4000 px phone photo."""
    photo = Image.new("RGB", (2000, 1500), "white")
    drawn = draw_detections(photo, [Detection("ClownFish", 0.9, (100, 300, 900, 800))])
    clownfish_color = (245, 130, 49)
    assert drawn.getpixel((101, 298)) == clownfish_color  # padding of a label sitting right above the box
    label_pixels = sum(drawn.getpixel((x, 270)) == clownfish_color for x in range(100, 400))
    assert label_pixels > 150  # the coloured label is wide enough to hold the text


def _perfect(dataset_root):
    return [{"boxes": t["boxes"].as_subclass(torch.Tensor), "labels": t["labels"], "scores": torch.ones(len(t["labels"]))}
            for _, t in FishDataset(list_images(dataset_root, "valid"))]


def test_a_perfect_detector_scores_one(dataset_root):
    pytest.importorskip("pycocotools")
    paths = list_images(dataset_root, "valid")
    result = evaluate(FakeDetector(_perfect(dataset_root)), paths, batch_size=2)
    assert result["map"] == pytest.approx(1.0) and result["map_50"] == pytest.approx(1.0)
    assert result["photo_species_accuracy"] == 1.0 and result["photo_detection_rate"] == 1.0


def test_a_detector_naming_the_wrong_species_scores_zero(dataset_root):
    pytest.importorskip("pycocotools")
    wrong = [dict(p, labels=p["labels"] % 13 + 1) for p in _perfect(dataset_root)]
    result = evaluate(FakeDetector(wrong), list_images(dataset_root, "valid"), batch_size=2)
    assert result["map"] == pytest.approx(0.0) and result["photo_species_accuracy"] == 0.0
