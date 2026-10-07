from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fishdet.cli import DEFAULT_WEIGHTS, build_parser, main  # noqa: E402
from fishdet.model import build_model, save_weights  # noqa: E402
from fishdet.train import TrainConfig  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_cli_defaults_follow_the_training_config():
    args = build_parser().parse_args(["train"])
    defaults = TrainConfig()
    assert (args.n_train, args.epochs, args.lr, args.flip) == (defaults.n_train, defaults.epochs, defaults.lr, defaults.flip)
    assert args.out == DEFAULT_WEIGHTS
    assert build_parser().parse_args(["train", "--n-train", "all"]).n_train is None


def test_predicting_without_weights_is_a_clean_error(tmp_path, capsys):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"")
    assert main(["predict", str(photo), "--weights", str(tmp_path / "missing.pt")]) == 1
    assert "fishdet train" in capsys.readouterr().err


def test_app_asks_for_weights_when_there_are_none(tmp_path, monkeypatch):
    monkeypatch.setenv("FISHDET_WEIGHTS", str(tmp_path / "missing.pt"))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert "No trained weights" in app.error[0].value


def test_app_waits_for_a_photo_once_the_weights_exist(tmp_path, monkeypatch):
    weights = save_weights(build_model(pretrained=False), tmp_path / "detector.pt")
    monkeypatch.setenv("FISHDET_WEIGHTS", str(weights))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert any("Drop a photo" in i.value for i in app.info)
