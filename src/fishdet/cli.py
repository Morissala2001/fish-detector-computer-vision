"""Command line: `fishdet download | train | evaluate | predict`."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .data import SPLITS, download_dataset, find_dataset_root, list_images
from .evaluate import evaluate
from .model import load_weights, save_weights
from .predict import detect, draw_detections
from .train import TrainConfig, train

DEFAULT_WEIGHTS = Path("models/fish_detector.pt")


def _dataset_root(args) -> Path:
    return find_dataset_root(args.data) if args.data else download_dataset()


def _count(value: str):
    return None if value == "all" else int(value)


def cmd_download(args) -> None:
    print(download_dataset())


def cmd_train(args) -> None:
    config = TrainConfig(
        n_train=args.n_train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, flip=args.flip,
        unfreeze_backbone=args.unfreeze_backbone, seed=args.seed, workers=args.workers,
    )
    root = _dataset_root(args)
    print(f"Training: {config}")
    model, _ = train(root, config, progress=lambda e, n, loss: print(f"  epoch {e}/{n}  total loss {loss:.3f}"))
    print(f"Weights saved to {save_weights(model, args.out)}")
    print(f"Next: fishdet evaluate --weights {args.out}")


def cmd_evaluate(args) -> None:
    paths = list_images(_dataset_root(args), args.split)
    if args.n:
        paths = random.Random(0).sample(paths, min(args.n, len(paths)))
    result = evaluate(load_weights(args.weights), paths, score_thresh=args.threshold)
    print(f"{result['photos']} photos of the '{args.split}' split")
    print(f"  mAP (IoU 0.50:0.95): {result['map']:.3f} | mAP@0.50: {result['map_50']:.3f} | mAP@0.75: {result['map_75']:.3f}")
    print(f"  confident detections on {result['photo_detection_rate']:.1%} of photos; "
          f"main species right on {result['photo_species_accuracy']:.1%} of those")
    print(f"  {result['ms_per_photo']:.0f} ms per photo on this machine")
    print("  AP per species: " + ", ".join(f"{s} {ap:.2f}" for s, ap in sorted(result["ap_per_species"].items())))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")


def cmd_predict(args) -> None:
    model = load_weights(args.weights)
    detections = detect(model, args.photo, args.threshold)
    if not detections:
        print(f"No fish detected above {args.threshold:.0%} confidence.")
    for d in detections:
        print(f"  {d.species:<24} {d.score:6.1%}  box {tuple(round(v) for v in d.box)}")
    if args.save:
        from PIL import Image

        draw_detections(Image.open(args.photo), detections).save(args.save)
        print(f"Annotated photo saved to {args.save}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fishdet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data(p):
        p.add_argument("--data", type=Path, default=None, help="dataset folder (default: download from Kaggle)")

    p = sub.add_parser("download", help="download the Kaggle dataset (about 340 MB)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="fine-tune the detector")
    add_data(p)
    defaults = TrainConfig()
    p.add_argument("--n-train", type=_count, default=defaults.n_train, help="training photos, or 'all'")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--flip", action="store_true", default=defaults.flip, help="mirror photos at random")
    p.add_argument("--unfreeze-backbone", action="store_true", default=defaults.unfreeze_backbone)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--workers", type=int, default=defaults.workers)
    p.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="measure mAP on a split")
    add_data(p)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--n", type=int, default=None, help="evaluate on a fixed random subset of this size")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=None, help="write the metrics as JSON")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="detect the fish of a photo")
    p.add_argument("photo", type=Path)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save", type=Path, default=None, help="save the photo with boxes drawn")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
