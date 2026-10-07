# 🐠 Fish detector

[![tests](https://github.com/Morissala2001/fish-detector-computer-vision/actions/workflows/tests.yml/badge.svg)](https://github.com/Morissala2001/fish-detector-computer-vision/actions/workflows/tests.yml)

*Detect the fish in an aquarium photo and name their species.*

A pre-trained object detector (Faster R-CNN) fine-tuned on 13 aquarium species. Given a photo, it answers two questions for every fish: **where is it** (a box) and **which species is it**. The project has a **web app** (drop a photo, see the boxes), a **CLI** (`fishdet`) and a small **library** (`src/fishdet`) with a built-in evaluation.

Why a detector and not a classifier? A photo of an aquarium rarely shows one fish. The detector found in the COCO dataset knows 91 everyday categories and no fish at all: on a close-up goldfish it sees a *bird* with 98% confidence. Fine-tuning teaches it the new species while keeping what it already knows about finding objects.

## Results

Evaluated on the **700 photos of the test split**, never used for training or tuning. The detector was fine-tuned in about 5 minutes on a laptop CPU, on 1,000 random training photos for 3 epochs (the quick setting of this project).

| Metric | Test split |
|---|---|
| **mAP** (IoU 0.50:0.95) | **0.297** |
| mAP@0.50 | 0.555 |
| mAP@0.75 | 0.287 |
| Photos with a confident detection (score ≥ 0.5) | 84.0% |
| Main species correct, on those photos | 91.5% |
| Speed | 57 ms per photo (Intel i7-13700H, CPU only) |

mAP is the standard metric of detection: a predicted box counts as right when it overlaps the true box enough (IoU) **and** names the right species, averaged over several overlap thresholds. The last two rows are easier to read: when the detector finds a fish, it names the main species of the photo correctly 9 times out of 10.

AP per species, with the number of boxes in the test split (an AP computed on a handful of boxes is not meaningful):

| Species | Test boxes | AP | | Species | Test boxes | AP |
|---|---|---|---|---|---|---|
| Gourami | 102 | 0.59 | | YellowTang | 65 | 0.41 |
| ZebraFish | 123 | 0.47 | | GoldFish | 83 | 0.33 |
| BlueTang | 168 | 0.42 | | PlatyFish | 76 | 0.10 |
| AngelFish | 227 | 0.41 | | YellowCichlid | 86 | 0.06 |
| ClownFish | 136 | 0.40 | | RibbonedSweetlips | 8 | 0.29 |
| MorishIdol | 14 | 0.21 | | ThreeStripedDamselfish | 14 | 0.16 |
| ButterflyFish | 3 | 0.00 | | | | |

## Run it

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13. A GPU is not needed.

```bash
git clone https://github.com/Morissala2001/fish-detector-computer-vision.git
cd fish-detector-computer-vision
uv sync --all-extras
```

**1. Get the data and train** (the dataset is about 340 MB; the COCO weights, 74 MB, are downloaded by torchvision on first use)

```bash
uv run fishdet download   # Kaggle dataset, cached outside the repository
uv run fishdet train      # about 5 minutes on a laptop CPU, writes models/fish_detector.pt
```

**2. Measure and use**

```bash
uv run fishdet evaluate                      # mAP on the test split
uv run fishdet predict my_photo.jpg --save annotated.jpg
uv run streamlit run app.py                  # web app: upload a photo or use the camera
```

`fishdet train` also accepts `--n-train all`, `--epochs`, `--flip` (mirror photos) and `--unfreeze-backbone`; only the quick default setting was measured here.

**Tests** (offline: a few tiny synthetic photos and a randomly initialised detector)

```bash
uv run pytest
```

## How it works

1. **Data** (`data.py`): the dataset is in YOLO format, one text line per fish: `class x_center y_center width height`, as fractions of the photo size. They are converted to pixel boxes.
2. **Dataset object**: `FishDataset` is a *recipe* that builds sample `i` on demand (photo and boxes), so photos are read one at a time instead of loaded in memory. Species ids are shifted by one because torchvision reserves label 0 for the background; a photo without fish gets an empty box list with the right integer type, otherwise the model refuses it. With `--flip`, boxes are mirrored together with the photo.
3. **Model** (`model.py`): `fasterrcnn_mobilenet_v3_large_320_fpn`, a light Faster R-CNN, pre-trained on COCO. Its backbone is frozen (it already knows how to see) and the box-classification head is replaced by a new one with 14 outputs: 13 species plus the background.
4. **Training** (`train.py`): in training mode the detector returns four losses (how well it classifies and places boxes, and the same for the region proposals that feed it). They are summed and back-propagated with SGD.
5. **Evaluation** (`evaluate.py`): mAP with `torchmetrics`, plus the two readable numbers above.
6. **Prediction** (`predict.py`): photos of any resolution are accepted; boxes come back in the original pixels. The weights are loaded without any download.

## Limits

- **A quick fine-tune.** 1,000 photos out of 6,842 and 3 epochs: the loss was still falling (1.28, 0.91, 0.76). Longer training, all the photos, mirroring and an unfrozen backbone are supported by the code but were not measured here, and would most likely help.
- **Some species are poorly separated**: YellowCichlid (AP 0.06) and PlatyFish (0.10) have many examples and still score low.
- **Noisy per-species numbers.** The test split holds only 3 ButterflyFish boxes against 1,929 in the training split: the species mix differs between the splits, so read those APs with care.
- **Photos are not independent.** The dataset is a Roboflow export in which several training photos are augmented versions of the same source photo (visible in the file names).
- **16% of test photos have no detection at confidence 0.5**; lowering the threshold in the app finds more fish and more mistakes.

## Structure

```
├── app.py                   # Streamlit web app
├── src/fishdet/
│   ├── species.py           # the 13 species, label convention
│   ├── data.py              # YOLO parsing, FishDataset, Kaggle download
│   ├── model.py             # detector, save / load weights
│   ├── train.py             # fine-tuning loop
│   ├── evaluate.py          # mAP and photo-level accuracy
│   ├── predict.py           # detect and draw
│   └── cli.py               # the `fishdet` command
└── tests/
```

## Dataset

[Fish dataset](https://www.kaggle.com/datasets/mahmoodyousaf/fish-dataset) by Mahmood Yousaf on Kaggle, exported from the Roboflow project *fish-detection-fztlb* (workspace *zehra-acer*), licensed **CC BY 4.0**. It is not included in this repository: `fishdet download` fetches it into your local Kaggle cache.

## Ideas for improvement

- Train on all the photos for more epochs, with mirroring, and measure each change on the test split.
- Add an "unknown fish" answer, for example by calibrating the confidence or training with photos of other species.
- Compare with a heavier detector (`fasterrcnn_resnet50_fpn`) or a YOLO model.
- Share the trained weights as a release asset so that the app works without training.

## License

[MIT](LICENSE). The dataset keeps its own CC BY 4.0 license.
