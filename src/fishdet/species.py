"""The 13 species of the dataset and the label convention of torchvision detectors."""

# Order of the class ids in the dataset's YOLO label files.
SPECIES = [
    "AngelFish", "BlueTang", "ButterflyFish", "ClownFish", "GoldFish", "Gourami", "MorishIdol",
    "PlatyFish", "RibbonedSweetlips", "ThreeStripedDamselfish", "YellowCichlid", "YellowTang", "ZebraFish",
]

# torchvision reserves label 0 for "no object": species number i has label i + 1.
BACKGROUND = "background"
LABEL_NAMES = [BACKGROUND] + SPECIES


def species_of(label: int) -> str:
    """Name of the species of a detector label (1 to 13)."""
    if not 1 <= label <= len(SPECIES):
        raise ValueError(f"label {label} is not a species (expected 1 to {len(SPECIES)}; 0 is the background)")
    return LABEL_NAMES[label]
