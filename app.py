"""Web app: drop a photo of an aquarium and see which fish the detector finds.

    uv run --extra app streamlit run app.py

It needs trained weights (`uv run fishdet train`), by default in models/fish_detector.pt.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from fishdet.model import load_weights
from fishdet.predict import detect, draw_detections
from fishdet.species import SPECIES

WEIGHTS = Path(os.environ.get("FISHDET_WEIGHTS", "models/fish_detector.pt"))

st.set_page_config(page_title="Fish detector", page_icon=":material/set_meal:")


@st.cache_resource(show_spinner="Loading the detector…")
def get_model(path: str):
    return load_weights(path)


st.title("Fish detector")
st.caption("Finds the fish in a photo and names their species. It was trained on 13 species only.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Minimum confidence", 0.05, 0.95, 0.5, 0.05)
    st.write("Species it knows")
    st.caption(", ".join(SPECIES))

if not WEIGHTS.exists():
    st.error(f"No trained weights found at `{WEIGHTS}`. Train them first (a few minutes on a laptop CPU):")
    st.code("uv run fishdet train", language="bash")
    st.stop()

uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
# The camera is opt-in: a camera widget asks the browser for camera access as soon as it is displayed.
shot = st.camera_input("Take a photo", label_visibility="collapsed") if st.toggle("Use the camera instead") else None

source = shot or uploaded
if source is None:
    st.info("Drop a photo of an aquarium, or take one with the camera.")
    st.stop()

image = Image.open(source)
detections = detect(get_model(str(WEIGHTS)), image, threshold)

st.image(draw_detections(image, detections), width="stretch",
         alt=f"The photo with {len(detections)} detected fish outlined and named")
if detections:
    st.dataframe(
        pd.DataFrame([{"Species": d.species, "Confidence": d.score} for d in detections]),
        column_config={"Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent")},
        hide_index=True, width="stretch", alt="Detected fish with their species and confidence",
    )
else:
    st.warning(f"No fish found above {threshold:.0%} confidence. Try lowering the threshold in the sidebar.")
