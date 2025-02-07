import os
import logging
import pandas as pd
from pathlib import Path
from typing import NamedTuple
from fpdf import FPDF  # For PDF generation
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

# Load YOLOv8 model
from ultralytics import YOLO

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="\ud83d\udcf7",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load YOLO model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Severity classification
def get_severity(box, score):
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height  # Bounding box area

    if area < 5000:
        return "Minor", "green"
    elif area < 15000 and score >= 0.6:
        return "Moderate", "orange"
    elif area >= 15000 and score >= 0.8:
        return "Severe", "red"
    else:
        return "Minor", "green"

st.title("Road Damage Detection - Image")
st.write("Detect road damage using an image input. Upload the image and start detecting.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if no damage is detected, and increase it if there are false predictions.")

if image_file is not None:
    image = Image.open(image_file)
    _image = np.array(image)
    h_ori, w_ori, _ = _image.shape

    col1, col2 = st.columns(2)

    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    # Draw bounding boxes and labels
    annotated_frame = _image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.box
        label_text = f"{det.label} {det.score:.2f}"
        severity, color = get_severity(det.box, det.score)

        font_scale = 1.2  # Increased for better visibility
        font_thickness = 3  # Increased thickness
        font = cv2.FONT_HERSHEY_SIMPLEX

        bbox_color = (0, 255, 0) if severity == "Minor" else (0, 165, 255) if severity == "Moderate" else (0, 0, 255)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 3)

        # Draw background rectangle for label
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x, text_y = x1, y1 - 10
        cv2.rectangle(annotated_frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5), bbox_color, -1)

        # Draw text with outline for better visibility
        cv2.putText(annotated_frame, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness + 2)  # Black outline
        cv2.putText(annotated_frame, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)  # White text

    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    with col1:
        st.write("### Original Image")
        st.image(_image)

    with col2:
        st.write("### Predicted Image")
        st.image(_image_pred)
