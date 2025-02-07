import os
import logging
import pandas as pd
from pathlib import Path
from typing import NamedTuple
from fpdf import FPDF  # For PDF generation

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file
from ultralytics import YOLO

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
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

CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Severity classification
def get_severity(box, score):
    width, height = box[2] - box[0], box[3] - box[1]
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
    # Load and convert the image
    image = Image.open(image_file)
    _image = np.array(image)
    h_ori, w_ori, _ = _image.shape

    col1, col2 = st.columns(2)

    # Resize image for YOLO input
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
    severity_set = set()  # Store unique (label, severity) pairs

    for det in detections:
        x1, y1, x2, y2 = det.box
        label_text = f"{det.label}: {det.score:.2f}"
        severity, color = get_severity(det.box, det.score)
        severity_set.add((det.label, severity, color))

        # Font settings
        font_scale, font_thickness = 0.5, 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Bounding box color
        bbox_color = (0, 255, 0) if severity == "Minor" else (0, 165, 255) if severity == "Moderate" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)

        # Draw label
        cv2.putText(annotated_frame, label_text, (x1, max(y1 - 5, 15)), font, font_scale, (0, 0, 0), font_thickness)

    # Resize annotated image back to original dimensions
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Display images
    with col1:
        st.write("#### Original Image")
        st.image(_image)

    with col2:
        st.write("#### Predicted Image")
        st.image(_image_pred)

        # Display unique severity levels
        if severity_set:
            st.markdown("#### Severity Levels:")
            severity_html = [
                f"<span style='color:{color}; font-weight:bold;'>{label} - {severity}</span>"
                for label, severity, color in severity_set
            ]
            st.markdown("<br>".join(severity_html), unsafe_allow_html=True)

        # Download annotated image
        buffer = BytesIO()
        Image.fromarray(_image_pred).save(buffer, format="PNG")
        st.download_button(
            label="Download Prediction Image",
            data=buffer.getvalue(),
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

        # Generate CSV report
        df = pd.DataFrame([{
            "Damage Type": det.label,
            "Confidence": det.score,
            "Bounding Box": str(det.box),
            "Severity Level": get_severity(det.box, det.score)[0]
        } for det in detections])

        st.download_button(
            label="Download CSV Report",
            data=df.to_csv(index=False),
            file_name="RDD_Report.csv",
            mime="text/csv"
        )

        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Road Damage Detection Report", ln=True, align="C")

        for det in detections:
            severity, _ = get_severity(det.box, det.score)
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Damage Type: {det.label}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {det.score:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Bounding Box: {det.box}", ln=True)
            pdf.cell(200, 10, txt=f"Severity Level: {severity}", ln=True)

        # Save PDF and provide download button
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)

        st.download_button(
            label="Download PDF Report",
            data=pdf_output.read(),
            file_name="RDD_Report.pdf",
            mime="application/pdf"
        )
