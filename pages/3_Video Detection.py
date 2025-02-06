import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, NamedTuple
from fpdf import FPDF

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
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

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    detections = []
    
    # Read until video is completed
    while(videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if ret == True:
            
            # Perform inference
            _image = np.array(frame)
            image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
            results = net.predict(image_resized, conf=score_threshold)
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                detections.extend([
                    Detection(
                        class_id=int(_box.cls),
                        label=CLASSES[int(_box.cls)],
                        score=float(_box.conf),
                        box=_box.xyxy[0].astype(int),
                    )
                    for _box in boxes
                ])
    
    # After processing video, generate a CSV and PDF report
    df = pd.DataFrame([{
        "Damage Type": det.label,
        "Confidence": det.score,
        "Bounding Box": str(det.box)
    } for det in detections])

    csv_report = df.to_csv(index=False)
    st.download_button(
        label="Download CSV Report",
        data=csv_report,
        file_name="RDD_Report.csv",
        mime="text/csv"
    )

    # Generate a PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Road Damage Detection Video Report", ln=True, align="C")

    for det in detections:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Damage Type: {det.label}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {det.score:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Bounding Box: {det.box}", ln=True)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.download_button(
        label="Download PDF Report",
        data=pdf_output,
        file_name="RDD_Report.pdf",
        mime="application/pdf"
    )

# Rest of the Streamlit app setup...
