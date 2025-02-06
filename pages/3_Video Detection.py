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

# Create temporary folder if it doesn't exist
TEMP_DIR = "./temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

temp_file_input = os.path.join(TEMP_DIR, "video_input.mp4")
temp_file_infer = os.path.join(TEMP_DIR, "video_infer.mp4")
temp_image_output = os.path.join(TEMP_DIR, "frame_output.png")
temp_csv_output = os.path.join(TEMP_DIR, "RDD_Report.csv")
temp_pdf_output = os.path.join(TEMP_DIR, "RDD_Report.pdf")

# Save BytesIO file to disk
def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)
    if not videoCapture.isOpened():
        st.error('Error opening the video file')
        return

    _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = videoCapture.get(cv2.CAP_PROP_FPS)
    _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"Video Duration: {int(_frame_count / _fps)} seconds")
    st.write(f"Resolution: {_width} x {_height}, FPS: {_fps}")

    inferenceBar = st.progress(0, text="Processing video, please wait...")
    imageLocation = st.empty()

    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

    detections_list = []
    _frame_counter = 0

    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = net.predict(frame, conf=score_threshold)

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

        annotated_frame = results[0].plot()
        _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)
        
        detections_list.extend(detections)
        
        _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
        cv2writer.write(_out_frame)

        if _frame_counter == 0:
            cv2.imwrite(temp_image_output, cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR))

        imageLocation.image(_image_pred)
        _frame_counter += 1
        inferenceBar.progress(_frame_counter / _frame_count, text="Processing video...")

    videoCapture.release()
    cv2writer.release()
    inferenceBar.empty()

    generate_csv_report(detections_list)
    generate_pdf_report(detections_list)

    st.success("Video Processing Completed!")

    col1, col2, col3 = st.columns(3)
    with col1:
        with open(temp_file_infer, "rb") as f:
            st.download_button("Download Video", data=f, file_name="RDD_Prediction.mp4", mime="video/mp4")

    with col2:
        with open(temp_csv_output, "rb") as f:
            st.download_button("Download CSV Report", data=f, file_name="RDD_Report.csv", mime="text/csv")

    with col3:
        with open(temp_pdf_output, "rb") as f:
            st.download_button("Download PDF Report", data=f, file_name="RDD_Report.pdf", mime="application/pdf")

    st.image(temp_image_output, caption="Annotated Frame", use_column_width=True)
    with open(temp_image_output, "rb") as f:
        st.download_button("Download Annotated Frame", data=f, file_name="RDD_Frame.png", mime="image/png")

def generate_csv_report(detections_list):
    df = pd.DataFrame([{
        "Damage Type": det.label,
        "Confidence": det.score,
        "Bounding Box": str(det.box)
    } for det in detections_list])

    df.to_csv(temp_csv_output, index=False)

def generate_pdf_report(detections_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Road Damage Detection Report", ln=True, align="C")

    for det in detections_list:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Damage Type: {det.label}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {det.score:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Bounding Box: {det.box}", ln=True)

    pdf.output(temp_pdf_output)

st.title("Road Damage Detection - Video")
st.write("Upload a video to detect road damage. Processed results include annotations, reports, and downloadable content.")

video_file = st.file_uploader("Upload Video", type=".mp4")
st.caption("There is a 1GB limit for video files. Resize or trim if necessary.")

score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
st.write("Adjust the threshold to balance detection sensitivity.")

if video_file is not None:
    if st.button("Process Video"):
        st.warning(f"Processing Video: {video_file.name}")
        processVideo(video_file, score_threshold)
