import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
from fpdf import FPDF
import io

def display_predicted_image(image_path, detections):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox, label, severity = detection
        draw.rectangle(bbox, outline='green', width=3)
        draw.text((bbox[0], bbox[1] - 10), f"{label} - {severity}", fill='green')
    
    st.image(image, use_column_width=True)

def generate_csv(detections):
    df = pd.DataFrame(detections, columns=['Bounding Box', 'Label', 'Severity'])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_pdf(detections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "Road Damage Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for detection in detections:
        bbox, label, severity = detection
        pdf.cell(200, 10, f"{label}: {severity}", ln=True)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer, 'F')
    return pdf_buffer.getvalue()

st.title("Predicted Image")
image_path = "image.png"  # Update this path accordingly
detections = [
    [(50, 50, 200, 200), "Alligator Crack", "Minor"],
    # Add more detections if needed
]
display_predicted_image(image_path, detections)
st.markdown("## Severity Levels:")
for detection in detections:
    _, label, severity = detection
    st.markdown(f"**{label} - {severity}**", unsafe_allow_html=True)

csv_data = generate_csv(detections)
st.download_button(label="Download CSV Report", data=csv_data, file_name="report.csv", mime="text/csv")

pdf_data = generate_pdf(detections)
st.download_button(label="Download PDF Report", data=pdf_data, file_name="report.pdf", mime="application/pdf")
