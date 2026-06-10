import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2

from ultralytics import YOLO

st.set_page_config(
    page_title="Multimodal Accident Risk Assessment",
    layout="wide"
)

st.title(
    "Multimodal Accident Risk Assessment System"
)

#Load models
@st.cache_resource
def load_models():

    rf_model = joblib.load(
        "models/csv_risk_model.pkl"
    )

    yolo_model = YOLO(
        "models/best.pt"
    )

    return rf_model, yolo_model

rf_model, yolo_model = load_models()

#Upload section
csv_file = st.file_uploader(
    "Upload Environmental CSV",
    type=["csv"]
)

image_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg","jpeg","png"]
)

#Analyze button
if st.button("Analyze"):


#csv prediction
df = pd.read_csv(csv_file)

csv_score = float(
    rf_model.predict(df)[0]
)

#yolo prediction
with open(
    "temp/test.jpg",
    "wb"
) as f:

    f.write(
        image_file.read()
    )

results = yolo_model.predict(
    source="temp/test.jpg",
    conf=0.25,
    verbose=False
)

#confidence
boxes = results[0].boxes

if len(boxes) > 0:

    sign_score = float(
        boxes.conf.max()
    )

    class_id = int(
        boxes.cls[0]
    )

    sign_name = (
        yolo_model.names[class_id]
    )

else:

    sign_score = 0

    sign_name = "No Sign"

#fusion
final_score = (
    0.5 * csv_score +
    0.5 * sign_score
)

#risk level
if final_score > 0.7:

    risk = "HIGH"

elif final_score > 0.4:

    risk = "MEDIUM"

else:

    risk = "LOW"

#output
st.subheader("Results")

st.write(
    "Environmental Risk Score:",
    round(csv_score,4)
)

st.write(
    "Traffic Sign:",
    sign_name
)

st.write(
    "Traffic Sign Confidence:",
    round(sign_score,4)
)

st.write(
    "Risk Level:",
    risk
)
