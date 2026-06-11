import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import cv2
st.write(cv2.__version__)
from ultralytics import YOLO

# Create temp folder if not exists
os.makedirs("temp", exist_ok=True)

st.set_page_config(
    page_title="Multimodal Accident Risk Assessment",
    layout="wide"
)

st.title(
    "Multimodal Accident Risk Assessment System"
)

# Load models
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

# Upload section
csv_file = st.file_uploader(
    "Upload Environmental CSV",
    type=["csv"]
)

image_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

# Analyze button
if st.button("Analyze"):

    # Check uploads
    if csv_file is None:
        st.error("Please upload CSV file")
        st.stop()

    if image_file is None:
        st.error("Please upload Traffic Sign Image")
        st.stop()

    # CSV Prediction
    df = pd.read_csv(csv_file)

    csv_score = float(
        rf_model.predict(df)[0]
    )

    # Save uploaded image
    image_path = "temp/test.jpg"

    with open(image_path, "wb") as f:
        f.write(image_file.read())

    # YOLO Prediction
    results = yolo_model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) > 0:

        sign_score = float(
            boxes.conf.max().cpu().numpy()
        )

        class_id = int(
            boxes.cls[0].cpu().numpy()
        )

        sign_name = yolo_model.names[class_id]

        annotated = results[0].plot()

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

    else:

        sign_score = 0.0
        sign_name = "No Sign Detected"
        annotated = cv2.imread(image_path)
        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

    # Fusion
    final_score = (
        0.5 * csv_score +
        0.5 * sign_score
    )

    # Risk Level
    if final_score > 0.7:
        risk = "HIGH"

    elif final_score > 0.4:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    # Results
    st.subheader("Results")

    st.write(
        "Environmental Risk Score:",
        round(csv_score, 4)
    )

    st.write(
        "Traffic Sign:",
        sign_name
    )

    st.write(
        "Traffic Sign Confidence:",
        round(sign_score, 4)
    )

    st.write(
        "Final Risk Score:",
        round(final_score, 4)
    )

    st.write(
        "Risk Level:",
        risk
    )

    st.image(
        annotated,
        caption="Traffic Sign Detection",
        use_container_width=True
    )
