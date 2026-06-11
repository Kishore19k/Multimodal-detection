import streamlit as st
import pandas as pd
import joblib
import os

from ultralytics import YOLO
from PIL import Image

# Create temp folder
os.makedirs("temp", exist_ok=True)

st.set_page_config(
    page_title="Multimodal Accident Risk Assessment",
    layout="wide"
)

st.title("Multimodal Accident Risk Assessment System")


@st.cache_resource
def load_models():

    try:
        rf_model = joblib.load(
            "models/csv_risk_model.pkl"
        )

    except Exception as e:
        st.error(f"RF Model Error: {e}")
        st.stop()

    try:
        yolo_model = YOLO(
            "models/best.pt"
        )

    except Exception as e:
        st.error(f"YOLO Model Error: {e}")
        st.stop()

    return rf_model, yolo_model


rf_model, yolo_model = load_models()

st.header("Upload Inputs")

csv_file = st.file_uploader(
    "Upload Environmental CSV",
    type=["csv"]
)

image_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

if st.button("Analyze"):

    if csv_file is None:
        st.error("Please upload CSV file")
        st.stop()

    if image_file is None:
        st.error("Please upload Traffic Sign Image")
        st.stop()

    try:

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

        else:

            sign_score = 0.0

            sign_name = "No Sign Detected"

            annotated = Image.open(image_path)

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

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Environmental Risk Score",
                round(csv_score, 4)
            )

            st.metric(
                "Traffic Sign Confidence",
                round(sign_score, 4)
            )

        with col2:

            st.metric(
                "Final Risk Score",
                round(final_score, 4)
            )

            st.metric(
                "Risk Level",
                risk
            )

        st.write(
            f"Detected Traffic Sign: **{sign_name}**"
        )

        st.image(
            annotated,
            caption="Traffic Sign Detection Output"
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )
