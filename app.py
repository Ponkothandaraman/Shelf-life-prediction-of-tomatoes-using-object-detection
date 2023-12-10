import streamlit as st
import subprocess
import pandas as pd
import os
import cv2
from PIL import Image
import io 

def run_detection(weights_path, img_size, confidence, source_path):
    yolo5_path = r'C:\Users\Ram\yolov5'  # Set the correct path to the YOLOv5 repository
    command = [
        "python",
        os.path.join(yolo5_path, "detect.py"),  # Use the absolute path to detect.py
        "--weights",
        weights_path,
        "--img",
        str(img_size),
        "--conf",
        str(confidence),
        "--source",
        source_path,
        "--save-csv",
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=yolo5_path)

    # Check if the command was successful
    if result.returncode == 0:
        st.success("Detection completed successfully.")

        # Get the path of the saved CSV file
        csv_path = os.path.join(yolo5_path, "runs", "detect", "exp", "predictions.csv")

        # Get the path of the saved image
        img_path = os.path.join(yolo5_path, "runs", "detect", "exp", source_path)  # Replace with the actual path

        return csv_path, img_path
    else:
        st.error("Error during detection.")
        st.text(result.stderr)
        return None, None


st.title("Tomato shelf life prediction using Yolo V5")


confidence = st.slider("Select confidence threshold:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
source_path = st.file_uploader("Choose an image or video file:", type=["jpg", "jpeg", "png"])
if st.button("Predict"):
    if source_path is not None:
        csv_path , img_path = run_detection("C:/Users/Ram/Desktop/lll/best.pt", 640, confidence, source_path.name)
        df= pd.read_csv(csv_path,header=None)
        df.columns = ['Img','Label','Confidence']
        aggregate = df['Label'].value_counts().reset_index()
        st.dataframe(aggregate)

        img = Image.open(img_path)
        st.image(img)