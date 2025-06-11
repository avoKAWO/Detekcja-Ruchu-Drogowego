import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from PIL import Image
import io

# 🔁 Załaduj model tylko raz
@st.cache_resource
def load_model():
    return YOLO("best_1.pt")

model = load_model()

# 🎨 Funkcja do oznaczenia obrazu
def detect_and_annotate(image: np.ndarray) -> np.ndarray:
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    return annotated_image

# 🌐 Interfejs Streamlit
st.title("🔍 YOLO Detekcja Obiektów")
uploaded_file = st.file_uploader("Wgraj obraz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Oryginalny obraz", use_container_width=True)

    annotated = detect_and_annotate(image_np)
    st.image(annotated, caption="Z oznaczeniami YOLO", use_container_width=True)
