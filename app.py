import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from PIL import Image
import io

# 🔁 Ładujemy model tylko raz
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 🧠 Funkcja do skalowania bboxów
def scale_detections(detections: sv.Detections, original_shape, resized_shape):
    scale_x = original_shape[1] / resized_shape[1]
    scale_y = original_shape[0] / resized_shape[0]

    detections.xyxy[:, [0, 2]] *= scale_x
    detections.xyxy[:, [1, 3]] *= scale_y

    return detections

# 🔍 Detekcja i rysowanie
def detect_and_annotate(image: np.ndarray) -> np.ndarray:
    original_shape = image.shape[:2]
    resized_image = cv2.resize(image, (640, 640))

    results = model(resized_image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    # Skalowanie bboxów do oryginalnych wymiarów
    detections = scale_detections(detections, original_shape, resized_image.shape[:2])

    # Rysowanie na oryginalnym obrazie
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
