import os
import pathlib
import streamlit as st
from PIL import Image
import torch
import numpy as np

# Fixing the torch classes path issue
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except AttributeError:
    torch.classes.__path__ = []

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path='yolov5best_aug_false.pt')
    return model

# Function to detect objects in the image
def detect_objects(image, conf_threshold):
    model = load_model()
    model.conf = conf_threshold  # Set confidence threshold
    results = model(image)
    return results

# Streamlit app
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
        margin-top: 50px;
    }
    .red-label {
        color: red;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Crop Disease Detection</div>', unsafe_allow_html=True)

# Add red color to the "Select the crop" label
st.markdown('<div class="red-label">Select the crop</div>', unsafe_allow_html=True)
crop_selection = st.selectbox("", ["Paddy", "Wheat", "Maize"])
st.write(f"Selected Crop: {crop_selection}")

# Image uploader and confidence threshold
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Precautions/remedies dictionary
precautions_dict = {
    'brown_spots': [
        "Remove infected leaves and stems to prevent the spread of the disease.",
        "Ensure proper irrigation to prevent waterlogging.",
        "Apply fungicide to treat the infection.",
        "Avoid overcrowding of plants to allow better air circulation."
    ],
    'yellowing_leaves': [
        "Improve soil nutrition by adding organic fertilizers.",
        "Ensure proper drainage to avoid water stress.",
        "Avoid over-fertilizing with nitrogen-based fertilizers.",
        "Use resistant crop varieties if available."
    ],
}

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    img_cv = np.array(img)

    if st.button("Run Detection"):
        results = detect_objects(img_cv, conf_threshold)
        st.subheader("Detection Results")
        inferenced_img = results.render()[0]
        inferenced_img_pil = Image.fromarray(inferenced_img)
        st.image(inferenced_img_pil, caption="Inferenced Image.", use_container_width=True)

        detected_classes = results.names
        pred_boxes = results.pred[0]

        if pred_boxes.shape[0] > 0:
            max_conf_idx = torch.argmax(pred_boxes[:, -2])
            max_conf_class_id = int(pred_boxes[max_conf_idx, -1])
            max_conf_score = round(pred_boxes[max_conf_idx, -2].item(), 2)
            max_conf_class_name = detected_classes[max_conf_class_id]

            st.success(f"Highest Confidence Prediction: {max_conf_class_name} (Confidence: {max_conf_score})")

            if max_conf_class_name in precautions_dict:
                st.subheader("Precautions / Remedies:")
                for precaution in precautions_dict[max_conf_class_name]:
                    st.write(f"- {precaution}")
            else:
                st.write("No specific precautions available for this disease.")
