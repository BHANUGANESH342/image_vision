import streamlit as st
from PIL import Image
import torch
import numpy as np
from pathlib import Path

# Ensure you are using Path, which will automatically select the correct platform type
script_path = Path(__file__).resolve().parent

# Function to load the YOLOv5 model (handles file upload and model loading)
@st.cache_resource
def load_model(model_file):
    try:
        model = torch.load(model_file, map_location=torch.device('cpu'))
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to detect objects in the image
def detect_objects(image, conf_threshold, model):
    model.conf = conf_threshold  # Set confidence threshold
    results = model(image)
    return results

# Streamlit app layout and design
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
        margin-top: 50px;
    }
    .top-left {
        position: absolute;
        top: 10px;
        left: 10px;
    }
    .top-right {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .red-label {
        color: red;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Upload logos through file uploader (instead of hardcoding paths)
logo_left = st.file_uploader("Upload Left Logo", type=["png", "jpg"])
logo_right = st.file_uploader("Upload Right Logo", type=["png", "jpg"])

if logo_left and logo_right:
    st.markdown(f'<div class="top-left"><img src="data:image/png;base64,{logo_left.getvalue().decode()}" width="100"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-right"><img src="data:image/png;base64,{logo_right.getvalue().decode()}" width="100"></div>', unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Crop Disease Detection</div>', unsafe_allow_html=True)

# Add red color to only the "Select the crop" label
st.markdown('<div class="red-label">Select the crop</div>', unsafe_allow_html=True)
crop_selection = st.selectbox("Select the crop", ["Paddy", "Wheat", "Maize"])
st.write(f"Selected Crop: {crop_selection}")

# Add default labels with no custom style for other widgets
st.write("Choose an image...")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.write("Set Confidence Threshold")
conf_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Define the precautions/remedies dictionary
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
    # Add more diseases and their precautions here...
}

# Upload model file via Streamlit
model_file = st.file_uploader("Upload YOLOv5 Model", type=["pt"])

if model_file is not None:
    model = load_model(model_file)
    if model is not None:
        st.success("Model loaded successfully!")

        if uploaded_image is not None:
            # Open the image
            img = Image.open(uploaded_image)

            # Display the uploaded image
            st.image(img, caption="Uploaded Image.", use_container_width=True)

            # Convert to NumPy array (OpenCV format)
            img_cv = np.array(img)

            # Detection button
            if st.button("Run Detection"):
                # Run detection
                results = detect_objects(img_cv, conf_threshold, model)

                # Display results
                st.subheader("Detection Results")

                # Show inferenced image with bounding boxes
                inferenced_img = results.render()[0]  # Get the inferenced image (with boxes)

                # Convert inferenced image back to PIL for streamlit display
                inferenced_img_pil = Image.fromarray(inferenced_img)
                st.image(inferenced_img_pil, caption="Inferenced Image.", use_container_width=True)

                # Extract detected class names and confidence scores
                detected_classes = results.names
                pred_boxes = results.pred[0]  # Get prediction results

                # Find the class with the highest confidence
                if pred_boxes.shape[0] > 0:  # Ensure there are detections
                    max_conf_idx = torch.argmax(pred_boxes[:, -2])  # Get index of max confidence
                    max_conf_class_id = int(pred_boxes[max_conf_idx, -1])
                    max_conf_score = round(pred_boxes[max_conf_idx, -2].item(), 2)
                    max_conf_class_name = detected_classes[max_conf_class_id]

                    # Show only the highest confidence class in a dialogue box
                    st.success(f"Highest Confidence Prediction: {max_conf_class_name} (Confidence: {max_conf_score})")

                    # Display precautions/remedies for the detected class
                    if max_conf_class_name in precautions_dict:
                        st.subheader("Precautions / Remedies:")
                        for precaution in precautions_dict[max_conf_class_name]:
                            st.write(f"- {precaution}")
                    else:
                        st.write("No specific precautions available for this disease.")
    else:
        st.error("Failed to load model.")
