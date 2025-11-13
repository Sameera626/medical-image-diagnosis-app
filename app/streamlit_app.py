import streamlit as st
import tensorflow as tf
import tempfile
import numpy as np
from pathlib import Path
from src.model_explain import make_gradcam_heatmap, save_and_display_gradcam





st.set_page_config(page_title="Medical Image Diagnosis", layout="centered")

st.title("Medical Image diagnosis Application")
st.write("Upload a chest X-ray image. The model predicts Normal vs Pneumonia and shows GRAd-CAM.")


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = None

try:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "saved_model" / "medical_image_model.keras"
    model = load_model(str(model_path))

except Exception as e:
    st.error(f"Cannot load model: {e}")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.getvalue())
    image_path = tfile.name

    img_arr = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    st.image(img_arr, caption="Input Image")

    # preprocess the uploaded image
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    img = tf.keras.preprocessing.image.img_to_array(img_arr)
    img_batch = np.expand_dims(img, axis=0)
    img_pre = preprocess(img_batch.copy())

    pred = model.predict(img_pre)[0][0]
    st.write(f"predicted probability of pneumonia: {pred:.4f}")
    label = "Pneumonia" if pred >= 0.7 else "Normal"
    st.subheader(f"Prediction: {label}")

    

    # # GRAd-CAM visualization
    last_conv = None
    base_model = model.get_layer("efficientnetb0")
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            print(f"Last convolutional layer: {last_conv}")
            break

    if last_conv is None:
        st.write("Cannot find a convelutional layer in the model.")

    else:
        heatmap = make_gradcam_heatmap(img_pre, model)
        cam_path = save_and_display_gradcam(image_path, heatmap)
        st.image(cam_path, caption="GRAD-CAM")





