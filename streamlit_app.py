import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the ESRGAN model from TensorFlow Hub
model = hub.load("https://www.kaggle.com/models/kaggle/esrgan-tf2/TensorFlow2/esrgan-tf2/1")

def load_and_preprocess_image(image_file):
    """Load and preprocess the image for super-resolution."""
    image = Image.open(image_file)
    image = image.convert("RGB")  # Ensure the image has three color channels
    image = image.resize((image.width, image.height))  # Resize to the same dimensions
    image_np = np.array(image)
    return image_np

def perform_super_resolution(low_res_image):
    """superRES"""
    low_res_image = tf.expand_dims(low_res_image, axis=0)  # Add batch dimension
    low_res_image = tf.cast(low_res_image, tf.float32)
    super_res_image = model(low_res_image)
    super_res_image = tf.squeeze(super_res_image, axis=0)  # Remove batch dimension
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)  # Clip values for display
    super_res_image = tf.cast(super_res_image, tf.uint8).numpy()  # Convert to uint8 format
    return super_res_image

st.title("Image Super-Resolution with ESRGAN")
st.write("Upload a low-resolution image, and the app will generate a higher-resolution version using ESRGAN.")

# Image upload
uploaded_image = st.file_uploader("Choose a low-resolution image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load and display the low-resolution image
    low_res_image = load_and_preprocess_image(uploaded_image)
    st.image(low_res_image, caption="Low-Resolution Image", use_column_width=True)
    
    # Convert the image to a format suitable for model processing
    low_res_image_tensor = tf.convert_to_tensor(low_res_image)

    # Perform super-resolution
    super_res_image = perform_super_resolution(low_res_image_tensor)

    # Check dimensions for debugging
    st.write("Original Image Dimensions:", low_res_image.shape)
    st.write("Super-Resolution Image Dimensions:", super_res_image.shape)

    # Display the super-resolved image
    st.image(super_res_image, caption="Super-Resolution Image", use_column_width=True)
