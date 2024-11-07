import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the ESRGAN model from TensorFlow Hub
model = hub.load("https://www.kaggle.com/models/kaggle/esrgan-tf2/TensorFlow2/esrgan-tf2/1")

def load_and_preprocess_image(image_file, downscale_factor=4):
    """Load the image and create a low-resolution version using bicubic interpolation."""
    # Open and convert the image to RGB
    image = Image.open(image_file).convert("RGB")

    # Calculate the new low-resolution size
    low_res_size = (image.width // downscale_factor, image.height // downscale_factor)

    # Resize to low resolution using bicubic interpolation
    low_res_image = image.resize(low_res_size, Image.BICUBIC)
    
    # Convert the low-resolution image to a numpy array
    low_res_image_np = np.array(low_res_image)
    
    return image, low_res_image_np

def perform_super_resolution(low_res_image):
    """Perform super-resolution using the ESRGAN model."""
    # Add batch dimension and cast to float32
    low_res_image = tf.expand_dims(low_res_image, axis=0)
    low_res_image = tf.cast(low_res_image, tf.float32)
    
    # Run super-resolution
    super_res_image = model(low_res_image)
    
    # Remove batch dimension and clip values to [0, 255]
    super_res_image = tf.squeeze(super_res_image, axis=0)
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)
    super_res_image = tf.cast(super_res_image, tf.uint8).numpy()
    
    return super_res_image

st.title("superRES")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load original image and generate a bicubic low-resolution image
    original_image, low_res_image = load_and_preprocess_image(uploaded_image)
    
    # Display the bicubic low-resolution image
    st.image(low_res_image, caption="Bicubic Low-Resolution Image", use_column_width=True)

    # Convert the low-resolution image to a tensor for the model
    low_res_image_tensor = tf.convert_to_tensor(low_res_image)

    # Perform super-resolution
    super_res_image = perform_super_resolution(low_res_image_tensor)

    # Display the original and super-resolved images
    st.image(super_res_image, caption="Super-Resolution Image", use_column_width=True)
