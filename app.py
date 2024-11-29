import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained VGG model
model = VGG16(weights='imagenet')

# Function to load and preprocess the image
def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit app title
st.title("Image Classification with VGG16")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Load and preprocess the image
        img = image.load_img(uploaded_file)
        img_array = load_and_preprocess_image(img)

        # Make predictions
        predictions = model.predict(img_array)
        top_prediction = decode_predictions(predictions, top=1)[0][0]

        # Display the top prediction
        st.subheader("Prediction:")
        imagenet_id, label, score = top_prediction
        percentage_score = score * 100  # Convert score to percentage
        st.write(f"{label} - {percentage_score:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")
