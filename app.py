import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
# Make sure you save your model after training e.g. `model.save("fish_model")`
model = tf.keras.models.load_model(r"C:\Users\Jeeva\Documents\Fishimageclassification\Fish_image_classification\fish_model.keras")

# Define a function to load and preprocess image
def preprocess_image(image: Image.Image, target_size=(150, 150)):
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # same rescaling as during training
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1
    return img_array

# Get class mapping: you should save this mapping when training
# For example: class_indices = train_generator.class_indices
class_indices = {
    "fish_class_1": 0,
    "fish_class_2": 1,
    "fish_class_3": 2,
    "fish_class_4": 3,
    "fish_class_5": 4,
    "fish_class_6": 5,
    "fish_class_7": 6,
    "fish_class_8": 7,
    "fish_class_9": 8,
    "fish_class_10": 9,
    "fish_class_11": 10
    # ...
    # make sure to fill with your actual classes
}
idx_to_class = {v: k for k, v in class_indices.items()}

st.title("Fish Species Classifier")

st.write("Upload a fish image, and the model will predict its class.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image …", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open image via PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess the image
    input_arr = preprocess_image(image)

    # Predict
    predictions = model.predict(input_arr)
    pred_index = np.argmax(predictions[0])
    pred_index = int(pred_index)  # convert numpy.int64 to plain int
    pred_class = idx_to_class.get(pred_index, "Unknown class")

    confidence = float(predictions[0][pred_index])

    st.write(f"**Predicted class:** {pred_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Optionally, show probabilities for all classes
    st.write("**All class probabilities:**")
    for class_name, idx in class_indices.items():
        st.write(f"{class_name}: {predictions[0][idx] * 100:.2f}%")
