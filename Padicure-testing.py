import streamlit as st
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from io import StringIO
import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model('../Model/Model-Acc82.h5')

st.markdown("<h1 style='text-align: center;'>Test Predict</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>BrownSpot, Healthy,Hispa,LeafBlast</h1>", unsafe_allow_html=True)
st.text('')
st.text('')
class_names = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast']

uploaded_files = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
if uploaded_files is not None:
    bytes_data = uploaded_files.getvalue()
    st.image(bytes_data)
    image = load_img(uploaded_files,target_size=(150,150))
    x = img_to_array(image)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    pred = new_model.predict(images)
    pred = class_names[np.argmax(pred)]
    st.header(f"Predictions: {pred}")
