import streamlit as st
from PIL import Image
import pickle
import numpy as np

# Load the hot dog detection model
with open('hotdog.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the Streamlit app
def main():
    st.file('Hot Dog Detector')
    st.write('Upload an image to check if a hot dog is present')

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform hot dog detection
        prediction = model.predict(image)

        if prediction == 1:
            st.write('We have a hot dog in our midst!')
        else:
            st.write('No hot dog was detected')

if __name__ == '__main__':
    main()