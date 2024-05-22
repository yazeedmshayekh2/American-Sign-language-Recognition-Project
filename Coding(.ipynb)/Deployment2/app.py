import streamlit as st
import os
from Detection import process_video  # Import the process function
from Inference import load_relevant_data_subset, TFLiteModel, decoder
from Model_Architecture import models
import numpy as np

# Function to save the uploaded video
def save_uploaded_file(uploaded_file):
    directory = "tempDir"
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False


# Streamlit interface
st.title('Video Upload and Processing')
st.write('Please upload a video for processing.')

uploaded_file = st.file_uploader("Choose a file...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success('File saved successfully!')
        video_path = os.path.join("tempDir", uploaded_file.name)
        st.video(video_path)

        # Placeholder for processing the video and generating output
        # Assume 'process_video' is your function to handle the video
        st.write(print(process_video(video_path)))

        tflite_keras_model = TFLiteModel(islr_models=models)
        relevant_data, data = load_relevant_data_subset('test_case.parquet')
        demo_output = tflite_keras_model(relevant_data)["outputs"]

        # Display the output in a text box
        st.text_area("Output", decoder(np.argmax(demo_output.numpy(), axis=-1)), height=300)
    else:
        st.error('Failed to save file.')
else:
    st.warning('No file uploaded yet.')
