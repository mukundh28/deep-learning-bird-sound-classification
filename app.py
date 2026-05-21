import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings

filterwarnings('ignore')


def streamlit_config():

    st.set_page_config(
        page_title='Classification',
        layout='centered'
    )

    page_background_color = """
    <style>

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    </style>
    """

    st.markdown(
        page_background_color,
        unsafe_allow_html=True
    )

    st.markdown(
        f'<h1 style="text-align: center;">Bird Sound Classification</h1>',
        unsafe_allow_html=True
    )

    add_vertical_space(4)


streamlit_config()


def prediction(audio_file):

    with open(
        r'C:\Users\mukun\Downloads\ML_Project\prediction.json',
        mode='r'
    ) as f:
        prediction_dict = json.load(f)

    audio, sample_rate = librosa.load(audio_file)

    mfccs_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_features = np.mean(mfccs_features, axis=1)

    mfccs_features = np.expand_dims(
        mfccs_features,
        axis=0
    )

    mfccs_features = np.expand_dims(
        mfccs_features,
        axis=2
    )

    mfccs_tensors = tf.convert_to_tensor(
        mfccs_features,
        dtype=tf.float32
    )

    model = tf.keras.models.load_model(
        r'C:\Users\mukun\Downloads\ML_Project\model.h5'
    )

    prediction = model.predict(mfccs_tensors)

    target_label = np.argmax(prediction)

    predicted_class = prediction_dict[str(target_label)]

    confidence = f"{np.max(prediction) * 100:.2f}"

    add_vertical_space(1)

    st.markdown(
        f"<h3 style='text-align:center;color:orange;'>{confidence}% Match Found</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        f'<h3 style="text-align: center; color: green;">{predicted_class}</h3>',
        unsafe_allow_html=True
    )

    image_path = os.path.join(
        r'C:\Users\mukun\Downloads\ML_Project\Inference_Images',
        f'{predicted_class}.jpg'
    )

    if os.path.exists(image_path):

        img = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (350, 300)
        )

        _, col2, _ = st.columns([0.1, 0.8, 0.1])

        with col2:
            st.image(img)

    else:
        st.info(
            f"Image for {predicted_class} not found locally, but prediction was successful!"
        )


_, col2, _ = st.columns([0.1, 0.9, 0.1])

with col2:

    input_audio = st.file_uploader(
        label='Upload the Audio',
        type=['mp3', 'wav']
    )

    if input_audio is not None:

        temp_audio_path = "temp_uploaded_audio.mp3"

        with open(temp_audio_path, "wb") as f:
            f.write(input_audio.getbuffer())

        _, col2, _ = st.columns([0.2, 0.8, 0.2])

        with col2:
            prediction(temp_audio_path)