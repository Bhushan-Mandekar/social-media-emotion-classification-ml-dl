
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Social Media Emotion Predictor",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# =====================================================
# LOAD SAVED MODEL ARTIFACTS
# =====================================================

@st.cache_resource
def load_artifacts():

    with open(
        OUTPUT_DIR / "model_metadata.json",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    preprocessor = joblib.load(
        OUTPUT_DIR / "preprocessor.pkl"
    )

    encoder = joblib.load(
        OUTPUT_DIR / "emotion_label_encoder.pkl"
    )

    model_type = metadata["model_type"]

    if model_type == "ML":

        model = joblib.load(
            OUTPUT_DIR / "final_ml_model.pkl"
        )

    elif model_type == "DL":

        from tensorflow.keras.models import load_model

        model = load_model(
            OUTPUT_DIR / "final_dl_model.keras"
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    return model, preprocessor, encoder, metadata


try:
    model, preprocessor, encoder, metadata = load_artifacts()

except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("🧠 Social Media Emotion Predictor")

st.markdown(
    """
    Predict a user's **dominant emotion** based on
    social media activity and usage patterns.

    This application uses a trained Machine Learning
    or Deep Learning classification model.
    """
)

st.divider()

# =====================================================
# USER INPUT FORM
# =====================================================

st.subheader("Enter User Information")

with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=13,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Non-binary"]
    )

    platform = st.selectbox(
        "Social Media Platform",
        [
            "Instagram",
            "Facebook",
            "Twitter",
            "LinkedIn",
            "Snapchat",
            "WhatsApp",
            "Telegram",
            "TikTok",
            "YouTube"
        ]
    )

    usage = st.number_input(
        "Daily Usage Time (minutes)",
        min_value=0,
        max_value=1440,
        value=120
    )

    posts = st.number_input(
        "Posts Per Day",
        min_value=0,
        value=3
    )

    likes = st.number_input(
        "Likes Received Per Day",
        min_value=0,
        value=50
    )

    comments = st.number_input(
        "Comments Received Per Day",
        min_value=0,
        value=10
    )

    messages = st.number_input(
        "Messages Sent Per Day",
        min_value=0,
        value=20
    )

    submitted = st.form_submit_button(
        "Predict Emotion",
        use_container_width=True
    )


# =====================================================
# PREDICTION
# =====================================================

if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Platform": platform,
        "Daily_Usage_Time (minutes)": usage,
        "Posts_Per_Day": posts,
        "Likes_Received_Per_Day": likes,
        "Comments_Received_Per_Day": comments,
        "Messages_Sent_Per_Day": messages
    }])

    feature_columns = metadata["features"]

    processed_data = preprocessor.transform(
        input_data[feature_columns]
    )

    if metadata["model_type"] == "DL":

        probabilities = model.predict(
            processed_data,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

    else:

        predicted_class = int(
            model.predict(processed_data)[0]
        )

        probabilities = (
            model.predict_proba(processed_data)[0]
            if hasattr(model, "predict_proba")
            else None
        )

    emotion = encoder.inverse_transform(
        [predicted_class]
    )[0]

    st.success(
        f"Predicted Dominant Emotion: {emotion}"
    )

    if probabilities is not None:

        probability_df = pd.DataFrame({
            "Emotion": encoder.classes_,
            "Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.subheader("Prediction Probabilities")

        st.bar_chart(
            probability_df.set_index("Emotion")
        )

        st.caption(
            "These are model-estimated probabilities "
            "and may not be calibrated."
        )

    st.info(
        "This prediction is for educational and "
        "research purposes only. It is not a "
        "clinical or psychological assessment."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Capstone-3 | Social Media Usage and "
    "Emotional Well-Being | Developed by Bhushan Mandekar"
)
