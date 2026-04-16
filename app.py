import os

import joblib
import streamlit as st

from data_preparation import prepare_final_dataset
from predictor import predict_yield
from train_model import train_and_save


st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Prediction")


def ensure_artifacts() -> None:
    if not os.path.exists("final_dataset.csv"):
        prepare_final_dataset(output_path="final_dataset.csv")

    needed = ["model.pkl", "crop_encoder.pkl", "state_encoder.pkl"]
    if not all(os.path.exists(path) for path in needed):
        train_and_save(
            dataset_path="final_dataset.csv",
            model_path="model.pkl",
            crop_encoder_path="crop_encoder.pkl",
            state_encoder_path="state_encoder.pkl",
        )


try:
    ensure_artifacts()
    crop_encoder = joblib.load("crop_encoder.pkl")
    state_encoder = joblib.load("state_encoder.pkl")

    available_crops = sorted(crop_encoder.classes_.tolist())
    available_states = sorted(state_encoder.classes_.tolist())

    crop = st.selectbox("Crop", options=available_crops)
    state = st.selectbox("State", options=available_states)
    rainfall = st.slider("Annual Rainfall (mm)", min_value=0.0, max_value=2000.0, value=800.0, step=1.0)
    temperature = st.slider("Average Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0, step=0.1)

    if st.button("Predict Yield"):
        raw_prediction = predict_yield(
            crop=crop,
            state=state,
            rainfall=rainfall,
            temperature=temperature,
            model_path="model.pkl",
            crop_encoder_path="crop_encoder.pkl",
            state_encoder_path="state_encoder.pkl",
        )
        final_prediction = float(raw_prediction)

        st.success(f"Predicted Yield: {final_prediction:.2f} kg/ha")
        st.write("Inputs used:")
        st.write(
            {
                "crop": crop,
                "state": state,
                "annual_rainfall_mm": rainfall,
                "avg_temperature_c": temperature,
                "rainfall_squared": rainfall ** 2,
            }
        )
        st.write("Debug:")
        st.write(
            {
                "raw_prediction_kg_per_ha": raw_prediction,
                "final_displayed_kg_per_ha": final_prediction,
            }
        )

        if (
            crop == "rice"
            and state == "PUNJAB"
            and 800.0 <= rainfall <= 1000.0
            and abs(temperature - 30.0) < 1e-9
            and not (2000.0 <= final_prediction <= 4000.0)
        ):
            st.warning("Prediction is outside expected 2000-4000 kg/ha range for this validation case.")
except FileNotFoundError as err:
    st.error(f"File not found: {err}")
except ValueError as err:
    st.error(f"Validation error: {err}")
except Exception as err:
    st.error(f"Unexpected error: {err}")
