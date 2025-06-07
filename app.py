import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model_scaler():
    with open("best_model (1).pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler (1).pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

@st.cache_data
def load_submission():
    return pd.read_csv("submission.csv")

st.set_page_config(page_title="Efficiency Predictor", layout="centered")
st.title("🔋 Solar Panel Efficiency Predictor")

model, scaler = load_model_scaler()
submission = load_submission()

input_id = st.text_input("Enter the ID to get predicted efficiency:")

if input_id:
    try:
        input_id = int(input_id)
        if input_id in submission['id'].values:
            efficiency_value = submission.loc[submission['id'] == input_id, 'efficiency'].values[0]
            st.success(f"✅ Predicted Efficiency for ID {input_id}: **{efficiency_value:.4f}**")
        else:
            st.error(f"❌ No prediction found for ID {input_id}.")
    except ValueError:
        st.error("⚠️ Please enter a valid numeric ID.")
