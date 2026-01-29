import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -----------------------------
# Load pretrained objects
# -----------------------------

@st.cache_resource
def load_artifacts():
    model = load_model("lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# Must be SAME as training
MAX_LEN = 100  

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mental Health Text Classifier", layout="centered")

st.title("🧠 Mental Health Text Classifier")
st.write("Enter text to detect mental health risk using LSTM.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # -----------------------------
        # Preprocess input
        # -----------------------------
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        prob = model.predict(padded_seq, verbose=0)[0][0]

        label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("Prediction Result")
        st.write(f"**Risk Level:** {label}")
        st.write(f"**Confidence Score:** {prob:.3f}")
