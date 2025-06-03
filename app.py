import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

st.set_page_config(page_title="IMDB Review Classifier", page_icon="🎬", layout="centered")
st.title("IMDB Review Classifier")
st.markdown("This app classifies IMDB movie reviews as **Positive** or **Negative** using a deep learning model.")

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("text_classifier.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the file 'text_classifier.h5' is present.")
        st.stop()

@st.cache_resource
def get_vectorizer():
    # Load vocabulary from vocab.txt
    with open("vocab.txt", "r") as f:
        vocab = [line.strip() for line in f]
    vectorizer = TextVectorization(
        max_tokens=10000,            # MUST match your training config!
        output_mode='int',
        output_sequence_length=200,  # MUST match your training config!
        vocabulary=vocab
    )
    return vectorizer

st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text.strip():
        with st.spinner("Analyzing review..."):
            model = load_model()
            vectorizer = get_vectorizer()
            # Vectorize the raw text input
            input_data = vectorizer(np.array([review_text.strip()]))
            try:
                pred = model.predict(input_data)
                label = "Positive" if pred[0][0] > 0.5 else "Negative"
                confidence = float(pred[0][0]) * 100 if label == "Positive" else (1 - float(pred[0][0])) * 100
                st.subheader("Classification Results:")
                st.write(f"**Predicted Class:** {label}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.progress(min(confidence/100, 1.0))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please enter a review to classify.")

st.markdown("---")
st.caption("Developed by @sntsemilio")