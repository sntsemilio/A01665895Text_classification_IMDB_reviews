import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="IMDB Review Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Review Classifier")
st.markdown("""
This app classifies IMDB movie reviews as **Good** or **Bad** using a deep learning model trained on review text data.
""")

@st.cache_resource
def load_model():
    try:
        with open('text_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the file 'text_classifier.pkl' is present.")
        st.stop()

# User input section
st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text.strip():
        with st.spinner("Analyzing review..."):
            model = load_model()
            # If your model expects simple text input (e.g., it's a pipeline with text preprocessing)
            try:
                pred = model.predict([review_text])
                # If model outputs class indices (e.g., 0/1), map them to labels
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    label = classes[pred[0]]
                else:
                    label = "Good" if pred[0] == 1 else "Bad"
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                st.stop()

            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {label}")
    else:
        st.warning("Please enter a review to classify.")

st.markdown("---")
st.caption("Developed by @sntsemilio")