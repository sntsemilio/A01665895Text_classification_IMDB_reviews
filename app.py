import streamlit as st
import re
import os
import numpy as np
import pickle

st.set_page_config(
    page_title="IMDB Review Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Review Text Classifier")

st.markdown("""
This app classifies IMDB movie reviews based on sentiment analysis.

How to use:
1. Enter your movie review in the text area below
2. Click the "Classify Review" button
3. See the classification results
""")

def simple_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

model = None
model_path = 'text_classifier.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error("Model file not found. Please make sure 'text_classifier.pkl' is in the app directory.")

st.header("Enter a Movie Review")

review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if not model:
        st.error("Model is not loaded. Cannot classify the review.")
    elif not review_text.strip():
        st.warning("Please enter a review to classify.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                processed_text = simple_preprocess_text(review_text)
                prediction = model.predict([processed_text])[0]
                probabilities = model.predict_proba([processed_text])[0]
                confidence = np.max(probabilities) * 100

                st.subheader("Classification Results:")
                st.write(f"Predicted Sentiment: {prediction}")
                st.write(f"Confidence: {confidence:.2f}%")

                st.progress(min(confidence / 100, 1.0))

            except Exception as e:
                st.error(f"Error during classification: {e}")

st.markdown("---")
st.caption("Developed by @sntsemilio")
