import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="IMDB Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Review Sentiment Analysis")
st.markdown("""
This app classifies IMDB movie reviews as **Positive** or **Negative** using a machine learning model trained on review text data.
""")

@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('A01665895_imdb_review_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('imdb_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}. Please ensure both .pkl files are present.")
        st.stop()

def preprocess(text, vectorizer):
    return vectorizer.transform([text])

# User input section
st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text.strip():
        with st.spinner("Analyzing review..."):
            model, vectorizer = load_model_and_vectorizer()
            X_input = preprocess(review_text, vectorizer)
            pred_proba = model.predict_proba(X_input)
            pred = np.argmax(pred_proba)
            confidence = float(np.max(pred_proba)) * 100
            sentiment = "Positive" if pred == 1 else "Negative"

            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(min(confidence/100, 1.0))
    else:
        st.warning("Please enter a review to classify.")

st.markdown("---")
st.caption("Developed by @sntsemilio")