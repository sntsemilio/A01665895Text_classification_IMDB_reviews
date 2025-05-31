import streamlit as st
import pickle
import re
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="IMDB Review Classifier",
    page_icon="🎬",
    layout="centered"
)

# App title
st.title("IMDB Review Text Classifier")

# Model description
st.markdown("""
This app classifies IMDB movie reviews based on textual content.

**How to use:**
1. Enter your movie review in the text area below
2. Click the "Classify Review" button
3. See the prediction results
""")

# Show error message about missing Keras module
st.error("Error loading model: No module named 'keras'. Using demo mode.")

# Simple text preprocessing function
def simple_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to make predictions (demo mode only)
def predict_review(text):
    # Demo mode - return random prediction
    import random
    classes = ["Positive", "Negative"]
    pred_idx = random.randint(0, 1)
    confidence = random.uniform(70, 95)
    return classes[pred_idx], confidence

# User input section
st.header("Enter a Movie Review")

review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text:
        with st.spinner("Analyzing review..."):
            # Make prediction (demo mode)
            prediction, confidence = predict_review(review_text)
            
            # Display results
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            st.info("Note: This is a demo prediction since the model could not be loaded.")
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")