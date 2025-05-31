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

# Simple text preprocessing function
def simple_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Check for model file and load if exists
model_path = 'text_classifier.pkl'
model_loaded = False

try:
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))
        model_loaded = True
        st.success("Model loaded successfully!")
    else:
        st.warning(f"Model file '{model_path}' not found. Using demo mode.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}. Using demo mode.")

# Function to make predictions
def predict_review(text, use_model=True):
    if use_model and model_loaded:
        # Use the actual model for prediction
        processed_text = simple_preprocess_text(text)
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        confidence = np.max(probabilities) * 100
        return prediction, confidence
    else:
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
            # Make prediction
            prediction, confidence = predict_review(review_text, model_loaded)
            
            # Display results
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            if not model_loaded:
                st.info("Note: This is a demo prediction since the model could not be loaded.")
    else:
        st.warning("Please enter a review to classify.")

# Information about model file location
st.sidebar.header("Troubleshooting")
st.sidebar.write(f"Looking for model at: {os.path.abspath(model_path)}")
st.sidebar.write("If the model file is missing, please make sure:")
st.sidebar.write("1. The file 'text_classifier.pkl' exists")
st.sidebar.write("2. It's in the same directory as this app")
st.sidebar.write("3. The app has permission to read the file")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")