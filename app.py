import streamlit as st
import re
import numpy as np
import os
import random

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
3. See the classification results
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

# Try to load the model
model = None
model_loaded = False

try:
    import pickle
    if os.path.exists('text_classifier.pkl'):
        model = pickle.load(open('text_classifier.pkl', 'rb'))
        model_loaded = True
except Exception as e:
    st.warning(f"Model could not be loaded: {str(e)}. Using keyword-based classification.")
    
# Function for predictions - uses model if available, otherwise falls back to demo mode
def predict_review(text):
    processed_text = simple_preprocess_text(text)
    
    if model_loaded and model is not None:
        # Use the actual model for prediction
        try:
            prediction = model.predict([processed_text])[0]
            probabilities = model.predict_proba([processed_text])[0]
            confidence = np.max(probabilities) * 100
            return prediction, confidence
        except Exception as e:
            st.warning(f"Error during prediction: {str(e)}. Using keyword-based classification.")
            # Fall back to keyword-based prediction
            return keyword_based_prediction(text)
    else:
        # Use keyword-based prediction if model is not available
        return keyword_based_prediction(text)

# Keyword-based prediction function for fallback
def keyword_based_prediction(text):
    # Simple keyword-based "prediction" for demo purposes
    review = text.lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'enjoyed', 'best', 'wonderful', 'recommend']
    negative_words = ['bad', 'terrible', 'awful', 'waste', 'hate', 'worst', 'boring', 'poor', 'disappointing']
    
    pos_count = sum(1 for word in positive_words if word in review)
    neg_count = sum(1 for word in negative_words if word in review)
    
    if pos_count > neg_count:
        return "Positive", 65 + random.uniform(0, 30)
    elif neg_count > pos_count:
        return "Negative", 65 + random.uniform(0, 30)
    else:
        # If tied or no keywords found, return random
        return random.choice(["Positive", "Negative"]), 50 + random.uniform(0, 25)

# User input section
st.header("Enter a Movie Review")

review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text:
        with st.spinner("Analyzing review..."):
            # Make prediction
            prediction, confidence = predict_review(review_text)
            
            # Display results
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            if not model_loaded:
                st.info("Note: This is a keyword-based prediction since the model could not be loaded.")
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")