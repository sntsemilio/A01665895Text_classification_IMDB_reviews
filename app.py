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
st.markdown("""
## Demo Mode
This application is currently running in demonstration mode. 
The actual model requires TensorFlow/Keras which is not compatible with the current Python environment.
""")

# Model description
st.markdown("""
### About This App
This app demonstrates how IMDB movie reviews can be classified based on their textual content.
In a fully functional version, it would use a machine learning model trained on thousands of reviews.

**How to use:**
1. Enter your movie review in the text area below
2. Click the "Classify Review" button
3. See the demonstration results
""")

# User input section
st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

# Function for demo predictions
def get_demo_prediction():
    # Simple keyword-based "prediction" for demo purposes
    review = review_text.lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'enjoyed', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'waste', 'hate', 'worst', 'boring']
    
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
            st.subheader("Demo Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            # Display demo notification
            st.info("Note: This is a demonstration prediction based on simple keyword matching, not an actual ML model.")
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")