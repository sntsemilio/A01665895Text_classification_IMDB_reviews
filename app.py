import streamlit as st
import pickle
import re
import numpy as np

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

# Load the model
model = pickle.load(open('text_classifier.pkl', 'rb'))

# Simple text preprocessing function
def simple_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# User input section
st.header("Enter a Movie Review")

review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text:
        with st.spinner("Analyzing review..."):
            # Preprocess the text
            processed_text = simple_preprocess_text(review_text)
            
            # Make prediction
            prediction = model.predict([processed_text])[0]
            probabilities = model.predict_proba([processed_text])[0]
            
            # Get maximum probability
            confidence = np.max(probabilities) * 100
            
            # Display results
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")