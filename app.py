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

# App title
st.title("IMDB Review Text Classifier")

# Information about demo mode
st.info("This app is running in demo mode as the classification model requires Keras which is not installed.")

# Model description
st.markdown("""
This app normally classifies IMDB movie reviews based on textual content.
Currently showing demo predictions only.

How to use:
1. Enter your movie review in the text area below
2. Click the "Classify Review" button
3. See the demonstration results
""")

def simple_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Silently try to load the model
try:
    import pickle
    model = pickle.load(open('text_classifier.pkl', 'rb'))
except Exception as e:
    model = None

st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

# Function for demo predictions
def get_demo_prediction():
    classes = ["Positive", "Negative"]
    pred_idx = random.randint(0, 1)
    confidence = random.uniform(70, 95)
    return classes[pred_idx], confidence

if st.button("Classify Review"):
    if not model:
        st.error("Model is not loaded. Cannot classify the review.")
    elif not review_text.strip():
        st.warning("Please enter a review to classify.")
    else:
        with st.spinner("Analyzing review..."):
            # Generate demo prediction
            prediction, confidence = get_demo_prediction()
            
            # Display results
            st.subheader("Demo Classification Results:")
            st.write(f"**Predicted Class:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            st.markdown("**Note:** This is a demonstration prediction only.")
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.markdown("### How it would work with a real model")
st.markdown("""
In a production environment with TensorFlow/Keras available:
1. Text would be preprocessed (tokenized, normalized)
2. Features would be extracted using techniques like TF-IDF or word embeddings
3. The trained model would predict sentiment or categories
4. Results would be shown with actual model confidence scores
""")

st.caption("Developed by @sntsemilio")
