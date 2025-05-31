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
## Demo Mode
This application is currently running in demonstration mode. 
The actual model requires TensorFlow/Keras which is not compatible with the current Python environment.
""")

st.markdown("""
This app classifies IMDB movie reviews based on textual content.

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

if st.button("Classify Review"):
    if not model:
        st.error("Model is not loaded. Cannot classify the review.")
    elif not review_text.strip():
        st.warning("Please enter a review to classify.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                # Process text
                processed_text = simple_preprocess_text(review_text)
                
                # Make prediction using model
                prediction = model.predict([processed_text])[0]
                probabilities = model.predict_proba([processed_text])[0]
                confidence = np.max(probabilities) * 100
                
                # Display results
                st.subheader("Classification Results:")
                st.write(f"**Predicted Sentiment:** {prediction}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                # Progress bar visualization
                st.progress(min(confidence/100, 1.0))
                
            except Exception:
                st.error("Unable to process the review. The model could not be loaded or used.")
    else:
        st.warning("Please enter a review to classify.")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")
