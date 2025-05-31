import streamlit as st
import re
import random

# Page configuration
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

**How to use:**
1. Enter your movie review in the text area below
2. Click the "Classify Review" button
3. See the demo prediction results
""")

# User input section
st.header("Enter a Movie Review")

review_text = st.text_area("Type or paste your review here:", height=150)

# Function for demo predictions
def get_demo_prediction():
    classes = ["Positive", "Negative"]
    pred_idx = random.randint(0, 1)
    confidence = random.uniform(70, 95)
    return classes[pred_idx], confidence

if st.button("Classify Review"):
    if review_text:
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
st.caption("Developed by @sntsemilio")