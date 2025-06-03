import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="IMDB Review Classifier",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Review Classifier")
st.markdown("""
This app classifies IMDB movie reviews as **Positive** or **Negative** using a deep learning model trained on review text data.
""")

@st.cache_resource
def load_model():
    try:
        with open("text_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the file 'text_classifier.pkl' is present.")
        st.stop()

st.header("Enter a Movie Review")
review_text = st.text_area("Type or paste your review here:", height=150)

if st.button("Classify Review"):
    if review_text.strip():
        with st.spinner("Analyzing review..."):
            model = load_model()
            try:
                # For models that require a numpy array as input
                pred = model.predict([review_text])
                # If output is probability, threshold at 0.5
                if pred.shape[-1] == 1 or len(pred.shape) == 1:
                    label = "Positive" if float(pred[0]) > 0.5 else "Negative"
                    confidence = float(pred[0]) * 100 if label == "Positive" else (1 - float(pred[0])) * 100
                else:
                    # If output is class logits
                    idx = np.argmax(pred)
                    label = "Positive" if idx == 1 else "Negative"
                    confidence = float(pred[0][idx]) * 100
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                st.stop()

            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(min(confidence/100, 1.0))
    else:
        st.warning("Please enter a review to classify.")

st.markdown("---")
st.caption("Developed by @sntsemilio")