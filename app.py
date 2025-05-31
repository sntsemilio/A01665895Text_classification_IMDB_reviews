import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Load the trained model
try:
    model = pickle.load(open('text_classifier.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'text_classifier.pkl' not found. Please ensure the model file is in the same directory as this app.")
    st.stop()

# Text preprocessing function
def preprocess_text(text):
    # Clean the text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    
    # Remove stopwords and stem
    all_stopwords = stopwords.words('english')
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    
    # Join the words back into a string
    text = ' '.join(text)
    return text

# Streamlit app
st.title('IMDB Review Text Classifier')

st.markdown("""
This app classifies IMDB movie reviews based on the text content.
Enter a review below and click 'Classify' to see the prediction.
""")

# User input
review_text = st.text_area("Enter your IMDB review here:", height=150)

# Prediction
if st.button('Classify'):
    if review_text:
        # Preprocess the text
        preprocessed_text = preprocess_text(review_text)
        
        # Make prediction
        try:
            prediction = model.predict([preprocessed_text])[0]
            probabilities = model.predict_proba([preprocessed_text])[0]
            
            # Get class names if available in the model
            class_names = getattr(model, 'classes_', None)
            if class_names is not None:
                predicted_class = class_names[prediction]
                st.success(f"Predicted class: {predicted_class}")
            else:
                st.success(f"Predicted class: {prediction}")
            
            # Display probabilities
            st.write("Classification probabilities:")
            if class_names is not None:
                prob_df = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probabilities
                })
            else:
                prob_df = pd.DataFrame({
                    'Class': [f"Class {i}" for i in range(len(probabilities))],
                    'Probability': probabilities
                })
            
            prob_df = prob_df.sort_values('Probability', ascending=False)
            st.dataframe(prob_df)
            
            # Create a bar chart
            st.bar_chart(prob_df.set_index('Class'))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please make sure the model was trained correctly and includes the necessary components.")
    else:
        st.warning("Please enter a review to classify.")

st.markdown("---")
st.markdown("### About")
st.markdown("""
This app uses a machine learning model trained on IMDB movie reviews to classify text.
The model analyzes the review and predicts its class based on the textual content.
""")