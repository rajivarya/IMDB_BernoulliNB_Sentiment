import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="",
    layout="wide"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        model = joblib.load('sentiment_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        return model, tfidf
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first.")
        return None, None

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text data"""
    text = BeautifulSoup(str(text), 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment
def predict_sentiment(text, model, tfidf):
    """Predict sentiment for given text"""
    if model is None or tfidf is None:
        return None, None
    
    cleaned_text = clean_text(text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    confidence = max(model.predict_proba(text_tfidf)[0]) * 100
    
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    
    return sentiment, confidence, prediction[0]

# Main app
def main():
    st.title("IMDB Movie Review Sentiment Analysis")
    st.markdown("---")
    
    # Load model
    model, tfidf = load_model()
    if model is None:
        st.stop()
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.subheader("Model Information")
        st.info("Bernoulli Naive Bayes classifier trained on 50,000 IMDB reviews")
        st.info("Expected accuracy: ~85%")
    
    # Main interface
    st.subheader("Analyze Review")
    
    # Text input
    user_input = st.text_area(
        "Enter your movie review:",
        height=150,
        placeholder="Type or paste your review here...",
        help="The model works best with reviews that are similar to the training data style."
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence, prediction = predict_sentiment(user_input, model, tfidf)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">😊 Positive Sentiment</h2>
                        <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">😔 Negative Sentiment</h2>
                        <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Prediction", prediction[0])
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Sample reviews
                st.subheader("Try Sample Reviews")
                
                sample_reviews = [
                    "This movie was absolutely brilliant! The acting was superb and the storyline was captivating.",
                    "Terrible waste of time. The plot was predictable and the acting was wooden.",
                    "An amazing cinematic experience with stunning visuals and powerful performances.",
                    "I was really disappointed with this film. It didn't live up to the hype."
                ]
                
                if st.button("Load Sample", key="load_sample"):
                    selected_review = st.selectbox("Choose a sample review:", sample_reviews)
                    user_input = selected_review
        
        else:
            st.warning("Please enter a review to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with Streamlit • Powered by Bernoulli Naive Bayes • Trained on IMDB Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()