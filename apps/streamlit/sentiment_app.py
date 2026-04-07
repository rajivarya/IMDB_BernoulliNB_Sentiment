import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import joblib
import os
import time

# Set page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .sample-review {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-review:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .stTextArea > div > textarea {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        # Try to load from current directory first
        model = joblib.load('sentiment_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        return model, tfidf
    except:
        try:
            # Try to load from apps/streamlit directory
            model = joblib.load('apps/streamlit/sentiment_model.joblib')
            tfidf = joblib.load('apps/streamlit/tfidf_vectorizer.joblib')
            return model, tfidf
        except:
            # If model files don't exist, create a simple demo model
            st.warning("Model files not found. Using demo mode.")
            return None, None

# Clean text function
def clean_text(text):
    """Clean and preprocess text"""
    text = BeautifulSoup(str(text), 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment
def predict_sentiment(text, model, tfidf):
    """Predict sentiment for given text"""
    if model is None or tfidf is None:
        # Demo mode - simple keyword-based prediction
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'brilliant', 'fantastic', 'love', 'best', 'beautiful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'boring', 'poor', 'disaster']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
            confidence = min(90, 60 + pos_count * 5)
        else:
            sentiment = "Negative"
            confidence = min(90, 60 + neg_count * 5)
            
        return sentiment, confidence, [0.4, 0.6] if sentiment == "Positive" else [0.6, 0.4]
    
    cleaned_text = clean_text(text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability) * 100
    
    return sentiment, confidence, probability

# Sample reviews for testing
SAMPLE_REVIEWS = [
    "This movie was absolutely brilliant! The acting was superb and the storyline kept me engaged throughout.",
    "I was really disappointed with this film. The plot was predictable and the acting was terrible.",
    "An okay movie with some good moments but overall quite forgettable.",
    "One of the best movies I've ever seen! Highly recommended to everyone.",
    "Complete waste of time. Don't bother watching this garbage.",
    "The cinematography was stunning, but the story left much to be desired."
]

# Main application
def main():
    # Load model
    model, tfidf = load_model_and_vectorizer()
    
    # Header
    st.markdown('<h1 class="main-header">IMDB Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## About")
    if model is not None:
        st.sidebar.info("""
        This app uses a Bernoulli Naive Bayes classifier trained on IMDB movie reviews to predict sentiment.
        
        **Model Details:**
        - Algorithm: Bernoulli Naive Bayes
        - Features: TF-IDF (5000 features)
        - Expected Accuracy: ~85%
        """)
    else:
        st.sidebar.warning("""
        **Demo Mode Active**
        
        The app is running in demo mode with keyword-based sentiment analysis.
        For full ML functionality, ensure model files are properly loaded.
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## Enter Your Movie Review")
        
        # Text input
        user_input = st.text_area(
            "Type or paste your movie review here:",
            height=150,
            placeholder="Enter your movie review here..."
        )
        
        # Predict button
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    time.sleep(1)  # Add a small delay for better UX
                    sentiment, confidence, probability = predict_sentiment(user_input, model, tfidf)
                    
                    # Display results
                    if sentiment == "Positive":
                        st.markdown(f"""
                        <div class="sentiment-positive">
                            <h2>Positive Sentiment</h2>
                            <p style="font-size: 1.5rem;">This review appears to be positive! {confidence:.1f}% confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <h2>Negative Sentiment</h2>
                            <p style="font-size: 1.5rem;">This review appears to be negative! {confidence:.1f}% confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display metrics
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Sentiment", sentiment)
                    with col_metric2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Probability breakdown
                    st.markdown("### Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative'],
                        'Probability': [probability[1] * 100, probability[0] * 100]
                    })
                    
                    st.bar_chart(prob_df.set_index('Sentiment'))
                    
            else:
                st.warning("Please enter a movie review to analyze.")
    
    with col2:
        st.markdown("## Sample Reviews")
        st.markdown("Click on any sample to test:")
        
        for i, review in enumerate(SAMPLE_REVIEWS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                st.session_state.selected_review = review
        
        # Display selected review if any
        if 'selected_review' in st.session_state:
            st.text_area("Selected Review:", st.session_state.selected_review, height=100, key="selected_display")
    
    # Sample reviews section at bottom
    st.markdown("---")
    st.markdown("## Try These Sample Reviews")
    
    col1, col2, col3 = st.columns(3)
    
    sample_cols = [col1, col2, col3]
    for i, review in enumerate(SAMPLE_REVIEWS[:6]):
        with sample_cols[i % 3]:
            st.markdown(f"""
            <div class="sample-review">
                <strong>Sample {i+1}:</strong><br>
                <small>{review[:100]}...</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Use Sample {i+1}", key=f"bottom_sample_{i}", use_container_width=True):
                st.session_state.selected_review = review
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit | Powered by Bernoulli Naive Bayes | Trained on IMDB Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()