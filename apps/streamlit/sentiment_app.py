import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Load or train the model
@st.cache_resource
def load_model_and_vectorizer():
    """Load or train the model and vectorizer"""
    try:
        # Try to load saved model first
        model = joblib.load('sentiment_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        return model, tfidf
    except:
        # Train model if not saved
        st.info("Training model... This may take a moment.")
        
        # Load and prepare data
        df = pd.read_csv('IMDB Dataset.csv')
        
        def clean_text(text):
            text = BeautifulSoup(str(text), 'html.parser').get_text()
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df['cleaned_review'] = df['review'].apply(clean_text)
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Train TF-IDF and model
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_tfidf = tfidf.fit_transform(df['cleaned_review'])
        
        model = BernoulliNB()
        model.fit(X_tfidf, df['label'])
        
        # Save the model and vectorizer
        joblib.dump(model, 'sentiment_model.joblib')
        joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
        
        return model, tfidf

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
    cleaned_text = clean_text(text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability) * 100
    
    return sentiment, confidence, probability

# Create confidence meter
def create_confidence_meter(confidence):
    """Create a visual confidence meter"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

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
    st.sidebar.info("""
    This app uses a Bernoulli Naive Bayes classifier trained on 50,000 IMDB movie reviews to predict sentiment.
    
    **Model Details:**
    - Algorithm: Bernoulli Naive Bayes
    - Features: TF-IDF (5000 features)
    - Training Data: 50,000 IMDB reviews
    - Expected Accuracy: ~85-87%
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
                    
                    # Confidence meter
                    st.markdown("### Confidence Score")
                    fig = create_confidence_meter(confidence)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability breakdown
                    st.markdown("### Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative'],
                        'Probability': [probability[1] * 100, probability[0] * 100]
                    })
                    
                    fig_prob = px.bar(prob_df, x='Sentiment', y='Probability', 
                                     color='Sentiment',
                                     color_discrete_map={'Positive': '#667eea', 'Negative': '#f5576c'},
                                     title="Sentiment Probability Distribution")
                    fig_prob.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
            else:
                st.warning("Please enter a movie review to analyze.")
    
    with col2:
        st.markdown("## Sample Reviews")
        st.markdown("Click on any sample to test:")
        
        for i, review in enumerate(SAMPLE_REVIEWS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                # This will be handled by the session state
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
            <div class="sample-review" onclick="selectSample({i})">
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
