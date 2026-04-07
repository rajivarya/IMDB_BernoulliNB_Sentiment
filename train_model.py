import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset from Google Sheets...")
df = pd.read_csv('https://docs.google.com/spreadsheets/d/13-DkhfR0sgIeleUEZJMzFXK4hYtZbSKMzXYJvUT2gdA/export?format=csv')
print(f"Dataset loaded: {df.shape}")

def clean_text(text):
    text = BeautifulSoup(str(text), 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Applying text cleaning...")
df['cleaned_review'] = df['review'].apply(clean_text)

print("Applying label encoding...")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['label'], test_size=0.2, random_state=42
)

print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training BernoulliNB model...")
model = BernoulliNB()
model.fit(X_train_tfidf, y_train)

print("Making predictions...")
y_pred = model.predict(X_test_tfidf)

print("Calculating accuracy...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("Saving model files...")

# Save model and vectorizer to apps/streamlit/
joblib.dump(model, '../apps/streamlit/sentiment_model.joblib')
joblib.dump(tfidf, '../apps/streamlit/tfidf_vectorizer.joblib')

print("Training completed and model files saved to apps/streamlit/")