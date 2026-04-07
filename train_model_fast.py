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

# Use a smaller subset for faster training
df_subset = df.sample(n=5000, random_state=42)
print(f"Using subset: {df_subset.shape}")

def clean_text(text):
    text = BeautifulSoup(str(text), 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Applying text cleaning...")
df_subset['cleaned_review'] = df_subset['review'].apply(clean_text)

print("Applying label encoding...")
df_subset['label'] = df_subset['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df_subset['cleaned_review'], df_subset['label'], test_size=0.2, random_state=42
)

print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
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