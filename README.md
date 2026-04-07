# IMDB Sentiment Analysis Project

## Project Overview
This project implements a complete IMDB movie review sentiment classification system using Bernoulli Naive Bayes.

## Deployment Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/github-deploy
cd github-deploy
```

### 2. Run Training Notebook
```bash
cd github-deploy/notebooks
jupyter nbconvert --to python --execute IMDB_BernoulliNB_Sentiment.ipynb
```

### 3. Deploy Web App
```bash
cd github-deploy
streamlit run apps/streamlit/sentiment_app.py
```

## Files Structure
```
github-deploy/
├── README.md
├── notebooks/
│   └── IMDB_BernoulliNB_Sentiment.ipynb
├── apps/
│   └── streamlit/
│       ├── sentiment_app.py
│       └── requirements.txt
└── workflows/
    └── deploy.yml
```

## Features

### Jupyter Notebook
- **Complete ML pipeline** with 8 cells
- **Data loading** from Google Sheets CSV
- **Text cleaning** with HTML tag removal
- **Label encoding** (positive→1, negative→0)
- **Train-test split** (80/20, random_state=42)
- **TF-IDF vectorization** (max_features=5000)
- **BernoulliNB training** and evaluation
- **Expected accuracy**: ~85%

### Streamlit Web App
- **Interactive interface** for sentiment analysis
- **Real-time predictions** with confidence scores
- **Sample reviews** for testing
- **Model information** display
- **Responsive design** with gradient backgrounds
- **Error handling** for missing model files

### Deployment Options

#### Option 1: Streamlit Cloud
```bash
streamlit run apps/streamlit/sentiment_app.py
```

#### Option 2: Heroku
```bash
heroku create app-name
git push heroku main
```

#### Option 3: Railway/Render/Fly.io
Similar to Streamlit Cloud deployment

## Getting Started
1. Update the GitHub repository URL in the notebook
2. Push all files to GitHub
3. Deploy using preferred method

## Model Files
The training notebook creates:
- `sentiment_model.joblib` - Trained BernoulliNB model
- `tfidf_vectorizer.joblib` - Fitted TF-IDF vectorizer

These are automatically loaded by the Streamlit app for predictions.