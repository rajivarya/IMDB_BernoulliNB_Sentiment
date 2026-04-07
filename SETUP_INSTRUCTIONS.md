# GitHub Repository Setup Instructions

## Step 1: Create GitHub Repository
1. Go to https://github.com/rajivarya?tab=repositories
2. Click "New" or "Create repository"
3. Repository name: `github-deploy`
4. Description: `IMDB Movie Review Sentiment Analysis using Bernoulli Naive Bayes`
5. Make it **Public**
6. Click "Create repository"

## Step 2: Push Code to GitHub
After creating the repository, run these commands:

```bash
cd /Users/reshurajrivansh/.codeium/windsurf/github-deploy
git push -u origin main
```

## Step 3: Deploy Streamlit App
Once the code is pushed to GitHub:

```bash
cd /Users/reshurajrivansh/.codeium/windsurf/github-deploy
streamlit run apps/streamlit/sentiment_app.py --server.headless=true --server.port=8502
```

## Repository Structure
```
github-deploy/
|-- README.md
|-- .github/workflows/deploy.yml
|-- apps/streamlit/sentiment_app.py
|-- apps/streamlit/requirements.txt
|-- notebooks/IMDB_BernoulliNB_Sentiment.ipynb
```

## Deployment Options
1. **Streamlit Cloud**: Connect GitHub repository
2. **Heroku**: Connect GitHub repository
3. **Railway**: Connect GitHub repository

## Next Steps
1. Create the GitHub repository at https://github.com/rajivarya?tab=repositories
2. Push the code using the commands above
3. Deploy the Streamlit app