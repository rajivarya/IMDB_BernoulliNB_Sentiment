import sys
import os

# Add current directory to path
sys.path.insert(0, '/Users/reshurajrivansh/.codeium/windsurf/github-deploy')

print('=== DEPLOYMENT DIAGNOSIS ===')
print('Current working directory:', os.getcwd())
print('Python path:', sys.path[:3])

# Check if model files exist
print('\nChecking for model files...')
model_files = ['sentiment_model.joblib', 'tfidf_vectorizer.joblib']
for file in model_files:
    if os.path.exists(file):
        print(f'✅ {file} exists')
    else:
        print(f'❌ {file} missing')

# Test Streamlit app import
print('\nTesting Streamlit app...')
try:
    import streamlit as st
    print('✅ Streamlit imported successfully')
except ImportError as e:
    print(f'❌ Streamlit import failed: {e}')

# Test other dependencies
try:
    import pandas as pd
    print('✅ pandas imported successfully')
except ImportError as e:
    print(f'❌ pandas import failed: {e}')

try:
    import numpy as np
    print('✅ numpy imported successfully')
except ImportError as e:
    print(f'❌ numpy import failed: {e}')

try:
    import joblib
    print('✅ joblib imported successfully')
except ImportError as e:
    print(f'❌ joblib import failed: {e}')

print('\n=== DIAGNOSIS COMPLETE ===')