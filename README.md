# Persian Banking Sentiment Analysis

## Overview

This project performs sentiment analysis on Persian user comments about banking services collected from Cafe Bazaar.

## Features

- ✅ Persian text preprocessing and normalization
- ✅ Web scraping from Cafe Bazaar
- ✅ Multiple feature extraction methods (TF-IDF, Word2Vec, BERT)
- ✅ Various ML models (Logistic Regression, Neural Networks, ParsBERT)
- ✅ Comprehensive evaluation and analysis
- ✅ Banking service category analysis

## Project Structure

```
persian_banking_sentiment/
├── data/                 # Data files
├── src/                  # Source code
├── models/              # Saved models
├── results/             # Results and reports
├── notebooks/           # Jupyter notebooks
├── scripts/             # Utility scripts
├── tests/               # Test files
└── docs/                # Documentation
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run setup script: `python setup.py`
3. Download Persian resources: `python scripts/setup_persian_resources.py`

## Usage

1. **Data Collection**: `python src/data_collection/cafe_bazaar_scraper.py`
2. **Preprocessing**: `python src/preprocessing/persian_cleaner.py`
3. **Training**: `python scripts/train_all_models.py`
4. **Evaluation**: `python scripts/generate_final_report.py`

## Results

- Baseline accuracy: 70%+
- Advanced models: 85%+ accuracy
- Comprehensive banking insights
- Error analysis and recommendations

## Technologies

- Python 3.8+
- Hazm (Persian NLP)
- Scikit-learn
- PyTorch
- Transformers (ParsBERT)
- Pandas, NumPy
- Matplotlib, Seaborn

## Author

Sina Sepahvand - Master's Student in AI
NLP Course Project
