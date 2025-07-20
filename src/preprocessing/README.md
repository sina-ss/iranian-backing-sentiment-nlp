# Persian Text Preprocessing System

A comprehensive text preprocessing pipeline specifically designed for Persian banking app comments. This system handles noise removal, normalization, tokenization, and advanced NLP tasks optimized for Persian text.

## 🎯 Features

### Core Capabilities

* **Multi-level Cleaning** : Light, medium, and heavy cleaning options
* **Persian Text Normalization** : Character mapping, diacritic removal, number normalization
* **Emoji Processing** : Convert emojis to Persian text or remove them completely
* **Noise Removal** : URLs, emails, phone numbers, mentions, hashtags
* **Tokenization** : Persian-aware word tokenization using Hazm
* **Stemming & Lemmatization** : Advanced morphological analysis
* **Stopword Removal** : Comprehensive Persian stopwords list
* **Batch Processing** : Efficient processing of large datasets

### Persian-Specific Features

* **Character Normalization** : Arabic to Persian character conversion
* **Diacritic Handling** : Optional removal of Persian diacritics
* **Number Normalization** : Persian to English number conversion
* **Banking Terminology** : Specialized handling of banking terms

## 📦 Installation & Setup

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Run setup script
python scripts/setup_preprocessing.py
```

### 2. External Resources

The system uses two external files:

* `data/external/persian_stopwords.txt` - Persian stopwords (auto-generated)
* `data/external/persian_emoji_mapping.json` - Emoji to Persian text mapping

## 🚀 Quick Start

### Command Line Usage

```bash
# Basic preprocessing with multiple versions
python scripts/preprocessing_pipeline.py

# Custom input file
python scripts/preprocessing_pipeline.py --input your_comments.csv

# Single version with specific settings
python scripts/preprocessing_pipeline.py --single-version --cleaning-level heavy --stem

# Process labeled data
python scripts/preprocessing_pipeline.py --labeled-input data/processed/labeled_comments.csv
```

### Programmatic Usage

```python
from src.preprocessing.persian_cleaner import PersianTextCleaner

# Initialize cleaner
cleaner = PersianTextCleaner()

# Clean single text
text = "سلام! اپ بانک عالیه 😊 www.example.com"
cleaned = cleaner.clean_text(text, level='medium')
print(cleaned)  # Output: "سلام اپ بانک عالیه خوشحال"

# Full preprocessing pipeline
tokens = cleaner.preprocess_text(
    text,
    tokenize=True,
    remove_stopwords=True,
    stem=True
)
print(tokens)  # Output: ['سلام', 'بانک', 'عالی', 'خوشحال']

# Process DataFrame
import pandas as pd
df = pd.read_csv('comments.csv')
processed_df = cleaner.process_dataframe(df, text_column='comment')
```

## 🛠️ Cleaning Levels

### Light Cleaning

* Basic whitespace normalization
* Minimal noise removal
* Preserve most content

```python
cleaned = cleaner.clean_text(text, level='light')
```

### Medium Cleaning (Recommended)

* Remove URLs, emails, phone numbers
* Convert emojis to Persian text
* Persian character normalization
* Punctuation cleaning

```python
cleaned = cleaner.clean_text(text, level='medium')
```

### Heavy Cleaning

* All medium cleaning features
* Remove English characters
* Aggressive noise removal
* Maximum normalization

```python
cleaned = cleaner.clean_text(text, level='heavy')
```

## 📊 Processing Pipeline

The system creates multiple preprocessed versions automatically:

1. **Light Version** (`comments_light_processed.csv`)
   * Minimal cleaning
   * Good for context preservation
2. **Medium Version** (`comments_medium_processed.csv`)
   * Balanced cleaning
   * **Recommended for most ML models**
3. **Heavy Stem Version** (`comments_heavy_stem_processed.csv`)
   * Aggressive cleaning + stemming
   * Good for feature extraction
4. **Heavy Lemma Version** (`comments_heavy_lemma_processed.csv`)
   * Aggressive cleaning + lemmatization
   * Good for semantic analysis

## 🔧 Configuration

### Main Configuration (`config.py`)

```python
PREPROCESSING_CONFIG = {
    "remove_english": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_phone_numbers": True,
    "remove_emojis": False,  # Convert instead
    "normalize_persian": True,
    "remove_diacritics": True,
    "min_word_length": 2,
    "max_word_length": 50,
    "min_comment_length": 10,
    "max_comment_length": 500
}
```

### Persian Language Settings

```python
PERSIAN_CONFIG = {
    "custom_stopwords": [
        "بانک", "اپ", "اپلیکیشن", "موبایل"
    ],
    "banking_terms": {
        "positive": ["عالی", "راحت", "سریع"],
        "negative": ["بد", "کند", "مشکل"],
        "neutral": ["متوسط", "عادی", "نظر"]
    }
}
```

## 📈 Analysis & Monitoring

### Processing Statistics

```python
# Get processing statistics
stats = cleaner.get_statistics()
print(stats['processing_stats'])
# Output: {'processed_texts': 100, 'removed_urls': 15, ...}
```

### Batch Processing Report

After running the pipeline, check:

* `results/reports/preprocessing_report.json` - Detailed processing report
* `logs/preprocessing.log` - Processing logs

### Analysis Notebook

Use the provided Jupyter notebook for detailed analysis:

```bash
jupyter notebook notebooks/02_preprocessing_analysis.ipynb
```

## 🎨 Emoji Processing

The system includes comprehensive emoji mapping:

```json
{
  "positive_emojis": {
    "😊": "خوشحال",
    "👍": "تایید",
    "❤": "قلب_قرمز"
  },
  "negative_emojis": {
    "😠": "عصبانی",
    "👎": "رد",
    "💔": "قلب_شکسته"
  },
  "banking_specific": {
    "💰": "پول",
    "🏧": "خودپرداز",
    "🏦": "بانک"
  }
}
```

## 🔍 Examples

### Basic Text Cleaning

```python
# Input
text = "سلام! اپ بانک عالیه 😊 خیلی راحت کار میکنه 👍 www.bank.com"

# Light cleaning
light = cleaner.clean_text(text, level='light')
# Output: "سلام! اپ بانک عالیه 😊 خیلی راحت کار میکنه 👍 www.bank.com"

# Medium cleaning  
medium = cleaner.clean_text(text, level='medium')
# Output: "سلام اپ بانک عالیه خوشحال خیلی راحت کار میکنه تایید"

# Heavy cleaning
heavy = cleaner.clean_text(text, level='heavy')
# Output: "سلام اپ بانک عالیه خوشحال خیلی راحت کار میکنه تایید"
```

### Advanced Processing

```python
# Full pipeline with tokenization
tokens = cleaner.preprocess_text(
    text,
    tokenize=True,
    remove_stopwords=True,
    stem=True
)
# Output: ['سلام', 'بانک', 'عالی', 'خوشحال', 'راحت', 'کار', 'تایید']
```

### DataFrame Processing

```python
import pandas as pd

# Load data
df = pd.read_csv('banking_comments.csv')
print(df.head())
#     id                              comment  rating
# 0    1    سلام! اپ بانک عالیه 😊 سریع است       5
# 1    2         برنامه خراب است 😡 کار نمیکنه       1

# Process DataFrame
processed_df = cleaner.process_dataframe(df, text_column='comment')
print(processed_df[['comment', 'comment_cleaned']].head())
#                              comment                    comment_cleaned
# 0    سلام! اپ بانک عالیه 😊 سریع است    سلام اپ بانک عالیه خوشحال سریع است
# 1         برنامه خراب است 😡 کار نمیکنه         برنامه خراب است عصبانی کار نمیکنه
```

## 📊 Performance Metrics

### Processing Speed

* **Light cleaning** : ~1000 comments/second
* **Medium cleaning** : ~500 comments/second
* **Heavy cleaning** : ~200 comments/second

### Memory Usage

* ~1MB per 1000 comments
* Efficient batch processing
* Automatic garbage collection

## 🔧 Troubleshooting

### Common Issues

#### 1. Hazm Installation Problems

```bash
# Install with specific version
pip install hazm==0.7.0

# Or install dependencies manually
pip install nltk libwapiti
```

#### 2. Unicode/Encoding Issues

```python
# Ensure UTF-8 encoding when reading files
df = pd.read_csv('file.csv', encoding='utf-8')
```

#### 3. Memory Issues with Large Files

```python
# Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    processed_chunk = cleaner.process_dataframe(chunk)
    # Save or append results
```

#### 4. Slow Processing

```python
# Use light cleaning for speed
cleaner.clean_text(text, level='light')

# Or disable expensive operations
cleaner.config['remove_english'] = False
```

### Error Messages

| Error                          | Solution                                                   |
| ------------------------------ | ---------------------------------------------------------- |
| `hazm not available`         | Install hazm:`pip install hazm`                          |
| `Emoji mapping not found`    | Run setup script:`python scripts/setup_preprocessing.py` |
| `Column 'comment' not found` | Check CSV column names                                     |
| `Memory error`               | Process data in smaller chunks                             |

## 🔄 Integration with ML Pipeline

### For Feature Extraction

```python
# Use medium version for TF-IDF
df_medium = pd.read_csv('data/processed/comments_medium_processed.csv')
texts = df_medium['comment_processed'].tolist()

# TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(texts)
```

### For Word Embeddings

```python
# Use heavy stem version for Word2Vec
df_stem = pd.read_csv('data/processed/comments_heavy_stem_processed.csv')
sentences = [text.split() for text in df_stem['comment_processed']]

# Train Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=200, window=5, min_count=2)
```

### For BERT/Transformer Models

```python
# Use light or medium version to preserve context
df_light = pd.read_csv('data/processed/comments_light_processed.csv')
texts = df_light['comment_cleaned'].tolist()

# Use with transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-base-parsbert-uncased')
tokens = tokenizer(texts, padding=True, truncation=True)
```

## 📚 Best Practices

### 1. Choose Right Cleaning Level

* **Light** : For transformer models (BERT, etc.)
* **Medium** : For traditional ML models (SVM, Logistic Regression)
* **Heavy** : For topic modeling and clustering

### 2. Preserve Sentiment Information

* Keep emojis converted to text rather than removing
* Don't over-clean if sentiment is important
* Test preprocessing impact on labeled data

### 3. Handle Domain-Specific Terms

* Add banking-specific terms to stopwords if needed
* Preserve important financial terminology
* Consider custom stemming rules

### 4. Monitor Processing Quality

* Check processing statistics regularly
* Validate output samples manually
* Use analysis notebook for insights

## 🎯 Next Steps

After preprocessing, continue with:

1. **Feature Extraction** : Use TF-IDF, Word2Vec, or BERT embeddings
2. **Model Training** : Apply classification algorithms
3. **Evaluation** : Compare different preprocessing versions
4. **Error Analysis** : Analyze misclassified examples

## 📞 Support

For issues or questions:

1. Check logs in `logs/preprocessing.log`
2. Review configuration in `config.py`
3. Run setup script: `python scripts/setup_preprocessing.py`
4. Use analysis notebook for debugging

The preprocessing system is designed to be robust and flexible for various Persian NLP tasks. Adjust configurations based on your specific requirements and model performance.
