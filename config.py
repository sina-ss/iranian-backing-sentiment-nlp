"""
Configuration file for Persian Banking Sentiment Analysis Project
"""

import os
from pathlib import Path

# Project Structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data Directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model Directories
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Results Directories
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR,
                  PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, SAVED_MODELS_DIR,
                  CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Collection Configuration
CAFE_BAZAAR_CONFIG = {
    "base_url": "https://cafebazaar.ir",
    "banking_apps": [
        "com.tejarat.ezam",           # Tejarat Bank
        "com.mellat.hamrah",          # Mellat Bank
        "com.parsian.pec.mobile",     # Parsian Bank
        "com.eghtesadnovin.enhbank",  # Eghtesad Novin
        "com.bmi.mobilebanking",      # Melli Bank
        "com.saderat.saderatemobile",  # Saderat Bank
        "com.dey.mobile",             # Dey Bank
        "com.resalat.mobile",         # Resalat Bank
        "com.karafarin.mobile",       # Karafarin Bank
        "com.saman.mobile"            # Saman Bank
    ],
    "max_comments_per_app": 200,
    "delay_between_requests": 2,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Text Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "remove_english": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_phone_numbers": True,
    "remove_emojis": False,  # Keep emojis for sentiment analysis
    "normalize_persian": True,
    "remove_diacritics": True,
    "min_word_length": 2,
    "max_word_length": 50,
    "min_comment_length": 10,
    "max_comment_length": 500
}

# Persian Language Configuration
PERSIAN_CONFIG = {
    "stopwords_file": EXTERNAL_DATA_DIR / "persian_stopwords.txt",
    "custom_stopwords": [
        "بانک", "اپ", "اپلیکیشن", "موبایل", "سیستم", "وب", "سایت",
        "برنامه", "نرم افزار", "دانلود", "نصب", "بروزرسانی"
    ],
    "banking_terms": {
        "positive": ["عالی", "فوق العاده", "راحت", "سریع", "آسان", "مناسب", "خوب", "بهترین"],
        "negative": ["بد", "ضعیف", "کند", "مشکل", "خرابی", "اشکال", "قطعی", "پیچیده"],
        "neutral": ["متوسط", "عادی", "قابل قبول", "نظر", "نگاه", "مشاهده"]
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "tfidf": {
        "max_features": 10000,
        "min_df": 2,
        "max_df": 0.8,
        "ngram_range": (1, 2),
        "use_idf": True,
        "smooth_idf": True
    },
    "word2vec": {
        "vector_size": 200,
        "window": 5,
        "min_count": 2,
        "workers": 4,
        "epochs": 100,
        "sg": 1  # Skip-gram
    },
    "bert": {
        "model_name": "HooshvareLab/bert-base-parsbert-uncased",
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "return_attention_mask": True
    }
}

# Model Configuration
MODEL_CONFIG = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "solver": "liblinear"
    },
    "neural_network": {
        "lstm": {
            "embedding_dim": 200,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True
        },
        "cnn": {
            "embedding_dim": 200,
            "num_filters": 100,
            "filter_sizes": [3, 4, 5],
            "dropout": 0.3
        }
    },
    "bert_finetuning": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "cross_validation_folds": 5,
    "early_stopping_patience": 3,
    "checkpoint_every": 100
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "average": "weighted",
    "labels": ["negative", "neutral", "positive"],
    "confusion_matrix": True,
    "classification_report": True,
    "roc_curve": True,
    "pr_curve": True
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "font_family": "Arial Unicode MS",  # Supports Persian text
    "font_size": 12,
    "save_format": "png"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "persian_sentiment_analysis.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Sentiment Labels
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Banking Service Categories
BANKING_CATEGORIES = {
    "mobile_app": ["اپ", "اپلیکیشن", "موبایل", "گوشی", "تلفن همراه"],
    "atm": ["خودپرداز", "ATM", "عابر بانک", "کارت"],
    "online_banking": ["آنلاین", "اینترنتی", "وب", "سایت", "پورتال"],
    "customer_service": ["خدمات", "پشتیبانی", "مشتری", "کارمند", "شعبه"],
    "transaction": ["تراکنش", "انتقال", "واریز", "پرداخت", "حواله"],
    "security": ["امنیت", "رمز", "پسورد", "احراز", "هویت"]
}

# File Paths
FILE_PATHS = {
    "raw_comments": RAW_DATA_DIR / "cafe_bazaar_comments.csv",
    "processed_comments": PROCESSED_DATA_DIR / "cleaned_comments.csv",
    "train_data": PROCESSED_DATA_DIR / "train_data.csv",
    "test_data": PROCESSED_DATA_DIR / "test_data.csv",
    "validation_data": PROCESSED_DATA_DIR / "validation_data.csv",
    "persian_stopwords": EXTERNAL_DATA_DIR / "persian_stopwords.txt",
    "model_scores": METRICS_DIR / "model_scores.json",
    "final_report": REPORTS_DIR / "final_analysis_report.html"
}

# API Keys and External Services (if needed)
API_CONFIG = {
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN", ""),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "google_translate_key": os.getenv("GOOGLE_TRANSLATE_KEY", "")
}

# Persian Text Processing Patterns
PERSIAN_PATTERNS = {
    "persian_chars": r'[آ-ی]',
    "persian_numbers": r'[۰-۹]',
    "english_chars": r'[a-zA-Z]',
    "english_numbers": r'[0-9]',
    "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    "emails": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone_numbers": r'(\+98|0)?9\d{9}',
    "punctuation": r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?]',
    "extra_whitespace": r'\s+',
    "persian_punctuation": r'[؟،؛]'
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "minimum_accuracy": 0.7,
    "minimum_f1": 0.65,
    "minimum_precision": 0.6,
    "minimum_recall": 0.6,
    "excellent_accuracy": 0.85,
    "excellent_f1": 0.8
}

# Export configurations for easy access
__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR',
    'CAFE_BAZAAR_CONFIG', 'PREPROCESSING_CONFIG', 'PERSIAN_CONFIG',
    'FEATURE_CONFIG', 'MODEL_CONFIG', 'TRAINING_CONFIG',
    'EVALUATION_CONFIG', 'VISUALIZATION_CONFIG', 'LOGGING_CONFIG',
    'SENTIMENT_LABELS', 'BANKING_CATEGORIES', 'FILE_PATHS',
    'API_CONFIG', 'PERSIAN_PATTERNS', 'PERFORMANCE_THRESHOLDS'
]
