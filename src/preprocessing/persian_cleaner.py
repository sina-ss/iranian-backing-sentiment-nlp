"""
Persian Text Preprocessing and Cleaning Module
Comprehensive text cleaning for Persian banking comments
"""

import re
import json
import string
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import logging

# Persian NLP libraries
try:
    from hazm import *
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    print("Warning: hazm library not available. Some features will be limited.")

# Import project configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        PREPROCESSING_CONFIG,
        PERSIAN_CONFIG,
        EXTERNAL_DATA_DIR,
        PERSIAN_PATTERNS,
        FILE_PATHS
    )
except ImportError as e:
    print(
        f"Warning: config module not found. Some features will be limited. {e}")


class PersianTextCleaner:
    """
    Comprehensive Persian text cleaning and preprocessing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Persian text cleaner

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or PREPROCESSING_CONFIG
        self.persian_config = PERSIAN_CONFIG
        self.patterns = PERSIAN_PATTERNS

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._load_external_resources()
        self._setup_hazm_components()
        self._compile_patterns()

        # Statistics tracking
        self.stats = {
            'processed_texts': 0,
            'removed_urls': 0,
            'removed_emails': 0,
            'removed_phone_numbers': 0,
            'converted_emojis': 0,
            'removed_stopwords': 0,
            'normalized_texts': 0,
            'stemmed_words': 0
        }

    def _load_external_resources(self):
        """Load external resources (stopwords, emoji mapping)"""
        try:
            # Load Persian stopwords
            stopwords_file = EXTERNAL_DATA_DIR / "persian_stopwords.txt"
            if stopwords_file.exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = [line.strip()
                                      for line in f if line.strip()]
            else:
                file_stopwords = []

            # Combine with custom stopwords from config
            self.stopwords = set(
                file_stopwords + self.persian_config.get("custom_stopwords", []))

            # Load emoji mapping
            emoji_file = EXTERNAL_DATA_DIR / "persian_emoji_mapping.json"
            if emoji_file.exists():
                with open(emoji_file, 'r', encoding='utf-8') as f:
                    self.emoji_mapping = json.load(f)
            else:
                self.emoji_mapping = {}

            # Create flat emoji mapping for easier lookup
            self.flat_emoji_mapping = {}
            for category in self.emoji_mapping.values():
                if isinstance(category, dict):
                    self.flat_emoji_mapping.update(category)

            self.logger.info(
                f"Loaded {len(self.stopwords)} stopwords and {len(self.flat_emoji_mapping)} emoji mappings")

        except Exception as e:
            self.logger.error(f"Error loading external resources: {e}")
            self.stopwords = set()
            self.emoji_mapping = {}
            self.flat_emoji_mapping = {}

    def _setup_hazm_components(self):
        """Setup Hazm NLP components"""
        if not HAZM_AVAILABLE:
            self.logger.warning(
                "Hazm not available. Using basic text processing.")
            self.normalizer = None
            self.stemmer = None
            self.lemmatizer = None
            self.word_tokenizer = None
            self.sent_tokenizer = None
            return

        try:
            # Initialize Hazm components
            self.normalizer = Normalizer(
                persian_style=True,
                persian_numbers=True,
                remove_diacritics=self.config.get('remove_diacritics', True),
                remove_specials_chars=False,  # We'll handle this manually
                decrease_repeated_chars=True
            )

            self.stemmer = Stemmer()
            self.lemmatizer = Lemmatizer()
            self.word_tokenizer = WordTokenizer()
            self.sent_tokenizer = SentenceTokenizer()

            self.logger.info("Hazm components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Hazm components: {e}")
            self.normalizer = None
            self.stemmer = None
            self.lemmatizer = None
            self.word_tokenizer = None
            self.sent_tokenizer = None

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.compiled_patterns = {
            'urls': re.compile(self.patterns['urls'], re.IGNORECASE),
            'emails': re.compile(self.patterns['emails'], re.IGNORECASE),
            'phone_numbers': re.compile(self.patterns['phone_numbers']),
            'english_chars': re.compile(self.patterns['english_chars']),
            'english_numbers': re.compile(self.patterns['english_numbers']),
            'extra_whitespace': re.compile(self.patterns['extra_whitespace']),
            'punctuation': re.compile(self.patterns['punctuation']),
            'persian_punctuation': re.compile(self.patterns['persian_punctuation'])
        }

        # User mention pattern
        self.compiled_patterns['mentions'] = re.compile(r'@\w+')

        # Hashtag pattern
        self.compiled_patterns['hashtags'] = re.compile(r'#\w+')

        # Multiple dots/punctuation
        self.compiled_patterns['repeated_punct'] = re.compile(r'[.!?]{2,}')

        # Digits pattern for number normalization
        self.compiled_patterns['persian_digits'] = re.compile(r'[۰-۹]+')
        self.compiled_patterns['english_digits'] = re.compile(r'[0-9]+')

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        cleaned = self.compiled_patterns['urls'].sub(' ', text)
        if len(cleaned) != len(text):
            self.stats['removed_urls'] += 1
        return cleaned

    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        cleaned = self.compiled_patterns['emails'].sub(' ', text)
        if len(cleaned) != len(text):
            self.stats['removed_emails'] += 1
        return cleaned

    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        cleaned = self.compiled_patterns['phone_numbers'].sub(' ', text)
        if len(cleaned) != len(text):
            self.stats['removed_phone_numbers'] += 1
        return cleaned

    def remove_mentions_hashtags(self, text: str) -> str:
        """Remove mentions and hashtags"""
        text = self.compiled_patterns['mentions'].sub(' ', text)
        text = self.compiled_patterns['hashtags'].sub(' ', text)
        return text

    def convert_emojis(self, text: str) -> str:
        """Convert emojis to Persian text equivalents"""
        if not self.config.get('remove_emojis', False) and self.flat_emoji_mapping:
            converted_count = 0
            for emoji, persian_text in self.flat_emoji_mapping.items():
                if emoji in text:
                    text = text.replace(emoji, f" {persian_text} ")
                    converted_count += 1

            if converted_count > 0:
                self.stats['converted_emojis'] += converted_count

        return text

    def remove_emojis(self, text: str) -> str:
        """Remove emojis completely"""
        # Unicode emoji pattern
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(' ', text)

    def normalize_persian_text(self, text: str) -> str:
        """Normalize Persian text using Hazm"""
        if self.normalizer:
            try:
                normalized = self.normalizer.normalize(text)
                self.stats['normalized_texts'] += 1
                return normalized
            except Exception as e:
                self.logger.warning(f"Hazm normalization failed: {e}")

        # Fallback manual normalization
        return self._manual_persian_normalization(text)

    def _manual_persian_normalization(self, text: str) -> str:
        """Manual Persian text normalization"""
        # Persian character mappings
        persian_char_map = {
            'ك': 'ک',  # Arabic kaf to Persian kaf
            'ي': 'ی',  # Arabic yeh to Persian yeh
            'ئ': 'ی',  # Hamza on yeh to yeh
            'ء': '',   # Remove hamza
            'أ': 'ا',  # Alef with hamza above to alef
            'إ': 'ا',  # Alef with hamza below to alef
            'آ': 'ا',  # Alef with madda to alef
            'ة': 'ه',  # Teh marbuta to heh
        }

        for arabic_char, persian_char in persian_char_map.items():
            text = text.replace(arabic_char, persian_char)

        # Convert Persian numbers to English if configured
        if self.config.get('normalize_numbers', True):
            persian_to_english = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
            text = text.translate(persian_to_english)

        # Remove diacritics if configured
        if self.config.get('remove_diacritics', True):
            diacritics = 'ًٌٍَُِّْ'
            for diacritic in diacritics:
                text = text.replace(diacritic, '')

        return text

    def remove_english(self, text: str) -> str:
        """Remove English characters if configured"""
        if self.config.get('remove_english', True):
            text = self.compiled_patterns['english_chars'].sub(' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Persian text"""
        if self.word_tokenizer:
            try:
                return self.word_tokenizer.tokenize(text)
            except Exception as e:
                self.logger.warning(f"Hazm tokenization failed: {e}")

        # Fallback tokenization
        return self._simple_tokenize(text)

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based tokenization"""
        # Replace punctuation with spaces
        text = self.compiled_patterns['punctuation'].sub(' ', text)
        text = self.compiled_patterns['persian_punctuation'].sub(' ', text)

        # Split on whitespace and filter
        tokens = text.split()

        # Filter by length
        min_len = self.config.get('min_word_length', 2)
        max_len = self.config.get('max_word_length', 50)

        return [token for token in tokens if min_len <= len(token) <= max_len]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Persian stopwords"""
        filtered_tokens = [
            token for token in tokens if token.lower() not in self.stopwords]
        removed_count = len(tokens) - len(filtered_tokens)
        if removed_count > 0:
            self.stats['removed_stopwords'] += removed_count
        return filtered_tokens

    def stem_words(self, tokens: List[str]) -> List[str]:
        """Stem Persian words"""
        if not self.stemmer:
            return tokens

        stemmed = []
        for token in tokens:
            try:
                stemmed_token = self.stemmer.stem(token)
                stemmed.append(stemmed_token)
                if stemmed_token != token:
                    self.stats['stemmed_words'] += 1
            except Exception as e:
                self.logger.warning(f"Stemming failed for '{token}': {e}")
                stemmed.append(token)

        return stemmed

    def lemmatize_words(self, tokens: List[str]) -> List[str]:
        """Lemmatize Persian words"""
        if not self.lemmatizer:
            return tokens

        lemmatized = []
        for token in tokens:
            try:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                lemmatized.append(lemmatized_token)
            except Exception as e:
                self.logger.warning(f"Lemmatization failed for '{token}': {e}")
                lemmatized.append(token)

        return lemmatized

    def clean_punctuation(self, text: str) -> str:
        """Clean and normalize punctuation"""
        # Replace repeated punctuation
        text = self.compiled_patterns['repeated_punct'].sub('.', text)

        # Remove extra punctuation but keep Persian punctuation
        if self.config.get('remove_punctuation', False):
            text = self.compiled_patterns['punctuation'].sub(' ', text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        return self.compiled_patterns['extra_whitespace'].sub(' ', text).strip()

    def validate_text_length(self, text: str) -> bool:
        """Validate if text meets length requirements"""
        min_len = self.config.get('min_comment_length', 10)
        max_len = self.config.get('max_comment_length', 500)

        return min_len <= len(text) <= max_len

    def clean_text(self, text: str, level: str = 'medium') -> str:
        """
        Main cleaning function with different intensity levels

        Args:
            text: Input text to clean
            level: Cleaning level ('light', 'medium', 'heavy')

        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        original_text = text

        # Always normalize whitespace first
        text = self.normalize_whitespace(text)

        if level in ['medium', 'heavy']:
            # Remove noise
            if self.config.get('remove_urls', True):
                text = self.remove_urls(text)

            if self.config.get('remove_emails', True):
                text = self.remove_emails(text)

            if self.config.get('remove_phone_numbers', True):
                text = self.remove_phone_numbers(text)

            # Handle mentions and hashtags
            text = self.remove_mentions_hashtags(text)

        if level == 'heavy':
            # Remove English characters
            text = self.remove_english(text)

        # Handle emojis
        if self.config.get('remove_emojis', False):
            text = self.remove_emojis(text)
        else:
            text = self.convert_emojis(text)

        # Persian normalization
        if self.config.get('normalize_persian', True):
            text = self.normalize_persian_text(text)

        # Clean punctuation
        text = self.clean_punctuation(text)

        # Final whitespace normalization
        text = self.normalize_whitespace(text)

        # Update statistics
        if text != original_text:
            self.stats['processed_texts'] += 1

        return text

    def preprocess_text(self, text: str,
                        tokenize: bool = True,
                        remove_stopwords: bool = True,
                        stem: bool = False,
                        lemmatize: bool = False) -> Union[str, List[str]]:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            tokenize: Whether to tokenize
            remove_stopwords: Whether to remove stopwords
            stem: Whether to stem words
            lemmatize: Whether to lemmatize words

        Returns:
            Processed text or list of tokens
        """
        # Clean text first
        cleaned_text = self.clean_text(text)

        if not tokenize:
            return cleaned_text

        # Tokenize
        tokens = self.tokenize(cleaned_text)

        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Stemming or lemmatization (mutually exclusive)
        if lemmatize:
            tokens = self.lemmatize_words(tokens)
        elif stem:
            tokens = self.stem_words(tokens)

        return tokens

    def process_dataframe(self, df: pd.DataFrame,
                          text_column: str = 'comment',
                          **kwargs) -> pd.DataFrame:
        """
        Process a pandas DataFrame

        Args:
            df: Input DataFrame
            text_column: Name of the text column to process
            **kwargs: Additional arguments for preprocess_text

        Returns:
            DataFrame with processed text
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Create a copy to avoid modifying original
        result_df = df.copy()

        # Process each text
        processed_texts = []
        for text in df[text_column]:
            processed = self.preprocess_text(text, **kwargs)

            # If tokens returned, join them back to string
            if isinstance(processed, list):
                processed = ' '.join(processed)

            processed_texts.append(processed)

        result_df[f'{text_column}_cleaned'] = processed_texts

        # Filter out texts that don't meet length requirements
        if self.config.get('filter_by_length', True):
            valid_mask = result_df[f'{text_column}_cleaned'].apply(
                self.validate_text_length)
            result_df = result_df[valid_mask].reset_index(drop=True)

        self.logger.info(
            f"Processed {len(df)} texts, {len(result_df)} remain after filtering")

        return result_df

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            'processing_stats': self.stats.copy(),
            'configuration': {
                'remove_urls': self.config.get('remove_urls', True),
                'remove_emails': self.config.get('remove_emails', True),
                'remove_emojis': self.config.get('remove_emojis', False),
                'normalize_persian': self.config.get('normalize_persian', True),
                'remove_diacritics': self.config.get('remove_diacritics', True),
                'hazm_available': HAZM_AVAILABLE
            },
            'resources': {
                'stopwords_count': len(self.stopwords),
                'emoji_mappings_count': len(self.flat_emoji_mapping)
            }
        }

    def reset_statistics(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0


def create_sample_pipeline():
    """Create a sample preprocessing pipeline for testing"""
    cleaner = PersianTextCleaner()

    sample_texts = [
        "سلام! اپ بانک عالیه 😊 خیلی راحت و سریع کار میکنه 👍",
        "این برنامه خراب است! 😡 هی قطع میشه... @bankSupport #مشکل",
        "نرم افزار متوسط است. www.example.com را ببینید. تلفن: 09123456789",
        "🚀 فوق العاده! بهترین اپ بانکی که دیدم ⭐⭐⭐⭐⭐"
    ]

    print("=== Persian Text Preprocessing Demo ===\n")

    for i, text in enumerate(sample_texts, 1):
        print(f"Original {i}: {text}")

        # Different cleaning levels
        light = cleaner.clean_text(text, level='light')
        medium = cleaner.clean_text(text, level='medium')
        heavy = cleaner.clean_text(text, level='heavy')

        print(f"Light:    {light}")
        print(f"Medium:   {medium}")
        print(f"Heavy:    {heavy}")

        # Tokenization
        tokens = cleaner.preprocess_text(
            text, tokenize=True, remove_stopwords=True)
        print(f"Tokens:   {tokens}")

        print("-" * 50)

    # Show statistics
    stats = cleaner.get_statistics()
    print("\nProcessing Statistics:")
    for key, value in stats['processing_stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    create_sample_pipeline()
