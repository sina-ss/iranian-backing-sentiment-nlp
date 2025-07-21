"""
Error Analysis Tool for Persian Banking Sentiment Classification
Advanced error analysis with pattern detection and visualization
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        RESULTS_DIR,
        FIGURES_DIR,
        SENTIMENT_LABELS
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {Path(__file__).parent.parent.parent}")


class PersianSentimentErrorAnalyzer:
    """
    Comprehensive error analyzer for Persian sentiment classification models
    """

    def __init__(self):
        """Initialize error analyzer"""
        self.setup_logging()

        # Data storage
        self.y_true = None
        self.y_pred = None
        self.y_proba = None
        self.texts = None
        self.class_names = None
        self.model_name = None

        # Error analysis results
        self.error_indices = None
        self.error_analysis = {}
        self.error_patterns = {}
        self.confidence_analysis = {}

        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_errors': 0,
            'error_rate': 0.0,
            'analysis_time': 0.0,
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for error analysis"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "error_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                       texts: List[str], y_proba: Optional[np.ndarray] = None,
                       class_names: Optional[List[str]] = None,
                       model_name: str = "Model") -> Dict:
        """
        Perform comprehensive error analysis

        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: List of text samples
            y_proba: Prediction probabilities (optional)
            class_names: Class names (optional)
            model_name: Name of the model being analyzed

        Returns:
            Dictionary containing error analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting error analysis for {model_name}...")

        # Store data
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.texts = texts
        self.model_name = model_name

        # Handle class names
        if class_names is None:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            if all(isinstance(label, str) for label in unique_labels):
                self.class_names = sorted(unique_labels)
            else:
                self.class_names = [f"Class_{i}" for i in unique_labels]
        else:
            self.class_names = class_names

        # Find error indices
        self.error_indices = np.where(y_true != y_pred)[0]

        # Update statistics
        self.stats['total_samples'] = len(y_true)
        self.stats['total_errors'] = len(self.error_indices)
        self.stats['error_rate'] = len(self.error_indices) / len(y_true)

        # Perform different types of error analysis
        self.error_analysis = self._analyze_error_types()
        self.error_patterns = self._find_error_patterns()

        if y_proba is not None:
            self.confidence_analysis = self._analyze_confidence_errors()

        # Calculate processing time
        analysis_time = (datetime.now() - start_time).total_seconds()
        self.stats['analysis_time'] = analysis_time
        self.stats['timestamp'] = datetime.now().isoformat()

        # Compile results
        results = {
            'model_name': model_name,
            'error_statistics': self.stats,
            'error_analysis': self.error_analysis,
            'error_patterns': self.error_patterns,
            'confidence_analysis': self.confidence_analysis
        }

        self.logger.info(
            f"Error analysis completed in {analysis_time:.3f} seconds")
        self.logger.info(
            f"Total errors: {self.stats['total_errors']} ({self.stats['error_rate']:.2%})")

        return results

    def _analyze_error_types(self) -> Dict:
        """Analyze types and patterns of errors"""
        error_analysis = {
            'confusion_pairs': {},
            'per_class_errors': {},
            'error_examples': {},
            'error_distribution': {}
        }

        # Analyze confusion pairs (True class -> Predicted class)
        confusion_pairs = defaultdict(int)
        for idx in self.error_indices:
            true_class = self.y_true[idx]
            pred_class = self.y_pred[idx]

            # Convert to class names if numeric
            if isinstance(true_class, (int, np.integer)):
                true_name = self.class_names[true_class]
                pred_name = self.class_names[pred_class]
            else:
                true_name = true_class
                pred_name = pred_class

            confusion_pairs[(true_name, pred_name)] += 1

        error_analysis['confusion_pairs'] = dict(confusion_pairs)

        # Analyze per-class error rates
        for i, class_name in enumerate(self.class_names):
            # Find samples of this class
            if isinstance(self.y_true[0], str):
                class_mask = (self.y_true == class_name)
            else:
                class_mask = (self.y_true == i)

            class_total = np.sum(class_mask)

            if class_total > 0:
                # Find errors for this class
                class_error_mask = class_mask & (self.y_true != self.y_pred)
                class_errors = np.sum(class_error_mask)

                error_analysis['per_class_errors'][class_name] = {
                    'total_samples': int(class_total),
                    'errors': int(class_errors),
                    'error_rate': float(class_errors / class_total),
                    'correct_predictions': int(class_total - class_errors)
                }
            else:
                error_analysis['per_class_errors'][class_name] = {
                    'total_samples': 0,
                    'errors': 0,
                    'error_rate': 0.0,
                    'correct_predictions': 0
                }

        # Collect error examples for each confusion pair
        for (true_class, pred_class), count in confusion_pairs.items():
            # Find examples of this specific error type
            examples = []
            example_count = 0

            for idx in self.error_indices:
                if example_count >= 5:  # Limit examples
                    break

                true_label = self.y_true[idx]
                pred_label = self.y_pred[idx]

                # Convert to class names for comparison
                if isinstance(true_label, (int, np.integer)):
                    true_label_name = self.class_names[true_label]
                    pred_label_name = self.class_names[pred_label]
                else:
                    true_label_name = true_label
                    pred_label_name = pred_label

                if true_label_name == true_class and pred_label_name == pred_class:
                    example_info = {
                        'text': self.texts[idx],
                        'true_label': true_class,
                        'predicted_label': pred_class,
                        'index': int(idx)
                    }

                    # Add confidence if available
                    if self.y_proba is not None:
                        proba = self.y_proba[idx]
                        max_prob = np.max(proba)
                        predicted_class_idx = np.argmax(proba)

                        example_info['confidence'] = float(max_prob)
                        example_info['all_probabilities'] = {
                            self.class_names[i]: float(proba[i])
                            for i in range(len(self.class_names))
                        }

                    examples.append(example_info)
                    example_count += 1

            error_analysis['error_examples'][f"{true_class}→{pred_class}"] = examples

        # Error distribution by text characteristics
        error_analysis['error_distribution'] = self._analyze_error_distribution()

        return error_analysis

    def _analyze_error_distribution(self) -> Dict:
        """Analyze error distribution by text characteristics"""
        distribution = {
            'by_text_length': {},
            'by_word_count': {},
            'by_sentiment_words': {}
        }

        # Define length bins
        all_lengths = [len(text) for text in self.texts]
        length_bins = [0, 50, 100, 200, 400, float('inf')]
        length_labels = ['Very Short (0-50)', 'Short (51-100)', 'Medium (101-200)',
                         'Long (201-400)', 'Very Long (>400)']

        # Analyze errors by text length
        for i, (low, high) in enumerate(zip(length_bins[:-1], length_bins[1:])):
            if high == float('inf'):
                length_mask = np.array(all_lengths) > low
            else:
                length_mask = (np.array(all_lengths) > low) & (
                    np.array(all_lengths) <= high)

            total_in_bin = np.sum(length_mask)
            if total_in_bin > 0:
                errors_in_bin = np.sum(
                    [idx in self.error_indices for idx in np.where(length_mask)[0]])
                distribution['by_text_length'][length_labels[i]] = {
                    'total_samples': int(total_in_bin),
                    'errors': int(errors_in_bin),
                    'error_rate': float(errors_in_bin / total_in_bin)
                }

        # Analyze errors by word count
        word_counts = [len(text.split()) for text in self.texts]
        word_bins = [0, 5, 10, 20, 50, float('inf')]
        word_labels = ['Very Few (1-5)', 'Few (6-10)', 'Medium (11-20)',
                       'Many (21-50)', 'Very Many (>50)']

        for i, (low, high) in enumerate(zip(word_bins[:-1], word_bins[1:])):
            if high == float('inf'):
                word_mask = np.array(word_counts) > low
            else:
                word_mask = (np.array(word_counts) > low) & (
                    np.array(word_counts) <= high)

            total_in_bin = np.sum(word_mask)
            if total_in_bin > 0:
                errors_in_bin = np.sum(
                    [idx in self.error_indices for idx in np.where(word_mask)[0]])
                distribution['by_word_count'][word_labels[i]] = {
                    'total_samples': int(total_in_bin),
                    'errors': int(errors_in_bin),
                    'error_rate': float(errors_in_bin / total_in_bin)
                }

        # Analyze errors by presence of sentiment words
        positive_words = {'خوب', 'عالی',
                          'فوق‌العاده', 'بهترین', 'راحت', 'سریع'}
        negative_words = {'بد', 'افتضاح', 'ضعیف', 'کند', 'مشکل', 'خراب'}

        for category, words in [('has_positive_words', positive_words),
                                ('has_negative_words', negative_words)]:
            has_words_mask = np.array([
                any(word in text for word in words) for text in self.texts
            ])

            total_with_words = np.sum(has_words_mask)
            total_without_words = len(self.texts) - total_with_words

            if total_with_words > 0:
                errors_with_words = np.sum(
                    [idx in self.error_indices for idx in np.where(has_words_mask)[0]])
                errors_without_words = len(
                    self.error_indices) - errors_with_words

                distribution['by_sentiment_words'][category] = {
                    'with_words': {
                        'total_samples': int(total_with_words),
                        'errors': int(errors_with_words),
                        'error_rate': float(errors_with_words / total_with_words) if total_with_words > 0 else 0.0
                    },
                    'without_words': {
                        'total_samples': int(total_without_words),
                        'errors': int(errors_without_words),
                        'error_rate': float(errors_without_words / total_without_words) if total_without_words > 0 else 0.0
                    }
                }

        return distribution

    def _find_error_patterns(self) -> Dict:
        """Find patterns in misclassified texts"""
        patterns = {
            'common_words_in_errors': {},
            'length_analysis': {},
            'lexical_patterns': {},
            'error_clusters': {}
        }

        # Analyze common words in errors for each confusion pair
        error_texts_by_pair = defaultdict(list)

        for idx in self.error_indices:
            true_label = self.y_true[idx]
            pred_label = self.y_pred[idx]

            # Convert to class names
            if isinstance(true_label, (int, np.integer)):
                true_name = self.class_names[true_label]
                pred_name = self.class_names[pred_label]
            else:
                true_name = true_label
                pred_name = pred_label

            pair_key = f"{true_name}→{pred_name}"
            error_texts_by_pair[pair_key].append(self.texts[idx])

        # Find common words in each error type
        for pair, texts in error_texts_by_pair.items():
            if len(texts) >= 3:  # Only analyze if we have enough examples
                # Extract words from all error texts of this type
                all_words = []
                for text in texts:
                    words = re.findall(r'\b\w+\b', text.lower())
                    all_words.extend(words)

                # Count word frequencies
                word_counts = Counter(all_words)

                # Get most common words (excluding very common ones)
                common_stopwords = {'و', 'در', 'به', 'از',
                                    'که', 'این', 'آن', 'را', 'با', 'تا', 'برای'}
                filtered_words = {word: count for word, count in word_counts.items()
                                  if word not in common_stopwords and len(word) > 2}

                top_words = dict(sorted(filtered_words.items(),
                                 key=lambda x: x[1], reverse=True)[:10])
                patterns['common_words_in_errors'][pair] = top_words

        # Analyze length patterns in errors
        error_lengths = [len(self.texts[idx]) for idx in self.error_indices]
        correct_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.error_indices)
        correct_lengths = [len(self.texts[idx]) for idx in correct_indices]

        patterns['length_analysis'] = {
            'error_length_stats': {
                'mean': float(np.mean(error_lengths)) if error_lengths else 0.0,
                'median': float(np.median(error_lengths)) if error_lengths else 0.0,
                'std': float(np.std(error_lengths)) if error_lengths else 0.0
            },
            'correct_length_stats': {
                'mean': float(np.mean(correct_lengths)) if correct_lengths else 0.0,
                'median': float(np.median(correct_lengths)) if correct_lengths else 0.0,
                'std': float(np.std(correct_lengths)) if correct_lengths else 0.0
            }
        }

        # Analyze lexical patterns
        patterns['lexical_patterns'] = self._analyze_lexical_patterns()

        return patterns

    def _analyze_lexical_patterns(self) -> Dict:
        """Analyze lexical patterns in errors"""
        lexical_patterns = {
            'emoji_usage': {},
            'punctuation_patterns': {},
            'capitalization_patterns': {},
            'special_characters': {}
        }

        # Analyze emoji usage in errors vs correct predictions
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')

        error_texts = [self.texts[idx] for idx in self.error_indices]
        correct_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.error_indices)
        correct_texts = [self.texts[idx] for idx in correct_indices]

        error_with_emoji = sum(
            1 for text in error_texts if emoji_pattern.search(text))
        correct_with_emoji = sum(
            1 for text in correct_texts if emoji_pattern.search(text))

        lexical_patterns['emoji_usage'] = {
            'errors_with_emoji': error_with_emoji,
            'errors_total': len(error_texts),
            'errors_emoji_rate': error_with_emoji / len(error_texts) if error_texts else 0.0,
            'correct_with_emoji': correct_with_emoji,
            'correct_total': len(correct_texts),
            'correct_emoji_rate': correct_with_emoji / len(correct_texts) if correct_texts else 0.0
        }

        # Analyze punctuation patterns
        punctuation_chars = '!@#$%^&*().,;:?'

        error_punct_density = [
            sum(1 for char in text if char in punctuation_chars) /
            len(text) if text else 0
            for text in error_texts
        ]
        correct_punct_density = [
            sum(1 for char in text if char in punctuation_chars) /
            len(text) if text else 0
            for text in correct_texts
        ]

        lexical_patterns['punctuation_patterns'] = {
            'error_punctuation_density': {
                'mean': float(np.mean(error_punct_density)) if error_punct_density else 0.0,
                'std': float(np.std(error_punct_density)) if error_punct_density else 0.0
            },
            'correct_punctuation_density': {
                'mean': float(np.mean(correct_punct_density)) if correct_punct_density else 0.0,
                'std': float(np.std(correct_punct_density)) if correct_punct_density else 0.0
            }
        }

        return lexical_patterns

    def _analyze_confidence_errors(self) -> Dict:
        """Analyze errors by model confidence"""
        if self.y_proba is None:
            return {}

        confidence_analysis = {
            'high_confidence_errors': [],
            'low_confidence_correct': [],
            'confidence_distribution': {},
            'calibration_analysis': {}
        }

        # Calculate confidence (max probability) for each prediction
        confidences = np.max(self.y_proba, axis=1)

        # Define confidence thresholds
        high_conf_threshold = 0.8
        low_conf_threshold = 0.6

        # High confidence errors (model was confident but wrong)
        high_conf_error_indices = []
        for idx in self.error_indices:
            if confidences[idx] >= high_conf_threshold:
                high_conf_error_indices.append(idx)

        # Collect examples of high confidence errors
        for idx in high_conf_error_indices[:10]:  # Limit to 10 examples
            example = {
                'text': self.texts[idx],
                'true_label': self.class_names[self.y_true[idx]] if isinstance(self.y_true[idx], (int, np.integer)) else self.y_true[idx],
                'predicted_label': self.class_names[self.y_pred[idx]] if isinstance(self.y_pred[idx], (int, np.integer)) else self.y_pred[idx],
                'confidence': float(confidences[idx]),
                'all_probabilities': {
                    self.class_names[i]: float(self.y_proba[idx][i])
                    for i in range(len(self.class_names))
                }
            }
            confidence_analysis['high_confidence_errors'].append(example)

        # Low confidence correct predictions (model was uncertain but right)
        correct_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.error_indices)
        low_conf_correct_indices = [
            idx for idx in correct_indices if confidences[idx] <= low_conf_threshold]

        for idx in low_conf_correct_indices[:10]:  # Limit to 10 examples
            example = {
                'text': self.texts[idx],
                'true_label': self.class_names[self.y_true[idx]] if isinstance(self.y_true[idx], (int, np.integer)) else self.y_true[idx],
                'predicted_label': self.class_names[self.y_pred[idx]] if isinstance(self.y_pred[idx], (int, np.integer)) else self.y_pred[idx],
                'confidence': float(confidences[idx]),
                'all_probabilities': {
                    self.class_names[i]: float(self.y_proba[idx][i])
                    for i in range(len(self.class_names))
                }
            }
            confidence_analysis['low_confidence_correct'].append(example)

        # Confidence distribution analysis
        conf_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ['Very Low (0-0.5)', 'Low (0.5-0.7)', 'Medium (0.7-0.8)',
                       'High (0.8-0.9)', 'Very High (0.9-1.0)']

        for i, (low, high) in enumerate(zip(conf_bins[:-1], conf_bins[1:])):
            conf_mask = (confidences > low) & (confidences <= high)
            total_in_bin = np.sum(conf_mask)

            if total_in_bin > 0:
                errors_in_bin = np.sum(
                    [idx in self.error_indices for idx in np.where(conf_mask)[0]])
                confidence_analysis['confidence_distribution'][conf_labels[i]] = {
                    'total_samples': int(total_in_bin),
                    'errors': int(errors_in_bin),
                    'accuracy': float((total_in_bin - errors_in_bin) / total_in_bin)
                }

        # Calibration analysis
        confidence_analysis['calibration_analysis'] = self._analyze_calibration()

        return confidence_analysis

    def _analyze_calibration(self) -> Dict:
        """Analyze model calibration"""
        if self.y_proba is None:
            return {}

        # Binned calibration analysis
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = np.max(self.y_proba, axis=1)
        predictions = np.argmax(self.y_proba, axis=1)
        accuracies = (predictions == self.y_true)

        calibration_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                calibration_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'avg_confidence': float(avg_confidence_in_bin),
                    'accuracy': float(accuracy_in_bin),
                    'count': int(in_bin.sum()),
                    'calibration_error': float(abs(avg_confidence_in_bin - accuracy_in_bin))
                })

        # Calculate overall calibration metrics
        if calibration_data:
            # Expected Calibration Error (ECE)
            ece = sum(
                (item['count'] / len(confidences)) * item['calibration_error']
                for item in calibration_data
            )

            # Maximum Calibration Error (MCE)
            mce = max(item['calibration_error'] for item in calibration_data)

            return {
                'bin_data': calibration_data,
                'expected_calibration_error': float(ece),
                'maximum_calibration_error': float(mce),
                'total_samples': len(confidences)
            }

        return {}

    def get_worst_predictions(self, n_examples: int = 20) -> List[Dict]:
        """
        Get the worst predictions (most confident errors)

        Args:
            n_examples: Number of examples to return

        Returns:
            List of worst prediction examples
        """
        if self.y_proba is None:
            self.logger.warning(
                "No probability data available for worst predictions analysis")
            return []

        # Calculate confidence for error predictions
        error_confidences = []
        error_info = []

        for idx in self.error_indices:
            confidence = np.max(self.y_proba[idx])

            info = {
                'index': int(idx),
                'text': self.texts[idx],
                'true_label': self.class_names[self.y_true[idx]] if isinstance(self.y_true[idx], (int, np.integer)) else self.y_true[idx],
                'predicted_label': self.class_names[self.y_pred[idx]] if isinstance(self.y_pred[idx], (int, np.integer)) else self.y_pred[idx],
                'confidence': float(confidence),
                'all_probabilities': {
                    self.class_names[i]: float(self.y_proba[idx][i])
                    for i in range(len(self.class_names))
                }
            }

            error_confidences.append(confidence)
            error_info.append(info)

        # Sort by confidence (highest first - these are the worst errors)
        sorted_indices = np.argsort(error_confidences)[::-1]
        worst_predictions = [error_info[i]
                             for i in sorted_indices[:n_examples]]

        return worst_predictions

    def get_confused_predictions(self, n_examples: int = 20) -> List[Dict]:
        """
        Get predictions where the model was most confused (lowest max probability)

        Args:
            n_examples: Number of examples to return

        Returns:
            List of most confused prediction examples
        """
        if self.y_proba is None:
            self.logger.warning(
                "No probability data available for confused predictions analysis")
            return []

        # Calculate confusion (1 - max_probability) for all predictions
        confidences = np.max(self.y_proba, axis=1)
        confusion_scores = 1 - confidences

        # Get most confused examples
        confused_indices = np.argsort(confusion_scores)[::-1][:n_examples]

        confused_predictions = []
        for idx in confused_indices:
            info = {
                'index': int(idx),
                'text': self.texts[idx],
                'true_label': self.class_names[self.y_true[idx]] if isinstance(self.y_true[idx], (int, np.integer)) else self.y_true[idx],
                'predicted_label': self.class_names[self.y_pred[idx]] if isinstance(self.y_pred[idx], (int, np.integer)) else self.y_pred[idx],
                'is_error': bool(idx in self.error_indices),
                'confusion_score': float(confusion_scores[idx]),
                'confidence': float(confidences[idx]),
                'all_probabilities': {
                    self.class_names[i]: float(self.y_proba[idx][i])
                    for i in range(len(self.class_names))
                }
            }
            confused_predictions.append(info)

        return confused_predictions

    def create_error_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive error analysis visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "error_analysis" / \
                f"{self.model_name.lower().replace(' ', '_')}"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Error distribution plots
        self._plot_error_distribution(save_dir)
        saved_plots.append(str(save_dir / 'error_distribution.png'))

        # 2. Confusion pair analysis
        self._plot_confusion_pairs(save_dir)
        saved_plots.append(str(save_dir / 'confusion_pairs.png'))

        # 3. Error patterns by text characteristics
        self._plot_error_characteristics(save_dir)
        saved_plots.append(str(save_dir / 'error_characteristics.png'))

        # 4. Confidence analysis (if available)
        if self.confidence_analysis:
            self._plot_confidence_analysis(save_dir)
            saved_plots.append(str(save_dir / 'confidence_analysis.png'))

        # 5. Word clouds for error types
        self._create_error_wordclouds(save_dir)
        saved_plots.extend([str(save_dir / 'error_wordclouds.png')])

        return saved_plots

    def _plot_error_distribution(self, save_dir: Path):
        """Plot error distribution by class"""
        per_class_errors = self.error_analysis['per_class_errors']

        classes = list(per_class_errors.keys())
        error_rates = [per_class_errors[cls]['error_rate'] for cls in classes]
        sample_counts = [per_class_errors[cls]['total_samples']
                         for cls in classes]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Error rates by class
        bars1 = axes[0].bar(classes, error_rates,
                            color='lightcoral', alpha=0.8)
        axes[0].set_title('Error Rate by Class')
        axes[0].set_ylabel('Error Rate')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for bar, rate in zip(bars1, error_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f'{rate:.3f}', ha='center', va='bottom')

        # Sample counts by class
        bars2 = axes[1].bar(classes, sample_counts, color='skyblue', alpha=0.8)
        axes[1].set_title('Sample Count by Class')
        axes[1].set_ylabel('Number of Samples')
        axes[1].grid(True, alpha=0.3)

        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sample_counts)*0.01,
                         f'{count}', ha='center', va='bottom')

        plt.suptitle(
            f'{self.model_name} - Error Distribution Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'error_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_pairs(self, save_dir: Path):
        """Plot confusion pairs analysis"""
        confusion_pairs = self.error_analysis['confusion_pairs']

        if not confusion_pairs:
            return

        # Prepare data
        pairs = list(confusion_pairs.keys())
        counts = list(confusion_pairs.values())

        # Sort by count
        sorted_pairs_counts = sorted(
            zip(pairs, counts), key=lambda x: x[1], reverse=True)
        pairs, counts = zip(*sorted_pairs_counts)

        # Create labels
        pair_labels = [
            f"{true_cls}→{pred_cls}" for true_cls, pred_cls in pairs]

        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(pair_labels)), counts,
                        color='lightcoral', alpha=0.8)

        plt.yticks(range(len(pair_labels)), pair_labels)
        plt.xlabel('Number of Errors')
        plt.title(f'{self.model_name} - Most Common Confusion Pairs')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2.,
                     str(count), ha='left', va='center')

        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_pairs.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_characteristics(self, save_dir: Path):
        """Plot error characteristics analysis"""
        distribution = self.error_analysis['error_distribution']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Error rate by text length
        length_data = distribution['by_text_length']
        if length_data:
            lengths = list(length_data.keys())
            error_rates = [length_data[length]['error_rate']
                           for length in lengths]

            axes[0, 0].bar(lengths, error_rates, color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Error Rate by Text Length')
            axes[0, 0].set_ylabel('Error Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

        # Error rate by word count
        word_data = distribution['by_word_count']
        if word_data:
            word_ranges = list(word_data.keys())
            error_rates = [word_data[range_]['error_rate']
                           for range_ in word_ranges]

            axes[0, 1].bar(word_ranges, error_rates,
                           color='lightgreen', alpha=0.8)
            axes[0, 1].set_title('Error Rate by Word Count')
            axes[0, 1].set_ylabel('Error Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # Error rate by sentiment words
        sentiment_data = distribution['by_sentiment_words']
        if sentiment_data:
            categories = []
            with_word_rates = []
            without_word_rates = []

            for category, data in sentiment_data.items():
                categories.append(category.replace(
                    'has_', '').replace('_', ' ').title())
                with_word_rates.append(data['with_words']['error_rate'])
                without_word_rates.append(data['without_words']['error_rate'])

            x = np.arange(len(categories))
            width = 0.35

            axes[1, 0].bar(x - width/2, with_word_rates, width,
                           label='With Words', alpha=0.8)
            axes[1, 0].bar(x + width/2, without_word_rates,
                           width, label='Without Words', alpha=0.8)
            axes[1, 0].set_title('Error Rate by Sentiment Words Presence')
            axes[1, 0].set_ylabel('Error Rate')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Length comparison: errors vs correct
        length_analysis = self.error_patterns['length_analysis']
        error_stats = length_analysis['error_length_stats']
        correct_stats = length_analysis['correct_length_stats']

        metrics = ['mean', 'median']
        error_values = [error_stats[metric] for metric in metrics]
        correct_values = [correct_stats[metric] for metric in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1, 1].bar(x - width/2, error_values, width,
                       label='Errors', alpha=0.8, color='red')
        axes[1, 1].bar(x + width/2, correct_values, width,
                       label='Correct', alpha=0.8, color='green')
        axes[1, 1].set_title('Text Length: Errors vs Correct Predictions')
        axes[1, 1].set_ylabel('Characters')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([m.title() for m in metrics])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f'{self.model_name} - Error Characteristics Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'error_characteristics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_analysis(self, save_dir: Path):
        """Plot confidence analysis"""
        if not self.confidence_analysis:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Confidence distribution vs accuracy
        conf_dist = self.confidence_analysis['confidence_distribution']
        if conf_dist:
            conf_ranges = list(conf_dist.keys())
            accuracies = [conf_dist[range_]['accuracy']
                          for range_ in conf_ranges]
            sample_counts = [conf_dist[range_]['total_samples']
                             for range_ in conf_ranges]

            axes[0, 0].bar(conf_ranges, accuracies, color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Accuracy by Confidence Range')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Sample counts
            axes[0, 1].bar(conf_ranges, sample_counts,
                           color='lightgreen', alpha=0.8)
            axes[0, 1].set_title('Sample Count by Confidence Range')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # Calibration analysis
        calibration = self.confidence_analysis.get('calibration_analysis', {})
        if calibration and 'bin_data' in calibration:
            bin_data = calibration['bin_data']

            confidences = [item['avg_confidence'] for item in bin_data]
            accuracies = [item['accuracy'] for item in bin_data]

            axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5,
                            label='Perfect Calibration')
            axes[1, 0].plot(confidences, accuracies, 'o-',
                            color='red', label='Model Calibration')
            axes[1, 0].set_xlabel('Mean Predicted Probability')
            axes[1, 0].set_ylabel('Fraction of Positives')
            axes[1, 0].set_title('Calibration Plot')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Calibration errors
            cal_errors = [item['calibration_error'] for item in bin_data]
            bin_centers = [
                (item['bin_lower'] + item['bin_upper']) / 2 for item in bin_data]

            axes[1, 1].bar(bin_centers, cal_errors, width=0.08,
                           color='orange', alpha=0.8)
            axes[1, 1].set_xlabel('Confidence Bin')
            axes[1, 1].set_ylabel('Calibration Error')
            axes[1, 1].set_title('Calibration Error by Bin')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.model_name} - Confidence Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_error_wordclouds(self, save_dir: Path):
        """Create word clouds for different error types"""
        common_words = self.error_patterns['common_words_in_errors']

        if not common_words:
            return

        # Create subplots for different error types
        n_pairs = len(common_words)
        if n_pairs == 0:
            return

        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_pairs == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for i, (pair, words) in enumerate(common_words.items()):
            if i < len(axes) and words:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=50,
                    colormap='viridis'
                ).generate_from_frequencies(words)

                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'Common Words in {pair} Errors')
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_pairs, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f'{self.model_name} - Error Type Word Clouds', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'error_wordclouds.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_error_report(self, report_path: Optional[str] = None,
                          include_examples: bool = True) -> str:
        """
        Save comprehensive error analysis report

        Args:
            report_path: Optional path to save report
            include_examples: Whether to include error examples

        Returns:
            Path to saved report
        """
        if report_path is None:
            safe_model_name = self.model_name.lower().replace(' ', '_').replace('/', '_')
            report_path = RESULTS_DIR / "error_analysis" / \
                f"{safe_model_name}_error_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Compile comprehensive report
        report = {
            'model_name': self.model_name,
            'analysis_timestamp': self.stats['timestamp'],
            'error_statistics': self.stats,
            'error_analysis': self.error_analysis,
            'error_patterns': self.error_patterns,
            'confidence_analysis': self.confidence_analysis
        }

        # Add example analyses
        if include_examples:
            report['worst_predictions'] = self.get_worst_predictions(10)
            report['confused_predictions'] = self.get_confused_predictions(10)

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Error analysis report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of error analyzer"""
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Generate sample texts
    sample_texts = [
        "اپلیکیشن عالی است و سریع کار می‌کند",
        "برنامه خراب است و مشکل دارد",
        "متوسط است و قابل قبول",
        "بسیار خوب و راحت استفاده می‌شود",
        "کند است و مشکل زیادی دارد"
    ] * (n_samples // 5)

    # Generate labels and predictions with some errors
    class_names = ['negative', 'neutral', 'positive']
    y_true = np.random.choice(3, n_samples)
    y_pred = y_true.copy()

    # Introduce some errors
    error_indices = np.random.choice(
        n_samples, size=int(0.15 * n_samples), replace=False)
    y_pred[error_indices] = np.random.choice(3, len(error_indices))

    # Generate probabilities
    y_proba = np.random.random((n_samples, 3))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    # Initialize analyzer
    analyzer = PersianSentimentErrorAnalyzer()

    # Perform error analysis
    results = analyzer.analyze_errors(
        y_true=y_true,
        y_pred=y_pred,
        texts=sample_texts,
        y_proba=y_proba,
        class_names=class_names,
        model_name="Sample Model"
    )

    # Create visualizations
    plots = analyzer.create_error_visualizations()

    # Get worst predictions
    worst_predictions = analyzer.get_worst_predictions(5)

    # Save report
    report_path = analyzer.save_error_report()

    # Display results
    print(f"\n✅ Error Analysis Completed!")
    print(
        f"📊 Total errors: {analyzer.stats['total_errors']} ({analyzer.stats['error_rate']:.2%})")
    print(f"📈 Visualizations: {len(plots)} plots created")
    print(f"⚠️ Worst predictions: {len(worst_predictions)} examples")
    print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
