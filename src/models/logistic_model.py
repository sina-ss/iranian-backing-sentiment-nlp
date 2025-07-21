"""
Logistic Regression Model for Persian Banking Sentiment Classification
Advanced logistic regression with multiple feature types and comprehensive evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        MODEL_CONFIG,
        TRAINING_CONFIG,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        SENTIMENT_LABELS
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {Path(__file__).parent.parent.parent}")


class PersianLogisticClassifier:
    """
    Advanced Logistic Regression classifier for Persian sentiment analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize logistic regression classifier

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or MODEL_CONFIG.get('logistic_regression', {})
        self.training_config = TRAINING_CONFIG
        self.setup_logging()

        # Model components
        self.model = None
        self.pipeline = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Keep track of which feature blocks were used to fit the model
        self.trained_feature_types: List[str] = ['tfidf']
        self.tfidf_vectorizer = None
        self.feature_extractors = {}

        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Results storage
        self.training_history = {}
        self.evaluation_results = {}
        self.cross_validation_results = {}

        # Statistics
        self.stats = {
            'training_time': 0.0,
            'prediction_time': 0.0,
            'feature_count': 0,
            'sample_count': 0,
            'model_size_mb': 0.0,
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for model training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "logistic_model.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: str, text_column: str = 'comment_processed',
                  label_column: str = 'sentiment_label',
                  additional_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare data for training

        Args:
            data_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column
            additional_features: Additional feature columns to include

        Returns:
            Tuple of (features_df, labels)
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            # Validate required columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found")

            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            df = df[df[text_column].str.len() > 0]

            # Prepare features DataFrame
            features_df = df[[text_column]].copy()

            # Add additional features if specified
            if additional_features:
                available_features = [
                    col for col in additional_features if col in df.columns]
                if available_features:
                    features_df = pd.concat(
                        [features_df, df[available_features]], axis=1)
                    self.logger.info(
                        f"Added additional features: {available_features}")

            # Prepare labels
            labels = df[label_column].values

            # Encode labels to numeric
            self.label_encoder.fit(labels)
            labels_encoded = self.label_encoder.transform(labels)

            self.logger.info(f"Loaded {len(df)} samples")
            self.logger.info(
                f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
            self.logger.info(f"Features shape: {features_df.shape}")

            return features_df, labels_encoded

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def prepare_features(self, features_df: pd.DataFrame,
                         feature_types: List[str] = ['tfidf'],
                         fit_transformers: bool = True) -> csr_matrix:
        """
        Prepare feature matrix from different feature types

        Args:
            features_df: Features DataFrame
            feature_types: Types of features to extract
            fit_transformers: Whether to fit transformers (True for training)

        Returns:
            Feature matrix
        """
        self.logger.info(f"Preparing features: {feature_types}")

        feature_matrices = []
        feature_names = []

        text_column = features_df.columns[0]  # Assume first column is text
        texts = features_df[text_column].fillna('').astype(str)

        # TF-IDF features
        if 'tfidf' in feature_types:
            if fit_transformers:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    lowercase=False,  # Already preprocessed
                    strip_accents=None,
                    analyzer='word'
                )
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.feature_extractors['tfidf'] = self.tfidf_vectorizer
            else:
                if self.tfidf_vectorizer is None:
                    raise ValueError("TF-IDF vectorizer not fitted")
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)

            feature_matrices.append(tfidf_matrix)
            tfidf_names = [
                f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            feature_names.extend(tfidf_names)
            self.logger.info(f"TF-IDF features: {tfidf_matrix.shape}")

        # Text length features
        if 'text_length' in feature_types:
            length_features = self._extract_text_length_features(texts)
            feature_matrices.append(csr_matrix(length_features))
            feature_names.extend(
                ['char_count', 'word_count', 'avg_word_length', 'sentence_count'])
            self.logger.info(f"Text length features: {length_features.shape}")

        # Sentiment lexicon features
        if 'sentiment_lexicon' in feature_types:
            lexicon_features = self._extract_sentiment_lexicon_features(texts)
            feature_matrices.append(csr_matrix(lexicon_features))
            feature_names.extend(
                ['positive_words', 'negative_words', 'sentiment_score'])
            self.logger.info(
                f"Sentiment lexicon features: {lexicon_features.shape}")

        # N-gram features
        if 'ngrams' in feature_types:
            ngram_features = self._extract_ngram_features(
                texts, fit_transformers)
            feature_matrices.append(ngram_features)
            ngram_names = [f"ngram_{i}" for i in range(
                ngram_features.shape[1])]
            feature_names.extend(ngram_names)
            self.logger.info(f"N-gram features: {ngram_features.shape}")

        # Combine all feature matrices
        if len(feature_matrices) == 1:
            combined_features = feature_matrices[0]
        else:
            combined_features = hstack(feature_matrices)

        self.feature_names = feature_names
        self.stats['feature_count'] = combined_features.shape[1]

        self.logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features

    def _extract_text_length_features(self, texts: pd.Series) -> np.ndarray:
        """Extract text length-based features"""
        features = []

        for text in texts:
            text = str(text)
            words = text.split()

            char_count = len(text)
            word_count = len(words)
            avg_word_length = np.mean([len(word)
                                      for word in words]) if words else 0
            sentence_count = text.count(
                '.') + text.count('!') + text.count('?') + 1

            features.append(
                [char_count, word_count, avg_word_length, sentence_count])

        return np.array(features)

    def _extract_sentiment_lexicon_features(self, texts: pd.Series) -> np.ndarray:
        """Extract sentiment lexicon-based features"""
        # Persian sentiment words (simplified)
        positive_words = {
            'خوب', 'عالی', 'فوق‌العاده', 'بهترین', 'راحت', 'سریع', 'آسان',
            'مناسب', 'کامل', 'درست', 'صحیح', 'موفق', 'مطلوب', 'قابل‌قبول'
        }

        negative_words = {
            'بد', 'افتضاح', 'ضعیف', 'کند', 'مشکل', 'خراب', 'اشتباه',
            'غلط', 'نادرست', 'ناکام', 'نامناسب', 'غیرقابل‌قبول'
        }

        features = []

        for text in texts:
            text = str(text)
            words = set(text.split())

            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            sentiment_score = positive_count - negative_count

            features.append([positive_count, negative_count, sentiment_score])

        return np.array(features)

    def _extract_ngram_features(self, texts: pd.Series, fit_transformers: bool) -> csr_matrix:
        """Extract character n-gram features"""
        if fit_transformers:
            ngram_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=5000,
                min_df=2
            )
            ngram_matrix = ngram_vectorizer.fit_transform(texts)
            self.feature_extractors['ngrams'] = ngram_vectorizer
        else:
            ngram_vectorizer = self.feature_extractors.get('ngrams')
            if ngram_vectorizer is None:
                raise ValueError("N-gram vectorizer not fitted")
            ngram_matrix = ngram_vectorizer.transform(texts)

        return ngram_matrix

    def create_model(self, custom_config: Optional[Dict] = None) -> LogisticRegression:
        """
        Create logistic regression model

        Args:
            custom_config: Optional custom configuration

        Returns:
            Logistic regression model
        """
        config = custom_config or self.config

        model = LogisticRegression(
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 1000),
            random_state=config.get('random_state', 42),
            solver=config.get('solver', 'liblinear'),
            multi_class='ovr',  # One-vs-rest for multiclass
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        )

        self.logger.info(
            f"Created logistic regression model with config: {config}")
        return model

    def train_model(self, features_df: pd.DataFrame, labels: np.ndarray,
                    feature_types: List[str] = ['tfidf'],
                    test_size: float = 0.2,
                    validation_split: bool = True) -> Dict:
        """
        Train the logistic regression model

        Args:
            features_df: Features DataFrame
            labels: Target labels
            feature_types: Types of features to use
            test_size: Test set size
            validation_split: Whether to create validation set

        Returns:
            Training results dictionary
        """
        start_time = datetime.now()
        self.logger.info("Starting logistic regression training...")

        # Prepare features
        X = self.prepare_features(
            features_df, feature_types, fit_transformers=True)
        y = labels

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        self.stats['sample_count'] = len(y)

        # Scale features if needed (for non-sparse features)
        if hasattr(X, 'toarray'):  # Sparse matrix
            # Don't scale sparse matrices (TF-IDF is already normalized)
            X_train_scaled = self.X_train
            X_test_scaled = self.X_test
        else:
            # Scale dense features
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled

        # Create and train model
        self.model = self.create_model()

        try:
            # Train model
            self.model.fit(self.X_train, self.y_train)

            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            self.stats['training_time'] = training_time

            # Evaluate on training and test sets
            train_predictions = self.model.predict(self.X_train)
            test_predictions = self.model.predict(self.X_test)

            # Calculate metrics
            training_results = {
                'training_time': training_time,
                'train_accuracy': accuracy_score(self.y_train, train_predictions),
                'test_accuracy': accuracy_score(self.y_test, test_predictions),
                'train_f1': f1_score(self.y_train, train_predictions, average='weighted'),
                'test_f1': f1_score(self.y_test, test_predictions, average='weighted'),
                'feature_types': feature_types,
                'feature_count': X.shape[1],
                'sample_count': len(y)
            }

            self.training_history = training_results

            # remember for later inference
            self.trained_feature_types = feature_types

            self.logger.info(
                f"Training completed in {training_time:.2f} seconds")
            self.logger.info(
                f"Train accuracy: {training_results['train_accuracy']:.4f}")
            self.logger.info(
                f"Test accuracy: {training_results['test_accuracy']:.4f}")

            # Perform cross-validation
            if validation_split:
                cv_results = self.perform_cross_validation(X, y)
                training_results['cross_validation'] = cv_results

            return training_results

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def perform_cross_validation(self, X: csr_matrix, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation evaluation

        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds

        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {cv_folds}-fold cross-validation...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Calculate different metrics
        cv_scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(self.model, X, y, cv=cv, scoring='precision_weighted'),
            'recall': cross_val_score(self.model, X, y, cv=cv, scoring='recall_weighted'),
            'f1': cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        }

        # Calculate summary statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'scores': scores.tolist()
            }

        self.cross_validation_results = cv_results

        self.logger.info(f"CV Results:")
        for metric, results in cv_results.items():
            self.logger.info(
                f"  {metric}: {results['mean']:.4f} ± {results['std']:.4f}")

        return cv_results

    def evaluate_model(self, detailed: bool = True) -> Dict:
        """
        Comprehensive model evaluation

        Args:
            detailed: Whether to include detailed analysis

        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.logger.info("Evaluating model performance...")

        start_time = datetime.now()

        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Get prediction probabilities
        y_train_proba = self.model.predict_proba(self.X_train)
        y_test_proba = self.model.predict_proba(self.X_test)

        prediction_time = (datetime.now() - start_time).total_seconds()
        self.stats['prediction_time'] = prediction_time

        # Calculate metrics
        evaluation = {
            'train_metrics': self._calculate_metrics(self.y_train, y_train_pred, y_train_proba),
            'test_metrics': self._calculate_metrics(self.y_test, y_test_pred, y_test_proba),
            'prediction_time': prediction_time
        }

        if detailed:
            # Feature importance analysis
            evaluation['feature_importance'] = self._analyze_feature_importance()

            # Classification report
            evaluation['classification_report'] = {
                'train': classification_report(
                    self.y_train, y_train_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                ),
                'test': classification_report(
                    self.y_test, y_test_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }

            # Confusion matrices
            evaluation['confusion_matrices'] = {
                'train': confusion_matrix(self.y_train, y_train_pred).tolist(),
                'test': confusion_matrix(self.y_test, y_test_pred).tolist()
            }

        self.evaluation_results = evaluation

        self.logger.info("Model evaluation completed")
        self.logger.info(
            f"Test accuracy: {evaluation['test_metrics']['accuracy']:.4f}")
        self.logger.info(
            f"Test F1-score: {evaluation['test_metrics']['f1_weighted']:.4f}")

        return evaluation

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }

        # Add ROC AUC for multiclass
        try:
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovo')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass

        return {k: float(v) for k, v in metrics.items()}

    def _analyze_feature_importance(self, top_n: int = 20) -> Dict:
        """Analyze feature importance"""
        if self.model is None or self.feature_names is None:
            return {}

        # Get coefficients (for multiclass, this is a matrix)
        coefficients = self.model.coef_

        feature_importance = {}

        if coefficients.ndim == 1:
            # Binary classification
            feature_importance['binary'] = self._get_top_features(
                coefficients, top_n)
        else:
            # Multiclass classification
            classes = self.label_encoder.classes_
            for i, class_name in enumerate(classes):
                feature_importance[class_name] = self._get_top_features(
                    coefficients[i], top_n)

        return feature_importance

    def _get_top_features(self, coefficients: np.ndarray, top_n: int) -> Dict:
        """Get top features by coefficient magnitude"""
        if len(coefficients) != len(self.feature_names):
            return {'error': 'Feature names mismatch'}

        # Get indices of top positive and negative coefficients
        abs_coef = np.abs(coefficients)
        top_indices = np.argsort(abs_coef)[-top_n:][::-1]

        top_features = {
            'positive': [],
            'negative': []
        }

        for idx in top_indices:
            feature_name = self.feature_names[idx]
            coefficient = float(coefficients[idx])

            if coefficient > 0:
                top_features['positive'].append((feature_name, coefficient))
            else:
                top_features['negative'].append((feature_name, coefficient))

        return top_features

    def predict(self, texts: List[str], feature_types: Optional[List[str]] = None):
        """
        Make predictions on new texts

        Args:
            texts: List of texts to predict
            feature_types: Types of features to use

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        # If the caller hasn’t specified anything, fall back to
        # the same blocks used during training.
        if feature_types is None:
            feature_types = self.trained_feature_types

        # Create DataFrame from texts
        text_df = pd.DataFrame(
            {self.feature_names[0] if self.feature_names else 'text': texts})

        # Prepare features
        X = self.prepare_features(
            text_df, feature_types, fit_transformers=False)

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Decode predictions
        predictions_decoded = self.label_encoder.inverse_transform(predictions)

        return predictions_decoded, probabilities

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "logistic_regression"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        if self.evaluation_results:
            # 1. Performance metrics comparison
            self._plot_performance_metrics(save_dir)
            saved_plots.append(
                str(save_dir / 'logistic_performance_metrics.png'))

            # 2. Confusion matrices
            self._plot_confusion_matrices(save_dir)
            saved_plots.append(
                str(save_dir / 'logistic_confusion_matrices.png'))

            # 3. Feature importance
            self._plot_feature_importance(save_dir)
            saved_plots.append(
                str(save_dir / 'logistic_feature_importance.png'))

            # 4. ROC curves
            self._plot_roc_curves(save_dir)
            saved_plots.append(str(save_dir / 'logistic_roc_curves.png'))

        if self.cross_validation_results:
            # 5. Cross-validation results
            self._plot_cross_validation_results(save_dir)
            saved_plots.append(str(save_dir / 'logistic_cross_validation.png'))

        return saved_plots

    def _plot_performance_metrics(self, save_dir: Path):
        """Plot performance metrics comparison"""
        train_metrics = self.evaluation_results['train_metrics']
        test_metrics = self.evaluation_results['test_metrics']

        metrics = ['accuracy', 'precision_weighted',
                   'recall_weighted', 'f1_weighted']
        train_values = [train_metrics[m] for m in metrics]
        test_values = [test_metrics[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, train_values,
                       width, label='Train', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values,
                       width, label='Test', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Logistic Regression Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_performance_metrics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, save_dir: Path):
        """Plot confusion matrices"""
        train_cm = np.array(
            self.evaluation_results['confusion_matrices']['train'])
        test_cm = np.array(
            self.evaluation_results['confusion_matrices']['test'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Train confusion matrix
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=axes[0])
        axes[0].set_title('Train Set Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # Test confusion matrix
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=axes[1])
        axes[1].set_title('Test Set Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_confusion_matrices.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, save_dir: Path):
        """Plot feature importance"""
        if 'feature_importance' not in self.evaluation_results:
            return

        feature_imp = self.evaluation_results['feature_importance']

        if len(feature_imp) == 1 and 'binary' in feature_imp:
            # Binary classification
            self._plot_binary_feature_importance(
                feature_imp['binary'], save_dir)
        else:
            # Multiclass classification
            self._plot_multiclass_feature_importance(feature_imp, save_dir)

    def _plot_binary_feature_importance(self, feature_imp: Dict, save_dir: Path):
        """Plot binary feature importance"""
        pos_features = feature_imp['positive'][:10]
        neg_features = feature_imp['negative'][:10]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Positive features
        if pos_features:
            names, values = zip(*pos_features)
            axes[0].barh(range(len(names)), values, color='green', alpha=0.7)
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names)
            axes[0].set_title('Top Positive Features')
            axes[0].set_xlabel('Coefficient Value')

        # Negative features
        if neg_features:
            names, values = zip(*neg_features)
            axes[1].barh(range(len(names)), values, color='red', alpha=0.7)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels(names)
            axes[1].set_title('Top Negative Features')
            axes[1].set_xlabel('Coefficient Value')

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_multiclass_feature_importance(self, feature_imp: Dict, save_dir: Path):
        """Plot multiclass feature importance"""
        classes = list(feature_imp.keys())
        n_classes = len(classes)

        fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        for i, class_name in enumerate(classes):
            class_features = feature_imp[class_name]

            # Combine positive and negative features
            all_features = class_features['positive'][:5] + \
                class_features['negative'][:5]

            if all_features:
                names, values = zip(*all_features)
                colors = ['green' if v > 0 else 'red' for v in values]

                axes[i].barh(range(len(names)), values,
                             color=colors, alpha=0.7)
                axes[i].set_yticks(range(len(names)))
                axes[i].set_yticklabels(names)
                axes[i].set_title(f'Top Features - {class_name}')
                axes[i].set_xlabel('Coefficient Value')

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, save_dir: Path):
        """Plot ROC curves"""
        if self.model is None:
            return

        # Get probabilities
        y_test_proba = self.model.predict_proba(self.X_test)

        classes = self.label_encoder.classes_
        n_classes = len(classes)

        plt.figure(figsize=(10, 8))

        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.y_test, y_test_proba[:, 1])
            roc_auc = roc_auc_score(self.y_test, y_test_proba[:, 1])

            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

        else:
            # Multiclass classification
            for i, class_name in enumerate(classes):
                # One-vs-rest ROC
                y_test_binary = (self.y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_binary, y_test_proba[:, i])
                roc_auc = roc_auc_score(y_test_binary, y_test_proba[:, i])

                plt.plot(fpr, tpr, lw=2,
                         label=f'{class_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Logistic Regression')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_roc_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_validation_results(self, save_dir: Path):
        """Plot cross-validation results"""
        cv_results = self.cross_validation_results

        metrics = list(cv_results.keys())
        means = [cv_results[m]['mean'] for m in metrics]
        stds = [cv_results[m]['std'] for m in metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot with error bars
        x = np.arange(len(metrics))
        bars = ax1.bar(x, means, yerr=stds, capsize=5,
                       alpha=0.7, color='skyblue')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Cross-Validation Results (Mean ± Std)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

        # Box plot of CV scores
        cv_scores_data = [cv_results[m]['scores'] for m in metrics]
        bp = ax2.boxplot(cv_scores_data, labels=[
                         m.replace('_', ' ').title() for m in metrics])
        ax2.set_ylabel('Score')
        ax2.set_title('Cross-Validation Score Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'logistic_cross_validation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model

        Args:
            model_path: Optional path to save the model

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if model_path is None:
            model_path = MODELS_DIR / "saved_models" / "logistic_regression_model.pkl"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model data
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_extractors': self.feature_extractors,
            'feature_names': self.feature_names,
            'config': self.config,
            'training_config': self.training_config,
            'stats': self.stats,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results,
            'cross_validation_results': self.cross_validation_results
        }

        # Save using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Calculate model size
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        self.stats['model_size_mb'] = model_size_mb

        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Model size: {model_size_mb:.2f} MB")

        return str(model_path)

    def load_model(self, model_path: str):
        """
        Load a saved model

        Args:
            model_path: Path to saved model
        """
        self.logger.info(f"Loading model from {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_extractors = model_data['feature_extractors']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', {})
        self.training_config = model_data.get('training_config', {})
        self.stats = model_data.get('stats', {})
        self.training_history = model_data.get('training_history', {})
        self.evaluation_results = model_data.get('evaluation_results', {})
        self.cross_validation_results = model_data.get(
            'cross_validation_results', {})

        # Update extractors
        self.tfidf_vectorizer = self.feature_extractors.get('tfidf')

        self.logger.info("Model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive model report"""
        self.stats['timestamp'] = datetime.now().isoformat()

        report = {
            'model_info': {
                'model_type': 'Logistic Regression',
                'timestamp': self.stats['timestamp'],
                'configuration': self.config,
                'training_configuration': self.training_config
            },
            'training_summary': self.training_history,
            'evaluation_results': self.evaluation_results,
            'cross_validation_results': self.cross_validation_results,
            'model_statistics': self.stats,
            'feature_analysis': {
                'feature_count': self.stats.get('feature_count', 0),
                'feature_types': self.training_history.get('feature_types', []),
                'top_features': self.evaluation_results.get('feature_importance', {})
            }
        }

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save model report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / "logistic_regression_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Model report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of logistic regression classifier"""
    # Initialize classifier
    classifier = PersianLogisticClassifier()

    # Load data
    data_path = PROCESSED_DATA_DIR / "comments_medium_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocessing first.")
        return

    features_df, labels = classifier.load_data(
        str(data_path),
        text_column='comment_processed',
        label_column='sentiment_label'
    )

    # Train model with different feature types
    training_results = classifier.train_model(
        features_df,
        labels,
        feature_types=['tfidf', 'text_length', 'sentiment_lexicon'],
        test_size=0.2
    )

    # Evaluate model
    evaluation_results = classifier.evaluate_model(detailed=True)

    # Create visualizations
    plots = classifier.create_visualizations()

    # Save model and report
    model_path = classifier.save_model()
    report_path = classifier.save_report()

    # Display results
    print(f"\n✅ Logistic Regression Training Completed!")
    print(
        f"📊 Test Accuracy: {evaluation_results['test_metrics']['accuracy']:.4f}")
    print(
        f"📊 Test F1-Score: {evaluation_results['test_metrics']['f1_weighted']:.4f}")
    print(f"⏱️ Training Time: {training_results['training_time']:.2f} seconds")
    print(f"🎯 Features Used: {training_results['feature_count']}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"📈 Visualizations: {len(plots)} plots created")

    # Test prediction
    test_texts = [
        "اپلیکیشن عالی است و سریع کار می‌کند",
        "برنامه خراب است و مشکل دارد",
        "متوسط است و قابل قبول"
    ]

    predictions, probabilities = classifier.predict(test_texts)
    print(f"\n🧪 Test Predictions:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print(
            f"Probabilities: {dict(zip(classifier.label_encoder.classes_, prob))}")
        print("-" * 50)


if __name__ == "__main__":
    main()
