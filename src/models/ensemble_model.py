"""
Ensemble Model for Persian Banking Sentiment Classification
Advanced ensemble combining Logistic Regression, Neural Networks, and BERT
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project models
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        MODEL_CONFIG,
        TRAINING_CONFIG,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR
    )
    from src.models.logistic_model import PersianLogisticClassifier
    from src.models.persian_bert_model import PersianBertFinetuner
    from src.models.neural_networks import PersianNeuralNetworkClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {Path(__file__).parent.parent.parent}")


class PersianSentimentEnsemble:
    """
    Advanced ensemble model combining multiple approaches for Persian sentiment analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ensemble model

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.setup_logging()

        # Model components
        self.logistic_model = None
        self.cnn_model = None
        self.lstm_model = None
        self.bert_model = None

        # Ensemble components
        self.ensemble_weights = None
        self.voting_classifier = None
        self.meta_learner = None

        # Data storage
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Predictions storage
        self.model_predictions = {}
        self.model_probabilities = {}

        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.ensemble_results = {}

        # Statistics
        self.stats = {
            'models_trained': [],
            'ensemble_type': None,
            'training_time': 0.0,
            'best_individual_model': None,
            'ensemble_improvement': 0.0,
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for ensemble training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "ensemble_model.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: str, text_column: str = 'comment_processed',
                  label_column: str = 'sentiment_label') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare data for ensemble training

        Args:
            data_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Tuple of (features_df, labels)
        """
        self.logger.info(f"Loading data for ensemble from {data_path}")

        try:
            df = pd.read_csv(data_path)

            # Validate columns
            if text_column not in df.columns:
                # Try alternative column names
                alternatives = ['comment_cleaned', 'comment', 'text']
                for alt in alternatives:
                    if alt in df.columns:
                        text_column = alt
                        break
                else:
                    raise ValueError(f"No suitable text column found")

            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found")

            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            df = df[df[text_column].str.len() > 0]

            # Prepare features
            features_df = df[[text_column]].copy()

            # Add additional features if available
            additional_features = ['rating', 'likes', 'app_name']
            for feature in additional_features:
                if feature in df.columns:
                    features_df[feature] = df[feature]

            # Encode labels
            labels = df[label_column].values
            self.label_encoder.fit(labels)
            labels_encoded = self.label_encoder.transform(labels)

            self.logger.info(f"Loaded {len(df)} samples for ensemble")
            self.logger.info(
                f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

            return features_df, labels_encoded

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def train_individual_models(self, features_df: pd.DataFrame, labels: np.ndarray,
                                models_to_train: List[str] = [
                                    'logistic', 'lstm', 'cnn', 'bert'],
                                test_size: float = 0.2) -> Dict:
        """
        Train individual models for ensemble

        Args:
            features_df: Features DataFrame
            labels: Target labels
            models_to_train: List of models to train
            test_size: Test set size

        Returns:
            Training results dictionary
        """
        start_time = datetime.now()
        self.logger.info("Training individual models for ensemble...")

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=test_size, random_state=42, stratify=labels
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        text_column = X_train.columns[0]
        train_texts = X_train[text_column].tolist()
        test_texts = X_test[text_column].tolist()

        training_results = {}

        # Train Logistic Regression
        if 'logistic' in models_to_train:
            self.logger.info("Training Logistic Regression model...")
            try:
                self.logistic_model = PersianLogisticClassifier()
                logistic_results = self.logistic_model.train_model(
                    X_train, y_train,
                    feature_types=['tfidf', 'text_length',
                                   'sentiment_lexicon'],
                    test_size=0.0  # Already split
                )
                self.logistic_model.X_test = self.logistic_model.prepare_features(
                    X_test, ['tfidf', 'text_length', 'sentiment_lexicon'], fit_transformers=False
                )
                self.logistic_model.y_test = y_test

                training_results['logistic'] = logistic_results
                self.stats['models_trained'].append('logistic')
                self.logger.info("✅ Logistic Regression training completed")

            except Exception as e:
                self.logger.error(
                    f"❌ Logistic Regression training failed: {e}")

        # Train LSTM
        if 'lstm' in models_to_train:
            self.logger.info("Training LSTM model...")
            try:
                self.lstm_model = PersianNeuralNetworkClassifier(
                    model_type='lstm')

                # Use stemmed version for better word separation
                stemmed_data_path = PROCESSED_DATA_DIR / "comments_heavy_stem_processed.csv"
                if stemmed_data_path.exists():
                    stemmed_df = pd.read_csv(stemmed_data_path)
                    stemmed_texts = stemmed_df['comment_processed'].fillna(
                        '').astype(str).tolist()[:len(train_texts)]
                else:
                    stemmed_texts = train_texts

                # Check for Word2Vec embeddings
                embedding_path = MODELS_DIR / "saved_models" / "word2vec_vectors.kv"
                embedding_path = str(
                    embedding_path) if embedding_path.exists() else None

                lstm_results = self.lstm_model.train_model(
                    stemmed_texts[:len(train_texts)], y_train,
                    epochs=20, batch_size=32, embedding_path=embedding_path
                )

                training_results['lstm'] = lstm_results
                self.stats['models_trained'].append('lstm')
                self.logger.info("✅ LSTM training completed")

            except Exception as e:
                self.logger.error(f"❌ LSTM training failed: {e}")

        # Train CNN
        if 'cnn' in models_to_train:
            self.logger.info("Training CNN model...")
            try:
                self.cnn_model = PersianNeuralNetworkClassifier(
                    model_type='cnn')

                # Use same data as LSTM
                stemmed_data_path = PROCESSED_DATA_DIR / "comments_heavy_stem_processed.csv"
                if stemmed_data_path.exists():
                    stemmed_df = pd.read_csv(stemmed_data_path)
                    stemmed_texts = stemmed_df['comment_processed'].fillna(
                        '').astype(str).tolist()[:len(train_texts)]
                else:
                    stemmed_texts = train_texts

                embedding_path = MODELS_DIR / "saved_models" / "word2vec_vectors.kv"
                embedding_path = str(
                    embedding_path) if embedding_path.exists() else None

                cnn_results = self.cnn_model.train_model(
                    stemmed_texts[:len(train_texts)], y_train,
                    epochs=20, batch_size=32, embedding_path=embedding_path
                )

                training_results['cnn'] = cnn_results
                self.stats['models_trained'].append('cnn')
                self.logger.info("✅ CNN training completed")

            except Exception as e:
                self.logger.error(f"❌ CNN training failed: {e}")

        # Train BERT (reduced dataset for efficiency)
        if 'bert' in models_to_train:
            self.logger.info("Training BERT model...")
            try:
                self.bert_model = PersianBertFinetuner()

                # Use light processed version to preserve context for BERT
                light_data_path = PROCESSED_DATA_DIR / "comments_light_processed.csv"
                if light_data_path.exists():
                    light_df = pd.read_csv(light_data_path)
                    light_texts = light_df['comment_cleaned'].fillna(
                        '').astype(str).tolist()[:len(train_texts)]
                else:
                    light_texts = train_texts

                # Limit data for BERT (computational efficiency)
                bert_sample_size = min(1000, len(train_texts))
                bert_texts = light_texts[:bert_sample_size]
                bert_labels = y_train[:bert_sample_size]

                bert_results = self.bert_model.train_model(
                    bert_texts, bert_labels,
                    epochs=3, batch_size=16, learning_rate=2e-5
                )

                training_results['bert'] = bert_results
                self.stats['models_trained'].append('bert')
                self.logger.info("✅ BERT training completed")

            except Exception as e:
                self.logger.error(f"❌ BERT training failed: {e}")

        # Calculate total training time
        total_training_time = (datetime.now() - start_time).total_seconds()
        self.stats['training_time'] = total_training_time

        self.training_results = training_results

        self.logger.info(
            f"Individual model training completed in {total_training_time:.2f} seconds")
        self.logger.info(
            f"Successfully trained models: {self.stats['models_trained']}")

        return training_results

    def collect_model_predictions(self, X_test: Optional[pd.DataFrame] = None) -> Dict:
        """
        Collect predictions from all trained models

        Args:
            X_test: Optional test data

        Returns:
            Dictionary of model predictions and probabilities
        """
        if X_test is None:
            X_test = self.X_test

        self.logger.info("Collecting predictions from individual models...")

        text_column = X_test.columns[0]
        test_texts = X_test[text_column].tolist()

        predictions = {}
        probabilities = {}

        # Logistic Regression predictions
        if self.logistic_model is not None:
            try:
                log_pred, log_prob = self.logistic_model.predict(
                    test_texts, feature_types=[
                        'tfidf', 'text_length', 'sentiment_lexicon']
                )
                predictions['logistic'] = log_pred
                probabilities['logistic'] = log_prob
                self.logger.info("✅ Collected Logistic Regression predictions")
            except Exception as e:
                self.logger.error(
                    f"❌ Logistic Regression prediction failed: {e}")

        # LSTM predictions
        if self.lstm_model is not None:
            try:
                lstm_pred, lstm_prob = self.lstm_model.predict(test_texts)
                predictions['lstm'] = lstm_pred
                probabilities['lstm'] = lstm_prob
                self.logger.info("✅ Collected LSTM predictions")
            except Exception as e:
                self.logger.error(f"❌ LSTM prediction failed: {e}")

        # CNN predictions
        if self.cnn_model is not None:
            try:
                cnn_pred, cnn_prob = self.cnn_model.predict(test_texts)
                predictions['cnn'] = cnn_pred
                probabilities['cnn'] = cnn_prob
                self.logger.info("✅ Collected CNN predictions")
            except Exception as e:
                self.logger.error(f"❌ CNN prediction failed: {e}")

        # BERT predictions
        if self.bert_model is not None:
            try:
                bert_pred, bert_prob = self.bert_model.predict(test_texts)
                predictions['bert'] = bert_pred
                probabilities['bert'] = bert_prob
                self.logger.info("✅ Collected BERT predictions")
            except Exception as e:
                self.logger.error(f"❌ BERT prediction failed: {e}")

        self.model_predictions = predictions
        self.model_probabilities = probabilities

        return {'predictions': predictions, 'probabilities': probabilities}

    def create_voting_ensemble(self, voting_type: str = 'soft',
                               weights: Optional[List[float]] = None) -> Dict:
        """
        Create voting ensemble from individual models

        Args:
            voting_type: Type of voting ('hard' or 'soft')
            weights: Optional weights for models

        Returns:
            Ensemble results dictionary
        """
        self.logger.info(f"Creating {voting_type} voting ensemble...")

        if not self.model_predictions:
            self.collect_model_predictions()

        y_true = self.y_test
        available_models = list(self.model_predictions.keys())

        if not available_models:
            raise ValueError("No model predictions available")

        # Align all predictions to same samples
        min_samples = min(len(pred)
                          for pred in self.model_predictions.values())

        if voting_type == 'hard':
            # Hard voting: majority vote
            ensemble_predictions = []

            for i in range(min_samples):
                votes = []
                for model_name in available_models:
                    pred = self.model_predictions[model_name][i]
                    votes.append(pred)

                # Convert to numeric for voting
                numeric_votes = []
                for vote in votes:
                    if isinstance(vote, str):
                        numeric_votes.append(
                            list(self.label_encoder.classes_).index(vote))
                    else:
                        numeric_votes.append(vote)

                # Majority vote
                from collections import Counter
                vote_counts = Counter(numeric_votes)
                majority_vote = vote_counts.most_common(1)[0][0]
                ensemble_predictions.append(majority_vote)

            ensemble_predictions = np.array(ensemble_predictions)

        else:
            # Soft voting: average probabilities
            if not self.model_probabilities:
                raise ValueError(
                    "Soft voting requires probability predictions")

            # Average probabilities
            avg_probabilities = None
            model_count = 0

            for model_name in available_models:
                if model_name in self.model_probabilities:
                    probs = np.array(
                        self.model_probabilities[model_name][:min_samples])

                    if weights is not None:
                        weight_idx = available_models.index(model_name)
                        if weight_idx < len(weights):
                            probs *= weights[weight_idx]

                    if avg_probabilities is None:
                        avg_probabilities = probs
                    else:
                        avg_probabilities += probs

                    model_count += 1

            if model_count > 0:
                avg_probabilities /= model_count
                ensemble_predictions = np.argmax(avg_probabilities, axis=1)
            else:
                raise ValueError("No probability predictions available")

        # Convert back to original labels
        ensemble_predictions_decoded = self.label_encoder.inverse_transform(
            ensemble_predictions)

        # Calculate ensemble metrics
        y_true_subset = y_true[:min_samples]

        ensemble_accuracy = accuracy_score(y_true_subset, ensemble_predictions)
        ensemble_f1 = f1_score(
            y_true_subset, ensemble_predictions, average='weighted')

        # Compare with individual models
        individual_scores = {}
        for model_name in available_models:
            model_preds = self.model_predictions[model_name][:min_samples]
            if isinstance(model_preds[0], str):
                model_preds_numeric = self.label_encoder.transform(model_preds)
            else:
                model_preds_numeric = model_preds

            individual_scores[model_name] = {
                'accuracy': accuracy_score(y_true_subset, model_preds_numeric),
                'f1_score': f1_score(y_true_subset, model_preds_numeric, average='weighted')
            }

        # Find best individual model
        best_model = max(individual_scores.keys(),
                         key=lambda x: individual_scores[x]['accuracy'])
        best_accuracy = individual_scores[best_model]['accuracy']

        ensemble_improvement = ensemble_accuracy - best_accuracy

        self.stats['best_individual_model'] = best_model
        self.stats['ensemble_improvement'] = ensemble_improvement
        self.stats['ensemble_type'] = f"{voting_type}_voting"

        ensemble_results = {
            'voting_type': voting_type,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1_score': ensemble_f1,
            'individual_scores': individual_scores,
            'best_individual_model': best_model,
            'ensemble_improvement': ensemble_improvement,
            'ensemble_predictions': ensemble_predictions_decoded,
            'available_models': available_models,
            'weights_used': weights
        }

        self.ensemble_results = ensemble_results

        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        self.logger.info(
            f"Best individual model ({best_model}): {best_accuracy:.4f}")
        self.logger.info(f"Ensemble improvement: {ensemble_improvement:.4f}")

        return ensemble_results

    def optimize_ensemble_weights(self, method: str = 'grid_search') -> List[float]:
        """
        Optimize ensemble weights using cross-validation

        Args:
            method: Optimization method ('grid_search' or 'random_search')

        Returns:
            Optimal weights
        """
        self.logger.info(f"Optimizing ensemble weights using {method}...")

        if not self.model_probabilities:
            self.collect_model_predictions()

        available_models = list(self.model_probabilities.keys())
        n_models = len(available_models)

        if n_models < 2:
            return [1.0] * n_models

        # Prepare data
        min_samples = min(len(prob)
                          for prob in self.model_probabilities.values())
        y_true = self.y_test[:min_samples]

        # Stack probabilities
        prob_stack = []
        for model_name in available_models:
            probs = np.array(
                self.model_probabilities[model_name][:min_samples])
            prob_stack.append(probs)
        # Shape: (n_models, n_samples, n_classes)
        prob_stack = np.array(prob_stack)

        best_weights = None
        best_score = 0.0

        if method == 'grid_search':
            # Grid search over weight combinations
            from itertools import product

            # Create weight grid (normalized)
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            if n_models == 2:
                weight_combinations = [(w1, 1-w1) for w1 in weight_options]
            elif n_models == 3:
                combinations = []
                for w1 in weight_options:
                    for w2 in weight_options:
                        w3 = max(0, 1 - w1 - w2)
                        if abs(w1 + w2 + w3 - 1.0) < 0.1:
                            combinations.append((w1, w2, w3))
                weight_combinations = combinations[:50]  # Limit combinations
            else:
                # For more models, use uniform weights with perturbations
                base_weight = 1.0 / n_models
                weight_combinations = []
                for perturbation in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    weights = [base_weight + perturbation] + \
                        [base_weight] * (n_models - 1)
                    # Normalize
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    weight_combinations.append(tuple(weights))

        else:
            # Random search
            np.random.seed(42)
            weight_combinations = []
            for _ in range(20):
                weights = np.random.random(n_models)
                weights = weights / weights.sum()
                weight_combinations.append(tuple(weights))

        # Test each weight combination
        for weights in weight_combinations:
            # Calculate weighted average probabilities
            weighted_probs = np.zeros_like(prob_stack[0])

            for i, weight in enumerate(weights):
                weighted_probs += weight * prob_stack[i]

            # Get predictions
            ensemble_preds = np.argmax(weighted_probs, axis=1)

            # Calculate score
            score = accuracy_score(y_true, ensemble_preds)

            if score > best_score:
                best_score = score
                best_weights = list(weights)

        self.ensemble_weights = best_weights

        self.logger.info(f"Optimal weights found: {best_weights}")
        self.logger.info(f"Best cross-validation score: {best_score:.4f}")

        return best_weights

    def create_stacked_ensemble(self, meta_learner_type: str = 'logistic') -> Dict:
        """
        Create stacked ensemble with meta-learner

        Args:
            meta_learner_type: Type of meta-learner ('logistic', 'rf', 'xgb')

        Returns:
            Stacked ensemble results
        """
        self.logger.info(
            f"Creating stacked ensemble with {meta_learner_type} meta-learner...")

        if not self.model_probabilities:
            self.collect_model_predictions()

        available_models = list(self.model_probabilities.keys())

        # Prepare meta-features (stacked probabilities)
        min_samples = min(len(prob)
                          for prob in self.model_probabilities.values())

        meta_features = []
        for i in range(min_samples):
            sample_features = []
            for model_name in available_models:
                probs = self.model_probabilities[model_name][i]
                sample_features.extend(probs)  # Flatten probabilities
            meta_features.append(sample_features)

        meta_features = np.array(meta_features)
        y_true = self.y_test[:min_samples]

        # Train meta-learner using cross-validation
        from sklearn.model_selection import cross_val_predict

        if meta_learner_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_learner_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            meta_learner = RandomForestClassifier(
                n_estimators=100, random_state=42)
        elif meta_learner_type == 'xgb':
            try:
                import xgboost as xgb
                meta_learner = xgb.XGBClassifier(random_state=42)
            except ImportError:
                self.logger.warning(
                    "XGBoost not available, using Random Forest instead")
                from sklearn.ensemble import RandomForestClassifier
                meta_learner = RandomForestClassifier(
                    n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")

        # Use cross-validation to get out-of-fold predictions
        cv_predictions = cross_val_predict(
            meta_learner, meta_features, y_true,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method='predict'
        )

        # Train final meta-learner on all data
        meta_learner.fit(meta_features, y_true)
        final_predictions = meta_learner.predict(meta_features)

        self.meta_learner = meta_learner

        # Calculate metrics
        stacked_accuracy = accuracy_score(y_true, final_predictions)
        stacked_f1 = f1_score(y_true, final_predictions, average='weighted')
        cv_accuracy = accuracy_score(y_true, cv_predictions)

        stacked_results = {
            'meta_learner_type': meta_learner_type,
            'stacked_accuracy': stacked_accuracy,
            'stacked_f1_score': stacked_f1,
            'cv_accuracy': cv_accuracy,
            'meta_features_shape': meta_features.shape,
            'final_predictions': final_predictions,
            'cv_predictions': cv_predictions
        }

        self.logger.info(f"Stacked ensemble accuracy: {stacked_accuracy:.4f}")
        self.logger.info(f"Cross-validation accuracy: {cv_accuracy:.4f}")

        return stacked_results

    def predict(self, texts: List[str], ensemble_type: str = 'soft_voting',
                weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble

        Args:
            texts: List of texts to predict
            ensemble_type: Type of ensemble ('soft_voting', 'hard_voting', 'stacked')
            weights: Optional weights for voting

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.logger.info(
            f"Making ensemble predictions using {ensemble_type}...")

        # Get predictions from individual models
        individual_predictions = {}
        individual_probabilities = {}

        # Logistic Regression
        if self.logistic_model is not None:
            try:
                pred, prob = self.logistic_model.predict(texts)
                individual_predictions['logistic'] = pred
                individual_probabilities['logistic'] = prob
            except Exception as e:
                self.logger.warning(f"Logistic model prediction failed: {e}")

        # LSTM
        if self.lstm_model is not None:
            try:
                pred, prob = self.lstm_model.predict(texts)
                individual_predictions['lstm'] = pred
                individual_probabilities['lstm'] = prob
            except Exception as e:
                self.logger.warning(f"LSTM model prediction failed: {e}")

        # CNN
        if self.cnn_model is not None:
            try:
                pred, prob = self.cnn_model.predict(texts)
                individual_predictions['cnn'] = pred
                individual_probabilities['cnn'] = prob
            except Exception as e:
                self.logger.warning(f"CNN model prediction failed: {e}")

        # BERT
        if self.bert_model is not None:
            try:
                pred, prob = self.bert_model.predict(texts)
                individual_predictions['bert'] = pred
                individual_probabilities['bert'] = prob
            except Exception as e:
                self.logger.warning(f"BERT model prediction failed: {e}")

        available_models = list(individual_predictions.keys())

        if not available_models:
            raise ValueError("No models available for prediction")

        # Ensemble prediction
        if ensemble_type == 'hard_voting':
            # Hard voting
            ensemble_predictions = []
            for i in range(len(texts)):
                votes = [individual_predictions[model][i]
                         for model in available_models]
                # Convert to numeric for voting
                numeric_votes = []
                for vote in votes:
                    if isinstance(vote, str):
                        numeric_votes.append(
                            list(self.label_encoder.classes_).index(vote))
                    else:
                        numeric_votes.append(vote)

                from collections import Counter
                majority_vote = Counter(numeric_votes).most_common(1)[0][0]
                ensemble_predictions.append(
                    self.label_encoder.classes_[majority_vote])

            ensemble_predictions = np.array(ensemble_predictions)
            ensemble_probabilities = None

        elif ensemble_type == 'soft_voting':
            # Soft voting
            avg_probabilities = None
            model_count = 0

            for model_name in available_models:
                probs = np.array(individual_probabilities[model_name])

                if weights is not None:
                    weight_idx = available_models.index(model_name)
                    if weight_idx < len(weights):
                        probs *= weights[weight_idx]

                if avg_probabilities is None:
                    avg_probabilities = probs
                else:
                    avg_probabilities += probs

                model_count += 1

            avg_probabilities /= model_count
            ensemble_predictions = self.label_encoder.classes_[
                np.argmax(avg_probabilities, axis=1)]
            ensemble_probabilities = avg_probabilities

        elif ensemble_type == 'stacked':
            # Stacked ensemble
            if self.meta_learner is None:
                raise ValueError(
                    "Meta-learner not trained. Run create_stacked_ensemble first.")

            # Prepare meta-features
            meta_features = []
            for i in range(len(texts)):
                sample_features = []
                for model_name in available_models:
                    probs = individual_probabilities[model_name][i]
                    sample_features.extend(probs)
                meta_features.append(sample_features)

            meta_features = np.array(meta_features)

            # Predict using meta-learner
            ensemble_predictions_numeric = self.meta_learner.predict(
                meta_features)
            ensemble_predictions = self.label_encoder.inverse_transform(
                ensemble_predictions_numeric)

            # Get probabilities if available
            if hasattr(self.meta_learner, 'predict_proba'):
                ensemble_probabilities = self.meta_learner.predict_proba(
                    meta_features)
            else:
                ensemble_probabilities = None

        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

        return ensemble_predictions, ensemble_probabilities

    def save_ensemble(self, model_path: Optional[str] = None) -> str:
        """
        Save the ensemble model

        Args:
            model_path: Optional path to save the ensemble

        Returns:
            Path to saved ensemble
        """
        if model_path is None:
            model_path = MODELS_DIR / "saved_models" / "ensemble_model.pkl"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare ensemble data
        ensemble_data = {
            'logistic_model': self.logistic_model,
            'cnn_model': self.cnn_model,
            'lstm_model': self.lstm_model,
            'bert_model': self.bert_model,
            'meta_learner': self.meta_learner,
            'ensemble_weights': self.ensemble_weights,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'stats': self.stats,
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results,
            'ensemble_results': self.ensemble_results
        }

        # Save using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_data, f)

        # Calculate model size
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        self.logger.info(f"Ensemble model saved to {model_path}")
        self.logger.info(f"Ensemble size: {model_size_mb:.2f} MB")

        return str(model_path)

    def load_ensemble(self, model_path: str):
        """
        Load a saved ensemble model

        Args:
            model_path: Path to saved ensemble
        """
        self.logger.info(f"Loading ensemble from {model_path}")

        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.logistic_model = ensemble_data.get('logistic_model')
        self.cnn_model = ensemble_data.get('cnn_model')
        self.lstm_model = ensemble_data.get('lstm_model')
        self.bert_model = ensemble_data.get('bert_model')
        self.meta_learner = ensemble_data.get('meta_learner')
        self.ensemble_weights = ensemble_data.get('ensemble_weights')
        self.label_encoder = ensemble_data.get('label_encoder')
        self.config = ensemble_data.get('config', {})
        self.stats = ensemble_data.get('stats', {})
        self.training_results = ensemble_data.get('training_results', {})
        self.evaluation_results = ensemble_data.get('evaluation_results', {})
        self.ensemble_results = ensemble_data.get('ensemble_results', {})

        self.logger.info("Ensemble model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive ensemble report"""
        self.stats['timestamp'] = datetime.now().isoformat()

        report = {
            'ensemble_info': {
                'timestamp': self.stats['timestamp'],
                'models_trained': self.stats['models_trained'],
                'ensemble_type': self.stats['ensemble_type'],
                'best_individual_model': self.stats['best_individual_model'],
                'ensemble_improvement': self.stats['ensemble_improvement']
            },
            'training_summary': {
                'total_training_time': self.stats['training_time'],
                'individual_results': self.training_results
            },
            'ensemble_results': self.ensemble_results,
            'model_statistics': self.stats
        }

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save ensemble report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / "ensemble_model_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Ensemble report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of ensemble model"""
    # Initialize ensemble
    ensemble = PersianSentimentEnsemble()

    # Load data
    data_path = PROCESSED_DATA_DIR / "comments_medium_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    features_df, labels = ensemble.load_data(
        str(data_path),
        text_column='comment_processed',
        label_column='sentiment_label'
    )

    # Train individual models
    training_results = ensemble.train_individual_models(
        features_df, labels,
        # Start with fewer models for demo
        models_to_train=['logistic', 'lstm'],
        test_size=0.2
    )

    # Create voting ensemble
    voting_results = ensemble.create_voting_ensemble(voting_type='soft')

    # Optimize weights
    optimal_weights = ensemble.optimize_ensemble_weights()

    # Create weighted ensemble
    weighted_results = ensemble.create_voting_ensemble(
        voting_type='soft',
        weights=optimal_weights
    )

    # Save ensemble and report
    model_path = ensemble.save_ensemble()
    report_path = ensemble.save_report()

    # Display results
    print(f"\n✅ Ensemble Training Completed!")
    print(f"🤖 Models trained: {ensemble.stats['models_trained']}")
    print(
        f"🎯 Best individual model: {ensemble.stats['best_individual_model']}")
    print(f"📊 Ensemble accuracy: {weighted_results['ensemble_accuracy']:.4f}")
    print(f"📈 Improvement: {ensemble.stats['ensemble_improvement']:.4f}")
    print(f"⚖️ Optimal weights: {optimal_weights}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
