"""
Neural Networks Models for Persian Banking Sentiment Classification
Advanced CNN and LSTM models with comprehensive training and evaluation
"""

import os
import sys
from transformers import AutoTokenizer
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Word embedding libraries

# Import project configuration
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
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {Path(__file__).parent.parent.parent}")


class PersianTextDataset(Dataset):
    """Dataset class for Persian text classification"""

    def __init__(self, texts: List[str], labels: np.ndarray,
                 word_to_idx: Dict[str, int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and convert to indices
        tokens = text.split()
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0))
                   for token in tokens[:self.max_length]]

        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.word_to_idx.get('<PAD>', 0)]
                           * (self.max_length - len(indices)))

        return {
            'text_indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text_length': torch.tensor(min(len(tokens), self.max_length), dtype=torch.long)
        }


class CNNSentimentClassifier(nn.Module):
    """CNN model for sentiment classification"""

    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int,
                 filter_sizes: List[int], num_classes: int, dropout: float = 0.3,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(CNNSentimentClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow fine-tuning

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])

        # Dropout and classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x, text_lengths=None):
        # x: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # (batch_size, num_filters, conv_length)
            conv_out = F.relu(conv(x))
            # (batch_size, num_filters, 1)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)

        # Concatenate all conv outputs
        # (batch_size, len(filter_sizes) * num_filters)
        x = torch.cat(conv_outputs, dim=1)

        # Apply dropout and final classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class LSTMSentimentClassifier(nn.Module):
    """LSTM model for sentiment classification"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int, dropout: float = 0.3,
                 bidirectional: bool = True, pretrained_embeddings: Optional[torch.Tensor] = None):
        super(LSTMSentimentClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x, text_lengths=None):
        # x: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # Pack padded sequences if lengths are provided
        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state (for bidirectional, concatenate both directions)
        if self.bidirectional:
            # hidden: (num_layers * 2, batch_size, hidden_dim)
            # Get last layer's forward and backward hidden states
            hidden = hidden[-2:, :, :]
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # Concatenate
        else:
            hidden = hidden[-1, :, :]  # Get last layer's hidden state

        # Apply dropout and classification
        output = self.dropout(hidden)
        output = self.fc(output)

        return output


class PersianNeuralNetworkClassifier:
    """
    Advanced Neural Network classifier for Persian sentiment analysis
    """

    def __init__(self, model_type: str = 'lstm', config: Optional[Dict] = None):
        """
        Initialize neural network classifier

        Args:
            model_type: Type of model ('lstm' or 'cnn')
            config: Optional configuration dictionary
        """
        self.model_type = model_type.lower()
        self.config = config or MODEL_CONFIG.get(
            'neural_network', {}).get(self.model_type, {})
        self.training_config = TRAINING_CONFIG

        self.setup_logging()

        # Model components
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Data components
        self.label_encoder = LabelEncoder()
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        self.embedding_matrix = None

        # Training data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Results storage
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': []
        }
        self.evaluation_results = {}

        # Statistics
        self.stats = {
            'model_type': self.model_type,
            'training_time': 0.0,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'model_size_mb': 0.0,
            'device_used': str(self.device),
            'timestamp': None
        }

        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def setup_logging(self):
        """Setup logging for neural network training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"neural_network_{self.model_type}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: str, text_column: str = 'comment_processed',
                  label_column: str = 'sentiment_label') -> Tuple[List[str], np.ndarray]:
        """
        Load and prepare data for training

        Args:
            data_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Tuple of (texts, labels)
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            # Validate columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found")

            # Clean data
            df = df.dropna(subset=[text_column, label_column])
            df = df[df[text_column].str.len() > 0]

            # Extract texts and labels
            texts = df[text_column].astype(str).tolist()
            labels = df[label_column].values

            # Encode labels
            self.label_encoder.fit(labels)
            labels_encoded = self.label_encoder.transform(labels)

            self.logger.info(f"Loaded {len(texts)} samples")
            self.logger.info(
                f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

            return texts, labels_encoded

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from texts

        Args:
            texts: List of texts
            min_freq: Minimum frequency for word inclusion

        Returns:
            Word to index mapping
        """
        self.logger.info("Building vocabulary...")

        # Count word frequencies
        word_freq = {}
        for text in texts:
            for word in str(text).split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add words with sufficient frequency
        idx = len(self.word_to_idx)
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        self.vocab_size = len(self.word_to_idx)

        self.logger.info(f"Vocabulary size: {self.vocab_size}")
        self.logger.info(
            f"Words with freq >= {min_freq}: {self.vocab_size - 2}")

        return self.word_to_idx

    def load_pretrained_embeddings(self, embedding_path: str,
                                   embedding_dim: int = 200) -> torch.Tensor:
        """
        Load pretrained word embeddings

        Args:
            embedding_path: Path to pretrained embeddings
            embedding_dim: Embedding dimension

        Returns:
            Embedding matrix tensor
        """
        self.logger.info(
            f"Loading pretrained embeddings from {embedding_path}")

        try:
            # Load Word2Vec model
            if embedding_path.endswith('.kv'):
                word_vectors = KeyedVectors.load(embedding_path)
            else:
                word_vectors = KeyedVectors.load_word2vec_format(
                    embedding_path)

            # Initialize embedding matrix
            embedding_matrix = np.random.normal(
                0, 0.1, (self.vocab_size, embedding_dim))

            # Fill with pretrained embeddings
            found_words = 0
            for word, idx in self.word_to_idx.items():
                if word in word_vectors:
                    embedding_matrix[idx] = word_vectors[word]
                    found_words += 1

            # Set padding embedding to zeros
            embedding_matrix[0] = np.zeros(embedding_dim)

            self.embedding_matrix = torch.FloatTensor(embedding_matrix)

            coverage = found_words / self.vocab_size
            self.logger.info(
                f"Embedding coverage: {coverage:.2%} ({found_words}/{self.vocab_size})")

            return self.embedding_matrix

        except Exception as e:
            self.logger.warning(f"Could not load pretrained embeddings: {e}")
            self.logger.info("Using random embeddings instead")
            return None

    def create_data_loaders(self, texts: List[str], labels: np.ndarray,
                            batch_size: int = 32, max_length: int = 128,
                            test_size: float = 0.2, val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing

        Args:
            texts: List of texts
            labels: Label array
            batch_size: Batch size
            max_length: Maximum sequence length
            test_size: Test set proportion
            val_size: Validation set proportion

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Creating data loaders...")

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        # Create datasets
        train_dataset = PersianTextDataset(
            X_train, y_train, self.word_to_idx, max_length)
        val_dataset = PersianTextDataset(
            X_val, y_val, self.word_to_idx, max_length)
        test_dataset = PersianTextDataset(
            X_test, y_test, self.word_to_idx, max_length)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        self.logger.info(
            f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return self.train_loader, self.val_loader, self.test_loader

    def create_model(self, num_classes: int, custom_config: Optional[Dict] = None) -> nn.Module:
        """
        Create neural network model

        Args:
            num_classes: Number of output classes
            custom_config: Optional custom configuration

        Returns:
            Neural network model
        """
        config = custom_config or self.config

        if self.model_type == 'cnn':
            model = CNNSentimentClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=config.get('embedding_dim', 200),
                num_filters=config.get('num_filters', 100),
                filter_sizes=config.get('filter_sizes', [3, 4, 5]),
                num_classes=num_classes,
                dropout=config.get('dropout', 0.3),
                pretrained_embeddings=self.embedding_matrix
            )

        elif self.model_type == 'lstm':
            model = LSTMSentimentClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=config.get('embedding_dim', 200),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 2),
                num_classes=num_classes,
                dropout=config.get('dropout', 0.3),
                bidirectional=config.get('bidirectional', True),
                pretrained_embeddings=self.embedding_matrix
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model = model.to(self.device)

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        self.stats['total_parameters'] = total_params
        self.stats['trainable_parameters'] = trainable_params

        self.logger.info(f"Created {self.model_type.upper()} model")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def train_model(self, texts: List[str], labels: np.ndarray,
                    epochs: int = 50, batch_size: int = 32,
                    learning_rate: float = 0.001,
                    embedding_path: Optional[str] = None,
                    early_stopping_patience: int = 5) -> Dict:
        """
        Train the neural network model

        Args:
            texts: List of texts
            labels: Label array
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            embedding_path: Optional path to pretrained embeddings
            early_stopping_patience: Early stopping patience

        Returns:
            Training results dictionary
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {self.model_type.upper()} training...")

        # Build vocabulary
        self.build_vocabulary(texts)

        # Load pretrained embeddings if provided
        if embedding_path:
            self.load_pretrained_embeddings(embedding_path)

        # Create data loaders
        self.create_data_loaders(texts, labels, batch_size)

        # Create model
        num_classes = len(np.unique(labels))
        self.model = self.create_model(num_classes)

        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(),
                               lr=learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch()

            # Validation phase
            val_loss, val_acc = self._validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['epochs'].append(epoch + 1)

            # Log progress
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}")
                self.logger.info(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.logger.info(
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.logger.info(
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model_temp.pth'))
        Path('best_model_temp.pth').unlink(missing_ok=True)

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        self.stats['training_time'] = training_time

        # Final evaluation
        final_train_loss, final_train_acc = self._validate_epoch(
            self.train_loader)
        final_val_loss, final_val_acc = self._validate_epoch(self.val_loader)

        training_results = {
            'epochs_trained': len(self.training_history['epochs']),
            'training_time': training_time,
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'best_val_loss': best_val_loss,
            'model_type': self.model_type,
            'total_parameters': self.stats['total_parameters'],
            'trainable_parameters': self.stats['trainable_parameters']
        }

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Final validation accuracy: {final_val_acc:.4f}")

        return training_results

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            text_indices = batch['text_indices'].to(self.device)
            labels = batch['label'].to(self.device)
            text_lengths = batch['text_length'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_indices, text_lengths)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def _validate_epoch(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """Validate for one epoch"""
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                text_indices = batch['text_indices'].to(self.device)
                labels = batch['label'].to(self.device)
                text_lengths = batch['text_length'].to(self.device)

                # Forward pass
                outputs = self.model(text_indices, text_lengths)
                loss = self.criterion(outputs, labels)

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def evaluate_model(self) -> Dict:
        """
        Comprehensive model evaluation

        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.logger.info("Evaluating model performance...")

        # Evaluate on test set
        test_loss, test_accuracy = self._validate_epoch(self.test_loader)

        # Get detailed predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                text_indices = batch['text_indices'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(text_indices)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate detailed metrics
        evaluation = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': classification_report(
                all_labels, all_predictions,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'f1_score_weighted': f1_score(all_labels, all_predictions, average='weighted'),
            'f1_score_macro': f1_score(all_labels, all_predictions, average='macro')
        }

        self.evaluation_results = evaluation

        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        self.logger.info(
            f"Test F1-score (weighted): {evaluation['f1_score_weighted']:.4f}")

        return evaluation

    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts

        Args:
            texts: List of texts to predict
            batch_size: Batch size for prediction

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Create dataset
        dummy_labels = np.zeros(len(texts))  # Dummy labels for dataset
        dataset = PersianTextDataset(texts, dummy_labels, self.word_to_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                text_indices = batch['text_indices'].to(self.device)

                outputs = self.model(text_indices)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Decode predictions
        predictions_decoded = self.label_encoder.inverse_transform(
            all_predictions)

        return predictions_decoded, np.array(all_probabilities)

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / f"neural_network_{self.model_type}"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Training history
        if self.training_history['epochs']:
            self._plot_training_history(save_dir)
            saved_plots.append(
                str(save_dir / f'{self.model_type}_training_history.png'))

        # 2. Confusion matrix
        if self.evaluation_results:
            self._plot_confusion_matrix(save_dir)
            saved_plots.append(
                str(save_dir / f'{self.model_type}_confusion_matrix.png'))

            # 3. Performance metrics
            self._plot_performance_metrics(save_dir)
            saved_plots.append(
                str(save_dir / f'{self.model_type}_performance_metrics.png'))

        return saved_plots

    def _plot_training_history(self, save_dir: Path):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = self.training_history['epochs']

        # Loss plot
        axes[0].plot(epochs, self.training_history['train_loss'],
                     'b-', label='Train Loss', alpha=0.8)
        axes[0].plot(epochs, self.training_history['val_loss'],
                     'r-', label='Validation Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(
            f'{self.model_type.upper()} Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(epochs, self.training_history['train_accuracy'],
                     'b-', label='Train Accuracy', alpha=0.8)
        axes[1].plot(epochs, self.training_history['val_accuracy'],
                     'r-', label='Validation Accuracy', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(
            f'{self.model_type.upper()} Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / f'{self.model_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, save_dir: Path):
        """Plot confusion matrix"""
        cm = np.array(self.evaluation_results['confusion_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'{self.model_type.upper()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.savefig(
            save_dir / f'{self.model_type}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_metrics(self, save_dir: Path):
        """Plot performance metrics"""
        class_report = self.evaluation_results['classification_report']

        # Extract metrics for each class
        classes = [cls for cls in class_report.keys()
                   if cls not in ['accuracy', 'macro avg', 'weighted avg']]

        metrics = ['precision', 'recall', 'f1-score']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(classes))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [class_report[cls][metric] for cls in classes]
            ax.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'{self.model_type.upper()} Performance Metrics by Class')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, metric in enumerate(metrics):
            values = [class_report[cls][metric] for cls in classes]
            for j, v in enumerate(values):
                ax.text(j + i*width, v + 0.01,
                        f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(
            save_dir / f'{self.model_type}_performance_metrics.png', dpi=300, bbox_inches='tight')
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
            model_path = MODELS_DIR / "saved_models" / \
                f"{self.model_type}_model.pth"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and associated data
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config,
            'label_encoder': self.label_encoder,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'stats': self.stats,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }

        torch.save(save_data, model_path)

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

        checkpoint = torch.load(model_path, map_location=self.device)

        # Restore configuration and data
        self.model_type = checkpoint['model_type']
        self.config = checkpoint['config']
        self.label_encoder = checkpoint['label_encoder']
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']
        self.vocab_size = checkpoint['vocab_size']
        self.stats = checkpoint.get('stats', {})
        self.training_history = checkpoint.get('training_history', {})
        self.evaluation_results = checkpoint.get('evaluation_results', {})

        # Recreate and load model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.create_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.logger.info("Model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive model report"""
        self.stats['timestamp'] = datetime.now().isoformat()

        report = {
            'model_info': {
                'model_type': self.model_type,
                'timestamp': self.stats['timestamp'],
                'configuration': self.config,
                'device_used': self.stats['device_used']
            },
            'model_architecture': {
                'total_parameters': self.stats['total_parameters'],
                'trainable_parameters': self.stats['trainable_parameters'],
                'vocabulary_size': self.vocab_size,
                'model_size_mb': self.stats['model_size_mb']
            },
            'training_summary': {
                'training_time': self.stats['training_time'],
                'epochs_trained': len(self.training_history.get('epochs', [])),
                'final_train_accuracy': self.training_history['train_accuracy'][-1] if self.training_history.get('train_accuracy') else None,
                'final_val_accuracy': self.training_history['val_accuracy'][-1] if self.training_history.get('val_accuracy') else None
            },
            'evaluation_results': self.evaluation_results,
            'training_history': self.training_history
        }

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save model report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / \
                f"{self.model_type}_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Model report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of neural network classifier"""
    # Test both CNN and LSTM models
    model_types = ['lstm', 'cnn']

    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}")

        # Initialize classifier
        classifier = PersianNeuralNetworkClassifier(model_type=model_type)

        # Load data
        data_path = PROCESSED_DATA_DIR / \
            "comments_heavy_stem_processed.csv"  # Use stemmed version

        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            continue

        texts, labels = classifier.load_data(
            str(data_path),
            text_column='comment_processed',
            label_column='sentiment_label'
        )

        # Check for Word2Vec embeddings
        embedding_path = MODELS_DIR / "saved_models" / "word2vec_vectors.kv"
        embedding_path = str(
            embedding_path) if embedding_path.exists() else None

        # Train model
        training_results = classifier.train_model(
            texts, labels,
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            embedding_path=embedding_path,
            early_stopping_patience=5
        )

        # Evaluate model
        evaluation_results = classifier.evaluate_model()

        # Create visualizations
        plots = classifier.create_visualizations()

        # Save model and report
        model_path = classifier.save_model()
        report_path = classifier.save_report()

        # Display results
        print(f"\n✅ {model_type.upper()} Training Completed!")
        print(f"📊 Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        print(
            f"📊 Test F1-Score: {evaluation_results['f1_score_weighted']:.4f}")
        print(
            f"⏱️ Training Time: {training_results['training_time']:.2f} seconds")
        print(f"🧠 Parameters: {training_results['total_parameters']:,}")
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
            max_prob = np.max(prob)
            print(f"Confidence: {max_prob:.3f}")
            print("-" * 50)


if __name__ == "__main__":
    main()
