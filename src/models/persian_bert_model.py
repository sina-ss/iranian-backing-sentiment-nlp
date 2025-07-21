"""
Persian BERT Fine-tuning Model for Banking Sentiment Classification
Advanced transformer-based model with comprehensive fine-tuning and evaluation
"""

import os
import sys
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
    get_scheduler
)
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
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

# Transformer libraries

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


class PersianBertDataset(Dataset):
    """Dataset class for BERT fine-tuning"""

    def __init__(self, texts: List[str], labels: np.ndarray,
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PersianBertForSentimentClassification(nn.Module):
    """
    Persian BERT model for sentiment classification with custom head
    """

    def __init__(self, model_name: str, num_classes: int,
                 dropout: float = 0.3, freeze_bert: bool = False):
        super(PersianBertForSentimentClassification, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load BERT model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooler_output': pooled_output
        }


class PersianBertFinetuner:
    """
    Advanced Persian BERT fine-tuner for sentiment classification
    """

    def __init__(self, model_name: str = 'HooshvareLab/bert-base-parsbert-uncased',
                 config: Optional[Dict] = None):
        """
        Initialize BERT fine-tuner

        Args:
            model_name: Name of the Persian BERT model to use
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or MODEL_CONFIG.get('bert_finetuning', {})
        self.training_config = TRAINING_CONFIG

        self.setup_logging()

        # Model components
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None
        self.scheduler = None

        # Data components
        self.label_encoder = LabelEncoder()

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
            'learning_rates': [],
            'epochs': []
        }
        self.evaluation_results = {}

        # Statistics
        self.stats = {
            'model_name': model_name,
            'training_time': 0.0,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'model_size_mb': 0.0,
            'device_used': str(self.device),
            'max_gpu_memory_mb': 0.0,
            'timestamp': None
        }

        # Available Persian BERT models
        self.available_models = {
            'parsbert-base': 'HooshvareLab/bert-base-parsbert-uncased',
            'parsbert-large': 'HooshvareLab/bert-large-parsbert-uncased',
            'distilbert-persian': 'HooshvareLab/distilbert-base-parsbert-uncased',
            'electra-persian': 'HooshvareLab/electra-base-parsbert-uncased',
            'albert-persian': 'HooshvareLab/albert-base-parsbert-uncased'
        }

        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def setup_logging(self):
        """Setup logging for BERT fine-tuning"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "bert_finetuning.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model_and_tokenizer(self, model_name: Optional[str] = None) -> Tuple[nn.Module, Any]:
        """
        Load Persian BERT model and tokenizer

        Args:
            model_name: Optional model name override

        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name:
            self.model_name = model_name

        self.logger.info(f"Loading BERT model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )

            # Add special tokens if needed
            special_tokens = ['<PAD>', '<UNK>']
            new_tokens = [
                token for token in special_tokens if token not in self.tokenizer.vocab]
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                self.logger.info(f"Added {len(new_tokens)} special tokens")

            self.logger.info(
                f"Tokenizer loaded. Vocabulary size: {len(self.tokenizer)}")

            return self.tokenizer

        except Exception as e:
            self.logger.error(f"Error loading model and tokenizer: {e}")
            raise

    def load_data(self, data_path: str, text_column: str = 'comment_cleaned',
                  label_column: str = 'sentiment_label') -> Tuple[List[str], np.ndarray]:
        """
        Load and prepare data for fine-tuning

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

            # Try alternative column names if not found
            if text_column not in df.columns:
                alternative_columns = ['comment_processed', 'comment', 'text']
                for alt_col in alternative_columns:
                    if alt_col in df.columns:
                        text_column = alt_col
                        self.logger.info(
                            f"Using column '{text_column}' for text")
                        break
                else:
                    raise ValueError(f"No suitable text column found")

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
            self.logger.info(f"Classes: {list(self.label_encoder.classes_)}")

            return texts, labels_encoded

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_data_loaders(self, texts: List[str], labels: np.ndarray,
                            batch_size: int = 16, max_length: int = 128,
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
        train_dataset = PersianBertDataset(
            X_train, y_train, self.tokenizer, max_length)
        val_dataset = PersianBertDataset(
            X_val, y_val, self.tokenizer, max_length)
        test_dataset = PersianBertDataset(
            X_test, y_test, self.tokenizer, max_length)

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

    def create_model(self, num_classes: int, freeze_bert: bool = False,
                     dropout: float = 0.3) -> nn.Module:
        """
        Create BERT model for fine-tuning

        Args:
            num_classes: Number of output classes
            freeze_bert: Whether to freeze BERT parameters
            dropout: Dropout rate for classification head

        Returns:
            BERT model
        """
        self.logger.info(f"Creating BERT model for {num_classes} classes")

        model = PersianBertForSentimentClassification(
            model_name=self.model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=freeze_bert
        )

        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > model.bert.config.vocab_size:
            model.bert.resize_token_embeddings(len(self.tokenizer))
            self.logger.info(
                f"Resized token embeddings to {len(self.tokenizer)}")

        model = model.to(self.device)

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        self.stats['total_parameters'] = total_params
        self.stats['trainable_parameters'] = trainable_params

        self.logger.info(
            f"Model created with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        if freeze_bert:
            self.logger.info("BERT parameters are frozen")

        return model

    def setup_training(self, num_training_steps: int, learning_rate: float = 2e-5,
                       warmup_steps: Optional[int] = None, weight_decay: float = 0.01) -> Tuple[Any, Any]:
        """
        Setup optimizer and scheduler for training

        Args:
            num_training_steps: Total number of training steps
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization

        Returns:
            Tuple of (optimizer, scheduler)
        """
        if warmup_steps is None:
            warmup_steps = num_training_steps // 10  # 10% warmup

        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=1e-8
        )

        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.logger.info(
            f"Setup training with lr={learning_rate}, warmup_steps={warmup_steps}")

        return self.optimizer, self.scheduler

    def train_model(self, texts: List[str], labels: np.ndarray,
                    epochs: int = 3, batch_size: int = 16,
                    learning_rate: float = 2e-5, max_length: int = 128,
                    freeze_bert: bool = False, warmup_ratio: float = 0.1,
                    early_stopping_patience: int = 3) -> Dict:
        """
        Fine-tune BERT model

        Args:
            texts: List of texts
            labels: Label array
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            freeze_bert: Whether to freeze BERT parameters
            warmup_ratio: Ratio of steps for warmup
            early_stopping_patience: Early stopping patience

        Returns:
            Training results dictionary
        """
        start_time = datetime.now()
        self.logger.info("Starting BERT fine-tuning...")

        # Load tokenizer
        self.load_model_and_tokenizer()

        # Create data loaders
        self.create_data_loaders(texts, labels, batch_size, max_length)

        # Create model
        num_classes = len(np.unique(labels))
        self.model = self.create_model(num_classes, freeze_bert)

        # Setup training
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.setup_training(total_steps, learning_rate, warmup_steps)

        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            train_loss, train_accuracy = self._train_epoch()

            # Validation phase
            val_loss, val_accuracy = self._validate_epoch()

            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr'])
            self.training_history['epochs'].append(epoch + 1)

            # Log progress
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            self.logger.info(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Track GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                self.stats['max_gpu_memory_mb'] = max(
                    self.stats['max_gpu_memory_mb'], gpu_memory)

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_bert_model_temp.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_bert_model_temp.pth'))
        Path('best_bert_model_temp.pth').unlink(missing_ok=True)

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        self.stats['training_time'] = training_time

        training_results = {
            'epochs_trained': len(self.training_history['epochs']),
            'training_time': training_time,
            'best_val_accuracy': best_val_accuracy,
            'final_train_accuracy': self.training_history['train_accuracy'][-1],
            'final_val_accuracy': self.training_history['val_accuracy'][-1],
            'model_name': self.model_name,
            'total_parameters': self.stats['total_parameters'],
            'trainable_parameters': self.stats['trainable_parameters'],
            'max_gpu_memory_mb': self.stats['max_gpu_memory_mb']
        }

        self.logger.info(
            f"Fine-tuning completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

        return training_results

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']
            logits = outputs['logits']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })

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
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs['loss']
                logits = outputs['logits']

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
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

        self.logger.info("Evaluating BERT model performance...")

        # Evaluate on test set
        test_loss, test_accuracy = self._validate_epoch(self.test_loader)

        # Get detailed predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

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

    def predict(self, texts: List[str], batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
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
        dummy_labels = np.zeros(len(texts))  # Dummy labels
        dataset = PersianBertDataset(texts, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

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
            save_dir = FIGURES_DIR / "bert_finetuning"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Training history
        if self.training_history['epochs']:
            self._plot_training_history(save_dir)
            saved_plots.append(str(save_dir / 'bert_training_history.png'))

        # 2. Confusion matrix
        if self.evaluation_results:
            self._plot_confusion_matrix(save_dir)
            saved_plots.append(str(save_dir / 'bert_confusion_matrix.png'))

            # 3. Performance metrics
            self._plot_performance_metrics(save_dir)
            saved_plots.append(str(save_dir / 'bert_performance_metrics.png'))

            # 4. Learning rate schedule
            if self.training_history['learning_rates']:
                self._plot_learning_rate_schedule(save_dir)
                saved_plots.append(str(save_dir / 'bert_learning_rate.png'))

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
        axes[0].set_title('BERT Fine-tuning: Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(epochs, self.training_history['train_accuracy'],
                     'b-', label='Train Accuracy', alpha=0.8)
        axes[1].plot(epochs, self.training_history['val_accuracy'],
                     'r-', label='Validation Accuracy', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('BERT Fine-tuning: Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, save_dir: Path):
        """Plot confusion matrix"""
        cm = np.array(self.evaluation_results['confusion_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('BERT Fine-tuning: Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
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

        colors = ['skyblue', 'lightgreen', 'lightcoral']

        for i, metric in enumerate(metrics):
            values = [class_report[cls][metric] for cls in classes]
            bars = ax.bar(x + i*width, values, width, label=metric.title(),
                          color=colors[i], alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('BERT Fine-tuning: Performance Metrics by Class')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_performance_metrics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_learning_rate_schedule(self, save_dir: Path):
        """Plot learning rate schedule"""
        plt.figure(figsize=(10, 6))

        # Calculate steps per epoch
        steps_per_epoch = len(self.train_loader)
        total_steps = []

        for epoch, lr in zip(self.training_history['epochs'], self.training_history['learning_rates']):
            step = (epoch - 1) * steps_per_epoch + \
                steps_per_epoch  # End of epoch
            total_steps.append(step)

        plt.plot(
            total_steps, self.training_history['learning_rates'], 'b-', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('BERT Fine-tuning: Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_learning_rate.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: Optional[str] = None,
                   save_tokenizer: bool = True) -> str:
        """
        Save the fine-tuned model

        Args:
            model_path: Optional path to save the model
            save_tokenizer: Whether to save tokenizer

        Returns:
            Path to saved model directory
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if model_path is None:
            model_path = MODELS_DIR / "saved_models" / "bert_finetuned"

        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self.model.state_dict(), model_path / "pytorch_model.bin")

        # Save configuration
        config_data = {
            'model_name': self.model_name,
            'num_classes': len(self.label_encoder.classes_),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'config': self.config,
            'stats': self.stats,
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }

        with open(model_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False,
                      indent=2, default=str)

        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(model_path)

        # Save label encoder
        import pickle
        with open(model_path / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Calculate model size
        model_size = sum(
            f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        model_size_mb = model_size / (1024 * 1024)
        self.stats['model_size_mb'] = model_size_mb

        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Model size: {model_size_mb:.2f} MB")

        return str(model_path)

    def load_model(self, model_path: str):
        """
        Load a saved fine-tuned model

        Args:
            model_path: Path to saved model directory
        """
        self.logger.info(f"Loading model from {model_path}")

        model_path = Path(model_path)

        # Load configuration
        with open(model_path / "config.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        self.model_name = config_data['model_name']
        num_classes = config_data['num_classes']
        self.config = config_data.get('config', {})
        self.stats = config_data.get('stats', {})
        self.training_history = config_data.get('training_history', {})
        self.evaluation_results = config_data.get('evaluation_results', {})

        # Load label encoder
        import pickle
        with open(model_path / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Recreate and load model
        self.model = self.create_model(num_classes)
        self.model.load_state_dict(torch.load(
            model_path / "pytorch_model.bin", map_location=self.device))
        self.model.eval()

        self.logger.info("Model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive model report"""
        self.stats['timestamp'] = datetime.now().isoformat()

        report = {
            'model_info': {
                'model_name': self.model_name,
                'model_type': 'BERT Fine-tuned',
                'timestamp': self.stats['timestamp'],
                'configuration': self.config,
                'device_used': self.stats['device_used']
            },
            'model_architecture': {
                'base_model': self.model_name,
                'total_parameters': self.stats['total_parameters'],
                'trainable_parameters': self.stats['trainable_parameters'],
                'model_size_mb': self.stats['model_size_mb'],
                'max_gpu_memory_mb': self.stats['max_gpu_memory_mb']
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
            report_path = RESULTS_DIR / "reports" / "bert_finetuning_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Model report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of BERT fine-tuner"""
    # Initialize fine-tuner
    fine_tuner = PersianBertFinetuner(
        model_name='HooshvareLab/bert-base-parsbert-uncased'
    )

    # Load data (use light processed to preserve context)
    data_path = PROCESSED_DATA_DIR / "comments_light_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocessing first.")
        return

    texts, labels = fine_tuner.load_data(
        str(data_path),
        text_column='comment_cleaned',
        label_column='sentiment_label'
    )

    # Fine-tune model (limit data for demo)
    training_results = fine_tuner.train_model(
        texts[:1000],  # Limit for demo
        labels[:1000],
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        max_length=128,
        freeze_bert=False,
        early_stopping_patience=2
    )

    # Evaluate model
    evaluation_results = fine_tuner.evaluate_model()

    # Create visualizations
    plots = fine_tuner.create_visualizations()

    # Save model and report
    model_path = fine_tuner.save_model()
    report_path = fine_tuner.save_report()

    # Display results
    print(f"\n✅ BERT Fine-tuning Completed!")
    print(f"🧠 Model: {fine_tuner.model_name}")
    print(f"📊 Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"📊 Test F1-Score: {evaluation_results['f1_score_weighted']:.4f}")
    print(f"⏱️ Training Time: {training_results['training_time']:.2f} seconds")
    print(f"🔧 Parameters: {training_results['total_parameters']:,}")
    print(f"🎯 GPU Memory: {training_results['max_gpu_memory_mb']:.1f} MB")
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"📈 Visualizations: {len(plots)} plots created")

    # Test prediction
    test_texts = [
        "اپلیکیشن عالی است و سریع کار می‌کند",
        "برنامه خراب است و مشکل دارد",
        "متوسط است و قابل قبول"
    ]

    predictions, probabilities = fine_tuner.predict(test_texts)
    print(f"\n🧪 Test Predictions:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {np.max(prob):.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
