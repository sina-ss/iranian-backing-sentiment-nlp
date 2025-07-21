"""
BERT Embeddings Extractor for Persian Banking Comments
Advanced transformer-based feature extraction with Persian BERT models
"""

import os
import sys
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel,
    AutoConfig, BertTokenizer, BertModel,
    pipeline
)
import numpy as np
import pandas as pd
import pickle
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Transformer libraries

# Import project configuration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        FEATURE_CONFIG,
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
    """Dataset class for efficient BERT processing"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'text': text
        }


class PersianBertExtractor:
    """
    Advanced BERT embeddings extractor for Persian banking comments
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BERT extractor

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or FEATURE_CONFIG.get('bert', {})

        # Define device BEFORE calling setup_logging
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.setup_logging()

        # Model components
        self.tokenizer = None
        self.model = None

        # Data storage
        self.embeddings = None
        self.documents = None
        self.labels = None

        # Analysis results
        self.similarity_matrix = None
        self.clusters = None

        # Statistics
        self.stats = {
            'model_name': '',
            'documents_processed': 0,
            'embedding_dimension': 0,
            'processing_time': 0.0,
            'device_used': str(self.device),
            'memory_usage_mb': 0.0,
            'timestamp': None
        }

        # Available Persian BERT models
        self.available_models = {
            'parsbert': 'HooshvareLab/bert-base-parsbert-uncased',
            'parsbert-large': 'HooshvareLab/bert-large-parsbert-uncased',
            'distilbert-persian': 'HooshvareLab/distilbert-base-parsbert-uncased',
            'electra-persian': 'HooshvareLab/electra-base-parsbert-uncased',
            'albert-persian': 'HooshvareLab/albert-base-parsbert-uncased'
        }

    def setup_logging(self):
        """Setup logging for BERT extraction"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "bert_extraction.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def load_model(self, model_name: str = 'parsbert',
                   use_cache: bool = True) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Load Persian BERT model and tokenizer

        Args:
            model_name: Name of the model to load
            use_cache: Whether to use cached models

        Returns:
            Tuple of (tokenizer, model)
        """
        if model_name in self.available_models:
            model_path = self.available_models[model_name]
        else:
            model_path = model_name  # Assume it's a direct model path

        self.logger.info(f"Loading BERT model: {model_path}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                local_files_only=False
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                model_path,
                output_hidden_states=True,
                output_attentions=False
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Update configuration
            self.config['model_name'] = model_path
            self.stats['model_name'] = model_path

            # Get model info
            config = self.model.config
            hidden_size = config.hidden_size
            num_layers = config.num_hidden_layers

            self.logger.info(f"Model loaded successfully")
            self.logger.info(
                f"Hidden size: {hidden_size}, Layers: {num_layers}")
            self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

            return self.tokenizer, self.model

        except Exception as e:
            self.logger.error(f"Error loading BERT model: {e}")
            raise

    def load_data(self, data_path: str, text_column: str = 'comment_cleaned',
                  label_column: Optional[str] = None) -> Tuple[List[str], Optional[List[str]]]:
        """
        Load preprocessed data for BERT embedding extraction

        Args:
            data_path: Path to preprocessed CSV file
            text_column: Name of the text column
            label_column: Optional name of the label column

        Returns:
            Tuple of (documents, labels)
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            if text_column not in df.columns:
                # Try alternative column names
                alternative_columns = ['comment_processed', 'comment', 'text']
                for alt_col in alternative_columns:
                    if alt_col in df.columns:
                        text_column = alt_col
                        self.logger.info(
                            f"Using column '{text_column}' for text")
                        break
                else:
                    raise ValueError(f"No suitable text column found in data")

            # Extract documents
            self.documents = df[text_column].fillna('').astype(str).tolist()

            # Filter out empty documents
            self.documents = [doc for doc in self.documents if doc.strip()]

            # Extract labels if available
            labels = None
            if label_column and label_column in df.columns:
                # Align labels with filtered documents
                original_docs = df[text_column].fillna('').astype(str).tolist()
                labels = df[label_column].tolist()
                aligned_labels = []

                for i, doc in enumerate(original_docs):
                    if doc.strip():
                        aligned_labels.append(labels[i])

                labels = aligned_labels

            self.labels = labels

            self.logger.info(f"Loaded {len(self.documents)} documents")
            if labels:
                label_counts = pd.Series(labels).value_counts()
                self.logger.info(
                    f"Label distribution: {label_counts.to_dict()}")

            # Check text lengths
            text_lengths = [len(doc) for doc in self.documents]
            self.logger.info(f"Text length stats - Mean: {np.mean(text_lengths):.1f}, "
                             f"Max: {np.max(text_lengths)}, Min: {np.min(text_lengths)}")

            return self.documents, labels

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def extract_embeddings(self, documents: Optional[List[str]] = None,
                           batch_size: int = 16,
                           pooling_strategy: str = 'cls',
                           layer_index: int = -1) -> np.ndarray:
        """
        Extract BERT embeddings for documents

        Args:
            documents: Optional list of documents
            batch_size: Batch size for processing
            pooling_strategy: Pooling strategy ('cls', 'mean', 'max', 'mean_pooling')
            layer_index: Which layer to use (-1 for last layer)

        Returns:
            Embeddings array
        """
        if documents is None:
            if self.documents is None:
                raise ValueError("No documents loaded. Call load_data first.")
            documents = self.documents

        if self.model is None:
            self.load_model()

        start_time = datetime.now()
        self.logger.info(
            f"Extracting BERT embeddings for {len(documents)} documents")
        self.logger.info(
            f"Batch size: {batch_size}, Pooling: {pooling_strategy}, Layer: {layer_index}")

        # Create dataset and dataloader
        dataset = PersianBertDataset(
            documents,
            self.tokenizer,
            max_length=self.config.get('max_length', 128)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues with BERT
        )

        all_embeddings = []

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx % 10 == 0:
                        self.logger.info(
                            f"Processing batch {batch_idx + 1}/{len(dataloader)}")

                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Get model outputs
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    # Extract embeddings based on strategy
                    batch_embeddings = self._pool_embeddings(
                        outputs,
                        attention_mask,
                        pooling_strategy,
                        layer_index
                    )

                    all_embeddings.append(batch_embeddings.cpu().numpy())

                    # Clear GPU cache periodically
                    if batch_idx % 20 == 0:
                        torch.cuda.empty_cache()

            # Concatenate all embeddings
            self.embeddings = np.vstack(all_embeddings)

            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._calculate_statistics(processing_time)

            self.logger.info(
                f"Embedding extraction completed in {processing_time:.2f} seconds")
            self.logger.info(f"Embeddings shape: {self.embeddings.shape}")

            return self.embeddings

        except Exception as e:
            self.logger.error(f"Error during embedding extraction: {e}")
            raise

    def _pool_embeddings(self, outputs, attention_mask, pooling_strategy: str, layer_index: int) -> torch.Tensor:
        """
        Pool embeddings using specified strategy

        Args:
            outputs: Model outputs
            attention_mask: Attention mask
            pooling_strategy: Pooling strategy
            layer_index: Layer index to use

        Returns:
            Pooled embeddings
        """
        # Get hidden states from specified layer
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[layer_index]
        else:
            hidden_states = outputs.last_hidden_state

        if pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :]

        elif pooling_strategy == 'mean':
            # Mean pooling over all tokens
            embeddings = torch.mean(hidden_states, dim=1)

        elif pooling_strategy == 'max':
            # Max pooling over all tokens
            embeddings = torch.max(hidden_states, dim=1)[0]

        elif pooling_strategy == 'mean_pooling':
            # Mean pooling with attention mask consideration
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        return embeddings

    def _calculate_statistics(self, processing_time: float):
        """Calculate and store extraction statistics"""
        # Estimate memory usage
        if self.embeddings is not None:
            memory_usage = self.embeddings.nbytes / (1024 * 1024)  # MB
        else:
            memory_usage = 0.0

        self.stats.update({
            'documents_processed': len(self.documents) if self.documents else 0,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'processing_time': processing_time,
            'memory_usage_mb': memory_usage,
            'timestamp': datetime.now().isoformat()
        })

    def compare_pooling_strategies(self, sample_size: int = 100) -> Dict:
        """
        Compare different pooling strategies on a sample of documents

        Args:
            sample_size: Number of documents to sample for comparison

        Returns:
            Comparison results dictionary
        """
        if self.documents is None:
            raise ValueError("No documents loaded")

        # Sample documents
        sample_docs = self.documents[:min(sample_size, len(self.documents))]
        self.logger.info(
            f"Comparing pooling strategies on {len(sample_docs)} documents")

        strategies = ['cls', 'mean', 'max', 'mean_pooling']
        strategy_embeddings = {}

        for strategy in strategies:
            self.logger.info(f"Testing pooling strategy: {strategy}")
            embeddings = self.extract_embeddings(
                documents=sample_docs,
                batch_size=8,
                pooling_strategy=strategy
            )
            strategy_embeddings[strategy] = embeddings

        # Analyze differences
        comparison_results = {}

        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i+1:]:
                # Calculate cosine similarity between strategies
                similarities = []
                for j in range(len(sample_docs)):
                    sim = cosine_similarity(
                        strategy_embeddings[strategy1][j:j+1],
                        strategy_embeddings[strategy2][j:j+1]
                    )[0, 0]
                    similarities.append(sim)

                comparison_results[f"{strategy1}_vs_{strategy2}"] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'min_similarity': np.min(similarities),
                    'max_similarity': np.max(similarities)
                }

        return comparison_results

    def analyze_embeddings(self) -> Dict:
        """
        Analyze the extracted embeddings

        Returns:
            Analysis results dictionary
        """
        if self.embeddings is None:
            raise ValueError(
                "No embeddings available. Extract embeddings first.")

        self.logger.info("Analyzing BERT embeddings...")

        analysis = {
            'basic_stats': self._calculate_basic_stats(),
            'dimension_analysis': self._analyze_dimensions(),
            'similarity_analysis': self._analyze_similarities(),
            'clustering_analysis': self._analyze_clustering()
        }

        if self.labels:
            analysis['label_analysis'] = self._analyze_by_labels()

        return analysis

    def _calculate_basic_stats(self) -> Dict:
        """Calculate basic statistics of embeddings"""
        return {
            'shape': self.embeddings.shape,
            'mean': float(np.mean(self.embeddings)),
            'std': float(np.std(self.embeddings)),
            'min': float(np.min(self.embeddings)),
            'max': float(np.max(self.embeddings)),
            'norm_mean': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'norm_std': float(np.std(np.linalg.norm(self.embeddings, axis=1)))
        }

    def _analyze_dimensions(self) -> Dict:
        """Analyze embedding dimensions"""
        # PCA analysis
        pca = PCA()
        pca.fit(self.embeddings)

        # Find number of components for 95% variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1

        return {
            'total_dimensions': self.embeddings.shape[1],
            'effective_dimensions_95': int(n_components_95),
            'explained_variance_top10': pca.explained_variance_ratio_[:10].tolist(),
            'cumulative_variance_top10': cumsum_var[:10].tolist()
        }

    def _analyze_similarities(self, sample_size: int = 200) -> Dict:
        """Analyze pairwise similarities"""
        # Use sample for large datasets
        if len(self.embeddings) > sample_size:
            indices = np.random.choice(
                len(self.embeddings), sample_size, replace=False)
            sample_embeddings = self.embeddings[indices]
        else:
            sample_embeddings = self.embeddings

        # Calculate pairwise similarities
        similarities = cosine_similarity(sample_embeddings)

        # Remove diagonal (self-similarity)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        similarity_values = similarities[mask]

        return {
            'mean_similarity': float(np.mean(similarity_values)),
            'std_similarity': float(np.std(similarity_values)),
            'min_similarity': float(np.min(similarity_values)),
            'max_similarity': float(np.max(similarity_values)),
            'sample_size': len(sample_embeddings)
        }

    def _analyze_clustering(self, n_clusters: int = 5) -> Dict:
        """Analyze clustering quality"""
        if len(self.embeddings) < n_clusters:
            return {'error': 'Insufficient data for clustering'}

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.embeddings)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(self.embeddings, clusters)

        # Analyze cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        self.clusters = clusters

        return {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'cluster_sizes': cluster_sizes,
            'inertia': float(kmeans.inertia_)
        }

    def _analyze_by_labels(self) -> Dict:
        """Analyze embeddings by sentiment labels"""
        if not self.labels:
            return {}

        label_analysis = {}
        unique_labels = list(set(self.labels))

        for label in unique_labels:
            # Get embeddings for this label
            label_indices = [i for i, l in enumerate(
                self.labels) if l == label]
            label_embeddings = self.embeddings[label_indices]

            # Calculate statistics
            label_analysis[label] = {
                'count': len(label_indices),
                'mean_norm': float(np.mean(np.linalg.norm(label_embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(label_embeddings, axis=1))),
                'centroid': np.mean(label_embeddings, axis=0).tolist()
            }

        # Calculate inter-label distances
        centroids = {label: np.array(data['centroid'])
                     for label, data in label_analysis.items()}
        inter_distances = {}

        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                distance = np.linalg.norm(
                    centroids[label1] - centroids[label2])
                inter_distances[f"{label1}_vs_{label2}"] = float(distance)

        label_analysis['inter_label_distances'] = inter_distances

        return label_analysis

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "bert"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. PCA visualization
        self._plot_pca_analysis(save_dir)
        saved_plots.append(str(save_dir / 'bert_pca_analysis.png'))

        # 2. t-SNE visualization
        self._plot_tsne_visualization(save_dir)
        saved_plots.append(str(save_dir / 'bert_tsne_visualization.png'))

        # 3. Embedding statistics
        self._plot_embedding_statistics(save_dir)
        saved_plots.append(str(save_dir / 'bert_embedding_statistics.png'))

        # 4. Label-based analysis (if available)
        if self.labels:
            self._plot_label_analysis(save_dir)
            saved_plots.append(str(save_dir / 'bert_label_analysis.png'))

        # 5. Clustering visualization (if available)
        if self.clusters is not None:
            self._plot_clustering_analysis(save_dir)
            saved_plots.append(str(save_dir / 'bert_clustering_analysis.png'))

        return saved_plots

    def _plot_pca_analysis(self, save_dir: Path):
        """Plot PCA analysis"""
        pca = PCA()
        pca_embeddings = pca.fit_transform(self.embeddings)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Explained variance
        axes[0, 0].plot(
            range(1, 21), pca.explained_variance_ratio_[:20], 'bo-')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].grid(True)

        # Cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0, 1].plot(range(1, 21), cumsum_var[:20], 'ro-')
        axes[0, 1].axhline(y=0.95, color='k', linestyle='--',
                           alpha=0.7, label='95% variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 2D PCA projection
        colors = self._get_point_colors()
        scatter = axes[1, 0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                                     c=colors, alpha=0.6, s=20)
        axes[1, 0].set_xlabel(
            f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 0].set_ylabel(
            f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1, 0].set_title('2D PCA Projection')

        # 3D view (PC1, PC2, PC3)
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2],
                   c=colors, alpha=0.6, s=20)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        ax.set_title('3D PCA Projection')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_pca_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tsne_visualization(self, save_dir: Path):
        """Plot t-SNE visualization"""
        # Use sample for large datasets
        n_samples = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(
            len(self.embeddings), n_samples, replace=False)
        sample_embeddings = self.embeddings[sample_indices]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, n_samples-1))
        tsne_embeddings = tsne.fit_transform(sample_embeddings)

        plt.figure(figsize=(12, 10))

        # Get colors for sample
        if self.labels:
            sample_labels = [self.labels[i] for i in sample_indices]
            colors = ['red' if l == 'negative' else 'gray' if l == 'neutral' else 'green'
                      for l in sample_labels]

            # Create legend
            unique_labels = list(set(sample_labels))
            for label in unique_labels:
                label_indices = [i for i, l in enumerate(
                    sample_labels) if l == label]
                label_colors = [colors[i] for i in label_indices]
                plt.scatter(tsne_embeddings[label_indices, 0],
                            tsne_embeddings[label_indices, 1],
                            c=label_colors[0], alpha=0.6, s=30, label=label)
            plt.legend()
        else:
            plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                        alpha=0.6, s=30, c='blue')

        plt.title(
            f't-SNE Visualization of BERT Embeddings\n(Sample of {n_samples} documents)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_tsne_visualization.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_embedding_statistics(self, save_dir: Path):
        """Plot embedding statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Embedding norms distribution
        norms = np.linalg.norm(self.embeddings, axis=1)
        axes[0, 0].hist(norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Embedding Norms')
        axes[0, 0].axvline(np.mean(norms), color='red',
                           linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
        axes[0, 0].legend()

        # Dimension-wise statistics
        dim_means = np.mean(self.embeddings, axis=0)
        dim_stds = np.std(self.embeddings, axis=0)

        axes[0, 1].plot(dim_means[:100], alpha=0.7, label='Mean')
        axes[0, 1].plot(dim_stds[:100], alpha=0.7, label='Std')
        axes[0, 1].set_xlabel('Dimension Index')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Dimension Statistics (First 100 dims)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Correlation matrix (sample of dimensions)
        sample_dims = min(50, self.embeddings.shape[1])
        corr_matrix = np.corrcoef(self.embeddings[:, :sample_dims].T)

        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title(
            f'Dimension Correlation Matrix\n(First {sample_dims} dimensions)')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 0])

        # Embedding values distribution
        axes[1, 1].hist(self.embeddings.flatten(), bins=100,
                        alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of All Embedding Values')
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_embedding_statistics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_label_analysis(self, save_dir: Path):
        """Plot label-based analysis"""
        if not self.labels:
            return

        unique_labels = list(set(self.labels))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Embedding norms by label
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(
                self.labels) if l == label]
            label_embeddings = self.embeddings[label_indices]
            norms = np.linalg.norm(label_embeddings, axis=1)

            axes[0, 0].hist(norms, bins=30, alpha=0.6, label=label)

        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Embedding Norms by Label')
        axes[0, 0].legend()

        # Average embeddings by label
        label_means = {}
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(
                self.labels) if l == label]
            label_embeddings = self.embeddings[label_indices]
            label_means[label] = np.mean(label_embeddings, axis=0)

        # Plot first 100 dimensions
        for label, mean_emb in label_means.items():
            axes[0, 1].plot(mean_emb[:100], alpha=0.7, label=label)

        axes[0, 1].set_xlabel('Dimension Index')
        axes[0, 1].set_ylabel('Average Value')
        axes[0, 1].set_title('Average Embeddings by Label (First 100 dims)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Inter-label distances
        if len(unique_labels) >= 2:
            distances = []
            label_pairs = []

            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    dist = np.linalg.norm(
                        label_means[label1] - label_means[label2])
                    distances.append(dist)
                    label_pairs.append(f"{label1}-{label2}")

            axes[1, 0].bar(range(len(distances)), distances)
            axes[1, 0].set_xticks(range(len(distances)))
            axes[1, 0].set_xticklabels(label_pairs, rotation=45)
            axes[1, 0].set_ylabel('Euclidean Distance')
            axes[1, 0].set_title('Inter-Label Centroid Distances')

        # Label distribution
        label_counts = pd.Series(self.labels).value_counts()
        axes[1, 1].pie(label_counts.values,
                       labels=label_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Label Distribution')

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_label_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_clustering_analysis(self, save_dir: Path):
        """Plot clustering analysis"""
        if self.clusters is None:
            return

        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(self.embeddings)

        plt.figure(figsize=(12, 8))

        # Plot clusters
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

        for cluster_id, color in zip(unique_clusters, colors):
            cluster_mask = self.clusters == cluster_id
            plt.scatter(pca_embeddings[cluster_mask, 0],
                        pca_embeddings[cluster_mask, 1],
                        c=[color], alpha=0.6, s=30, label=f'Cluster {cluster_id}')

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('BERT Embeddings Clustering (K-means)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_dir / 'bert_clustering_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _get_point_colors(self) -> List:
        """Get colors for plotting points based on labels"""
        if self.labels:
            color_map = {'negative': 'red',
                         'neutral': 'gray', 'positive': 'green'}
            return [color_map.get(label, 'blue') for label in self.labels]
        else:
            return ['blue'] * len(self.embeddings)

    def save_embeddings(self, embeddings_path: Optional[str] = None) -> str:
        """
        Save extracted embeddings

        Args:
            embeddings_path: Optional path to save embeddings

        Returns:
            Path to saved embeddings
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save")

        if embeddings_path is None:
            embeddings_path = MODELS_DIR / "saved_models" / "bert_embeddings.npz"

        embeddings_path = Path(embeddings_path)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings and metadata
        np.savez_compressed(
            embeddings_path,
            embeddings=self.embeddings,
            labels=np.array(self.labels) if self.labels else None,
            documents=np.array(self.documents) if self.documents else None
        )

        # Save metadata
        metadata_path = embeddings_path.parent / \
            f"{embeddings_path.stem}_metadata.json"
        metadata = {
            'config': self.config,
            'stats': self.stats,
            'embedding_shape': self.embeddings.shape,
            'model_name': self.stats.get('model_name', ''),
            'extraction_timestamp': self.stats.get('timestamp')
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"BERT embeddings saved to {embeddings_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")

        return str(embeddings_path)

    def load_embeddings(self, embeddings_path: str):
        """
        Load saved embeddings

        Args:
            embeddings_path: Path to saved embeddings
        """
        self.logger.info(f"Loading BERT embeddings from {embeddings_path}")

        embeddings_path = Path(embeddings_path)

        # Load embeddings
        data = np.load(embeddings_path)
        self.embeddings = data['embeddings']

        if 'labels' in data and data['labels'] is not None:
            self.labels = data['labels'].tolist()

        if 'documents' in data and data['documents'] is not None:
            self.documents = data['documents'].tolist()

        # Load metadata if available
        metadata_path = embeddings_path.parent / \
            f"{embeddings_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.config = metadata.get('config', {})
            self.stats = metadata.get('stats', {})

        self.logger.info("BERT embeddings loaded successfully")
        self.logger.info(f"Embeddings shape: {self.embeddings.shape}")

    def generate_report(self) -> Dict:
        """Generate comprehensive BERT analysis report"""
        if self.embeddings is None:
            raise ValueError("No embeddings available for analysis")

        analysis = self.analyze_embeddings()

        report = {
            'extraction_summary': {
                'timestamp': self.stats.get('timestamp'),
                'model_name': self.stats.get('model_name'),
                'documents_processed': self.stats.get('documents_processed'),
                'embedding_dimension': self.stats.get('embedding_dimension'),
                'processing_time_seconds': self.stats.get('processing_time'),
                'memory_usage_mb': self.stats.get('memory_usage_mb'),
                'device_used': self.stats.get('device_used')
            },
            'configuration': self.config,
            'embedding_analysis': analysis,
            'model_info': {
                'available_models': list(self.available_models.keys()),
                'used_model': self.stats.get('model_name', 'unknown')
            }
        }

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save analysis report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / "bert_analysis_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"BERT analysis report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of BERT extractor"""
    # Initialize extractor
    extractor = PersianBertExtractor()

    # Load Persian BERT model
    tokenizer, model = extractor.load_model('parsbert')

    # Load data (use light processed version to preserve context)
    data_path = PROCESSED_DATA_DIR / "comments_light_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocessing first.")
        return

    documents, labels = extractor.load_data(
        str(data_path),
        text_column='comment_cleaned',
        label_column='sentiment_label'
    )

    # Extract BERT embeddings
    embeddings = extractor.extract_embeddings(
        documents=documents[:100],  # Limit for demo
        batch_size=8,
        pooling_strategy='cls'
    )

    # Analyze embeddings
    analysis = extractor.analyze_embeddings()

    # Create visualizations
    plots = extractor.create_visualizations()

    # Save embeddings and report
    embeddings_path = extractor.save_embeddings()
    report_path = extractor.save_report()

    # Display results
    print(f"\n✅ BERT Feature Extraction Completed!")
    print(f"🧠 Model: {extractor.stats['model_name']}")
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"⚡ Device used: {extractor.stats['device_used']}")
    print(f"💾 Embeddings saved to: {embeddings_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"📈 Visualizations: {len(plots)} plots created")


if __name__ == "__main__":
    main()
