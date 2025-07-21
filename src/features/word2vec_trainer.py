"""
Word2Vec Trainer for Persian Banking Comments
Advanced word embedding training with visualization and analysis capabilities
"""

import os
import sys
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import pickle
import json
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

# Word2Vec and NLP libraries

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


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization

    Args:
        obj: Object to convert

    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys AND values to handle numpy types in keys
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class TrainingCallback(CallbackAny2Vec):
    """Callback to track training progress"""

    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        self.epoch += 1
        if self.epoch % 10 == 0:
            print(f"Epoch {self.epoch}, Loss: {loss}")


class PersianWord2VecTrainer:
    """
    Advanced Word2Vec trainer optimized for Persian banking comments
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Word2Vec trainer

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or FEATURE_CONFIG.get('word2vec', {})
        self.setup_logging()

        # Model components
        self.model = None
        self.word_vectors = None
        self.vocabulary = None

        # Training data
        self.sentences = None
        self.documents = None
        self.labels = None

        # Analysis results
        self.word_similarities = {}
        self.word_clusters = {}
        self.evaluation_metrics = {}

        # Statistics
        self.stats = {
            'vocabulary_size': 0,
            'total_words': 0,
            'training_time': 0.0,
            'model_size_mb': 0.0,
            'epochs_trained': 0,
            'training_losses': [],
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for Word2Vec training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "word2vec_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: str, text_column: str = 'comment_processed',
                  label_column: Optional[str] = None) -> Tuple[List[List[str]], Optional[List[str]]]:
        """
        Load preprocessed data for Word2Vec training

        Args:
            data_path: Path to preprocessed CSV file
            text_column: Name of the text column
            label_column: Optional name of the label column

        Returns:
            Tuple of (sentences, labels)
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            if text_column not in df.columns:
                raise ValueError(
                    f"Text column '{text_column}' not found in data")

            # Convert documents to sentences (list of words)
            self.documents = df[text_column].fillna('').astype(str).tolist()
            self.sentences = [doc.split()
                              for doc in self.documents if doc.strip()]

            # Filter out empty sentences
            self.sentences = [sent for sent in self.sentences if len(sent) > 1]

            # Extract labels if available
            labels = None
            if label_column and label_column in df.columns:
                labels = df[label_column].tolist()
                # Align labels with filtered sentences
                labels = [labels[i] for i, doc in enumerate(
                    self.documents) if doc.strip() and len(doc.split()) > 1]

            self.labels = labels

            # Calculate statistics
            total_words = sum(len(sent) for sent in self.sentences)
            unique_words = len(
                set(word for sent in self.sentences for word in sent))

            self.logger.info(f"Loaded {len(self.sentences)} sentences")
            self.logger.info(
                f"Total words: {total_words}, Unique words: {unique_words}")
            self.logger.info(
                f"Average sentence length: {total_words / len(self.sentences):.2f}")

            return self.sentences, labels

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_model(self, custom_config: Optional[Dict] = None) -> Word2Vec:
        """
        Create Word2Vec model with optimized configuration

        Args:
            custom_config: Optional custom configuration

        Returns:
            Word2Vec model
        """
        config = custom_config or self.config

        # Determine number of workers
        workers = config.get('workers', min(multiprocessing.cpu_count(), 4))

        model = Word2Vec(
            vector_size=config.get('vector_size', 200),
            window=config.get('window', 5),
            min_count=config.get('min_count', 2),
            workers=workers,
            epochs=config.get('epochs', 100),
            sg=config.get('sg', 1),  # 1 for skip-gram, 0 for CBOW
            hs=config.get('hs', 0),  # 1 for hierarchical softmax
            negative=config.get('negative', 5),  # negative sampling
            alpha=config.get('alpha', 0.025),  # learning rate
            min_alpha=config.get('min_alpha', 0.0001),
            seed=config.get('seed', 42),
            compute_loss=True  # Enable loss computation
        )

        self.logger.info(f"Created Word2Vec model with config: {config}")
        return model

    def train_model(self, sentences: Optional[List[List[str]]] = None,
                    save_progress: bool = True) -> Word2Vec:
        """
        Train Word2Vec model on the sentences

        Args:
            sentences: Optional sentences to train on
            save_progress: Whether to save training progress

        Returns:
            Trained Word2Vec model
        """
        if sentences is None:
            if self.sentences is None:
                raise ValueError("No sentences loaded. Call load_data first.")
            sentences = self.sentences

        start_time = datetime.now()
        self.logger.info("Starting Word2Vec training...")

        # Create model
        self.model = self.create_model()

        # Set up training callback
        callback = TrainingCallback()

        try:
            # Build vocabulary
            self.logger.info("Building vocabulary...")
            self.model.build_vocab(sentences, progress_per=10000)
            vocab_size = len(self.model.wv.key_to_index)
            self.logger.info(f"Vocabulary built with {vocab_size} words")

            # Train model
            self.logger.info("Training Word2Vec model...")
            self.model.train(
                sentences,
                total_examples=len(sentences),
                epochs=self.config.get('epochs', 100),
                callbacks=[callback] if save_progress else None
            )

            # Extract word vectors
            self.word_vectors = self.model.wv
            self.vocabulary = list(self.word_vectors.key_to_index.keys())

            # Calculate training statistics
            training_time = (datetime.now() - start_time).total_seconds()
            self._calculate_statistics(training_time, callback.losses)

            self.logger.info(
                f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Final vocabulary size: {len(self.vocabulary)}")

            return self.model

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def _calculate_statistics(self, training_time: float, losses: List[float]):
        """Calculate and store training statistics"""
        model_size = 0
        if self.model:
            # Estimate model size (approximate)
            vocab_size = len(self.vocabulary)
            vector_size = self.config.get('vector_size', 200)
            model_size = (vocab_size * vector_size * 4) / (1024 * 1024)  # MB

        self.stats.update({
            'vocabulary_size': len(self.vocabulary) if self.vocabulary else 0,
            'total_words': sum(len(sent) for sent in self.sentences) if self.sentences else 0,
            'training_time': training_time,
            'model_size_mb': model_size,
            'epochs_trained': self.config.get('epochs', 100),
            'training_losses': losses,
            'timestamp': datetime.now().isoformat()
        })

    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained Word2Vec model

        Returns:
            Dictionary with evaluation metrics
        """
        if self.word_vectors is None:
            raise ValueError("No trained model available")

        self.logger.info("Evaluating Word2Vec model...")

        evaluation = {
            'vocabulary_coverage': self._calculate_vocabulary_coverage(),
            'word_similarities': self._calculate_word_similarities(),
            'analogy_accuracy': self._test_analogies(),
            'clustering_quality': self._evaluate_clustering(),
            'semantic_coherence': self._test_semantic_coherence()
        }

        self.evaluation_metrics = evaluation
        return evaluation

    def _calculate_vocabulary_coverage(self) -> Dict:
        """Calculate vocabulary coverage statistics"""
        if not self.sentences:
            return {}

        all_words = [word for sent in self.sentences for word in sent]
        unique_words = set(all_words)

        coverage = {
            'total_unique_words': len(unique_words),
            'words_in_model': len(self.vocabulary),
            'coverage_ratio': len(self.vocabulary) / len(unique_words),
            'most_frequent_words': pd.Series(all_words).value_counts().head(20).to_dict()
        }

        return coverage

    def _calculate_word_similarities(self) -> Dict:
        """Calculate similarities for important banking terms"""
        banking_terms = [
            'بانک', 'اپ', 'اپلیکیشن', 'کارت', 'پول', 'حساب', 'انتقال',
            'رمز', 'پسورد', 'امنیت', 'سریع', 'راحت', 'خوب', 'بد', 'مشکل'
        ]

        similarities = {}

        for term in banking_terms:
            if term in self.word_vectors:
                try:
                    similar_words = self.word_vectors.most_similar(
                        term, topn=10)
                    similarities[term] = similar_words
                except:
                    continue

        return similarities

    def _test_analogies(self) -> Dict:
        """Test word analogies for Persian banking context"""
        analogy_tests = [
            # Format: [word1, word2, word3] -> word4
            # word1 is to word2 as word3 is to word4
            (['بانک', 'پول', 'اپ'], 'سرویس'),
            (['سریع', 'کند', 'خوب'], 'بد'),
            (['ورود', 'خروج', 'باز'], 'بسته')
        ]

        correct_analogies = 0
        total_analogies = 0

        for analogy, expected in analogy_tests:
            if all(word in self.word_vectors for word in analogy + [expected]):
                try:
                    result = self.word_vectors.most_similar(
                        positive=[analogy[1], analogy[2]],
                        negative=[analogy[0]],
                        topn=5
                    )

                    # Check if expected word is in top 5 results
                    top_words = [word for word, _ in result]
                    if expected in top_words:
                        correct_analogies += 1

                    total_analogies += 1
                except:
                    continue

        accuracy = correct_analogies / total_analogies if total_analogies > 0 else 0
        return {
            'correct_analogies': correct_analogies,
            'total_analogies': total_analogies,
            'accuracy': accuracy
        }

    def _evaluate_clustering(self) -> Dict:
        """Evaluate word clustering quality"""
        if len(self.vocabulary) < 50:
            return {'error': 'Insufficient vocabulary for clustering'}

        # Sample words for clustering
        sample_words = self.vocabulary[:min(200, len(self.vocabulary))]
        vectors = np.array([self.word_vectors[word] for word in sample_words])

        # Perform K-means clustering
        n_clusters = min(10, len(sample_words) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)

        # Analyze clusters
        cluster_words = {}
        for i, word in enumerate(sample_words):
            cluster_id = clusters[i]
            if cluster_id not in cluster_words:
                cluster_words[cluster_id] = []
            cluster_words[cluster_id].append(word)

        # Calculate silhouette score approximation
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(vectors, clusters)

        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'cluster_sizes': [len(words) for words in cluster_words.values()],
            # Show top 5 words per cluster
            'sample_clusters': {k: v[:5] for k, v in cluster_words.items()}
        }

    def _test_semantic_coherence(self) -> Dict:
        """Test semantic coherence of embeddings"""
        # Define semantic groups for banking domain
        semantic_groups = {
            'positive_sentiment': ['خوب', 'عالی', 'راحت', 'سریع', 'بهترین'],
            'negative_sentiment': ['بد', 'مشکل', 'کند', 'خراب', 'ضعیف'],
            'banking_services': ['کارت', 'حساب', 'انتقال', 'واریز', 'پرداخت'],
            'technology': ['اپ', 'اپلیکیشن', 'موبایل', 'سیستم', 'نرم‌افزار']
        }

        coherence_scores = {}

        for group_name, words in semantic_groups.items():
            # Filter words that exist in vocabulary
            valid_words = [word for word in words if word in self.word_vectors]

            if len(valid_words) < 2:
                continue

            # Calculate average pairwise similarity within group
            similarities = []
            for i, word1 in enumerate(valid_words):
                for word2 in valid_words[i+1:]:
                    try:
                        sim = self.word_vectors.similarity(word1, word2)
                        similarities.append(sim)
                    except:
                        continue

            if similarities:
                coherence_scores[group_name] = {
                    'avg_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'words_tested': valid_words
                }

        return coherence_scores

    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a specific word

        Args:
            word: Word to get embedding for

        Returns:
            Embedding vector or None if word not in vocabulary
        """
        if self.word_vectors and word in self.word_vectors:
            return self.word_vectors[word]
        return None

    def get_document_embedding(self, document: Union[str, List[str]],
                               method: str = 'average') -> Optional[np.ndarray]:
        """
        Get document embedding by aggregating word embeddings

        Args:
            document: Document as string or list of words
            method: Aggregation method ('average', 'sum', 'max')

        Returns:
            Document embedding vector
        """
        if self.word_vectors is None:
            return None

        if isinstance(document, str):
            words = document.split()
        else:
            words = document

        # Get embeddings for words in vocabulary
        embeddings = []
        for word in words:
            if word in self.word_vectors:
                embeddings.append(self.word_vectors[word])

        if not embeddings:
            return None

        embeddings = np.array(embeddings)

        if method == 'average':
            return np.mean(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def find_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to given word

        Args:
            word: Input word
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if self.word_vectors and word in self.word_vectors:
            return self.word_vectors.most_similar(word, topn=topn)
        return []

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "word2vec"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Training progress plot
        if self.stats.get('training_losses'):
            self._plot_training_progress(save_dir)
            saved_plots.append(
                str(save_dir / 'word2vec_training_progress.png'))

        # 2. Word similarity visualization
        self._plot_word_similarities(save_dir)
        saved_plots.append(str(save_dir / 'word2vec_similarities.png'))

        # 3. t-SNE visualization
        self._plot_tsne_visualization(save_dir)
        saved_plots.append(str(save_dir / 'word2vec_tsne.png'))

        # 4. PCA visualization
        self._plot_pca_visualization(save_dir)
        saved_plots.append(str(save_dir / 'word2vec_pca.png'))

        # 5. Semantic groups analysis
        if self.evaluation_metrics:
            self._plot_semantic_analysis(save_dir)
            saved_plots.append(
                str(save_dir / 'word2vec_semantic_analysis.png'))

        return saved_plots

    def _plot_training_progress(self, save_dir: Path):
        """Plot training progress"""
        losses = self.stats.get('training_losses', [])
        if not losses:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Word2Vec Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'word2vec_training_progress.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_word_similarities(self, save_dir: Path):
        """Plot word similarities for banking terms"""
        if not self.word_similarities:
            similarities = self._calculate_word_similarities()
        else:
            similarities = self.word_similarities

        if not similarities:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        terms = list(similarities.keys())[:4]  # Show top 4 terms

        for i, term in enumerate(terms):
            if i < len(axes):
                ax = axes[i]
                similar_words = similarities[term]

                words = [word for word, _ in similar_words[:10]]
                scores = [score for _, score in similar_words[:10]]

                ax.barh(range(len(words)), scores)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.set_xlabel('Cosine Similarity')
                ax.set_title(f'Most Similar Words to "{term}"')
                ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_dir / 'word2vec_similarities.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tsne_visualization(self, save_dir: Path, n_words: int = 100):
        """Create t-SNE visualization of word embeddings"""
        if len(self.vocabulary) < n_words:
            n_words = len(self.vocabulary)

        # Select words for visualization
        sample_words = self.vocabulary[:n_words]
        embeddings = np.array([self.word_vectors[word]
                              for word in sample_words])

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, n_words-1))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create plot
        plt.figure(figsize=(12, 10))

        # Color points by semantic groups if available
        colors = self._assign_semantic_colors(sample_words)

        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=colors, alpha=0.7, s=50)

        # Add labels for some important words
        important_words = ['بانک', 'اپ', 'کارت',
                           'پول', 'خوب', 'بد', 'سریع', 'کند']
        for word in important_words:
            if word in sample_words:
                idx = sample_words.index(word)
                plt.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=10)

        plt.title('t-SNE Visualization of Word Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.tight_layout()
        plt.savefig(save_dir / 'word2vec_tsne.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pca_visualization(self, save_dir: Path, n_words: int = 100):
        """Create PCA visualization of word embeddings"""
        if len(self.vocabulary) < n_words:
            n_words = len(self.vocabulary)

        sample_words = self.vocabulary[:n_words]
        embeddings = np.array([self.word_vectors[word]
                              for word in sample_words])

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))

        colors = self._assign_semantic_colors(sample_words)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                    c=colors, alpha=0.7, s=50)

        # Add labels for important words
        important_words = ['بانک', 'اپ', 'کارت',
                           'پول', 'خوب', 'بد', 'سریع', 'کند']
        for word in important_words:
            if word in sample_words:
                idx = sample_words.index(word)
                plt.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=10)

        plt.title(f'PCA Visualization of Word Embeddings\n'
                  f'Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, '
                  f'PC2={pca.explained_variance_ratio_[1]:.2%}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

        plt.tight_layout()
        plt.savefig(save_dir / 'word2vec_pca.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _assign_semantic_colors(self, words: List[str]) -> List[str]:
        """Assign colors based on semantic groups"""
        semantic_groups = {
            'positive': ['خوب', 'عالی', 'راحت', 'سریع', 'بهترین', 'فوق‌العاده'],
            'negative': ['بد', 'مشکل', 'کند', 'خراب', 'ضعیف'],
            'banking': ['بانک', 'کارت', 'حساب', 'انتقال', 'واریز', 'پرداخت', 'پول'],
            'technology': ['اپ', 'اپلیکیشن', 'موبایل', 'سیستم', 'نرم‌افزار']
        }

        color_map = {
            'positive': 'green',
            'negative': 'red',
            'banking': 'blue',
            'technology': 'orange',
            'other': 'gray'
        }

        colors = []
        for word in words:
            assigned = False
            for group, group_words in semantic_groups.items():
                if word in group_words:
                    colors.append(color_map[group])
                    assigned = True
                    break
            if not assigned:
                colors.append(color_map['other'])

        return colors

    def _plot_semantic_analysis(self, save_dir: Path):
        """Plot semantic coherence analysis"""
        if 'semantic_coherence' not in self.evaluation_metrics:
            return

        coherence = self.evaluation_metrics['semantic_coherence']

        if not coherence:
            return

        groups = list(coherence.keys())
        similarities = [coherence[group]['avg_similarity'] for group in groups]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, similarities, color=[
                       'green', 'red', 'blue', 'orange'][:len(groups)])

        plt.title('Semantic Coherence by Word Groups')
        plt.xlabel('Semantic Groups')
        plt.ylabel('Average Intra-group Similarity')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, sim in zip(bars, similarities):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{sim:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_dir / 'word2vec_semantic_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: Optional[str] = None,
                   save_full_model: bool = False) -> str:
        """
        Save the trained Word2Vec model

        Args:
            model_path: Optional path to save the model
            save_full_model: Whether to save full model or just vectors

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if model_path is None:
            if save_full_model:
                model_path = MODELS_DIR / "saved_models" / "word2vec_model.bin"
            else:
                model_path = MODELS_DIR / "saved_models" / "word2vec_vectors.kv"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if save_full_model:
            # Save full model (can continue training)
            self.model.save(str(model_path))
        else:
            # Save only word vectors (smaller file, faster loading)
            self.word_vectors.save(str(model_path))

        # Save metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        metadata = {
            'config': self.config,
            'stats': self.stats,
            'evaluation_metrics': self.evaluation_metrics,
            'vocabulary_size': len(self.vocabulary),
            'model_type': 'full' if save_full_model else 'vectors_only'
        }

        # Convert numpy types to Python types for JSON serialization
        metadata = convert_numpy_types(metadata)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False,
                      indent=2, default=str)

        self.logger.info(f"Word2Vec model saved to {model_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")

        return str(model_path)

    def load_model(self, model_path: str, load_full_model: bool = False):
        """
        Load a saved Word2Vec model

        Args:
            model_path: Path to saved model
            load_full_model: Whether to load full model or just vectors
        """
        self.logger.info(f"Loading Word2Vec model from {model_path}")

        model_path = Path(model_path)

        if load_full_model:
            self.model = Word2Vec.load(str(model_path))
            self.word_vectors = self.model.wv
        else:
            self.word_vectors = KeyedVectors.load(str(model_path))

        self.vocabulary = list(self.word_vectors.key_to_index.keys())

        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.config = metadata.get('config', {})
            self.stats = metadata.get('stats', {})
            self.evaluation_metrics = metadata.get('evaluation_metrics', {})

        self.logger.info("Word2Vec model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive Word2Vec analysis report"""
        report = {
            'training_summary': {
                'timestamp': self.stats.get('timestamp'),
                'vocabulary_size': self.stats.get('vocabulary_size'),
                'total_words': self.stats.get('total_words'),
                'training_time_seconds': self.stats.get('training_time'),
                'model_size_mb': self.stats.get('model_size_mb'),
                'epochs_trained': self.stats.get('epochs_trained')
            },
            'configuration': self.config,
            'evaluation_metrics': self.evaluation_metrics,
            'model_quality': {
                'vocabulary_coverage': self.evaluation_metrics.get('vocabulary_coverage', {}),
                'analogy_accuracy': self.evaluation_metrics.get('analogy_accuracy', {}),
                'semantic_coherence': self.evaluation_metrics.get('semantic_coherence', {})
            }
        }

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save analysis report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / "word2vec_analysis_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        # Convert numpy types to Python types for JSON serialization
        # Ensures all numpy types are converted
        report = convert_numpy_types(report)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Word2Vec analysis report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of Word2Vec trainer"""
    # Initialize trainer
    trainer = PersianWord2VecTrainer()

    # Load data (use heavy stem version for better word separation)
    data_path = PROCESSED_DATA_DIR / "comments_heavy_stem_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocessing first.")
        return

    sentences, labels = trainer.load_data(
        str(data_path),
        text_column='comment_processed',
        label_column='sentiment_label'
    )

    # Train Word2Vec model
    model = trainer.train_model(sentences)

    # Evaluate model
    evaluation = trainer.evaluate_model()

    # Create visualizations
    plots = trainer.create_visualizations()

    # Save model and report
    model_path = trainer.save_model(save_full_model=False)  # Save vectors only
    report_path = trainer.save_report()

    # Display results
    print(f"\n✅ Word2Vec Training Completed!")
    print(f"📚 Vocabulary size: {len(trainer.vocabulary)}")
    print(f"🧠 Vector size: {trainer.config.get('vector_size', 200)}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"📈 Visualizations: {len(plots)} plots created")

    # Show some similar words
    test_words = ['بانک', 'اپ', 'خوب', 'بد']
    for word in test_words:
        if word in trainer.word_vectors:
            similar = trainer.find_similar_words(word, topn=5)
            print(f"\nWords similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")


if __name__ == "__main__":
    main()
