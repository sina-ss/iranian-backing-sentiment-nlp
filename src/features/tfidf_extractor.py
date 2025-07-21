"""
TF-IDF Feature Extractor for Persian Banking Comments
Optimized for Persian text with comprehensive analysis and visualization capabilities
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging
from datetime import datetime

# Import project configuration
import sys
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


class PersianTfidfExtractor:
    """
    Advanced TF-IDF feature extractor optimized for Persian banking comments
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TF-IDF extractor

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or FEATURE_CONFIG.get('tfidf', {})
        self.setup_logging()

        # Initialize components
        self.vectorizer = None
        self.feature_matrix = None
        self.feature_names = None
        self.vocabulary = None

        # Data storage
        self.documents = None
        self.labels = None
        self.document_info = None

        # Analysis results
        self.feature_stats = {}
        self.similarity_matrix = None

        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'vocabulary_size': 0,
            'feature_matrix_shape': None,
            'sparsity': 0.0,
            'processing_time': 0.0,
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for feature extraction"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "tfidf_extraction.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def persian_analyzer(self, text):
        """
        Custom analyzer for Persian text - moved to class method to enable pickling

        Args:
            text: Input text to analyze

        Returns:
            List of tokens
        """
        # Basic tokenization (words already preprocessed)
        tokens = text.split()

        # Filter by length
        min_len = 2
        max_len = 20
        tokens = [token for token in tokens if min_len <=
                  len(token) <= max_len]

        return tokens

    def load_data(self, data_path: str, text_column: str = 'comment_processed',
                  label_column: Optional[str] = None) -> Tuple[List[str], Optional[List[str]], pd.DataFrame]:
        """
        Load preprocessed data for feature extraction

        Args:
            data_path: Path to preprocessed CSV file
            text_column: Name of the text column
            label_column: Optional name of the label column

        Returns:
            Tuple of (documents, labels, document_info)
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            df = pd.read_csv(data_path)

            if text_column not in df.columns:
                raise ValueError(
                    f"Text column '{text_column}' not found in data")

            # Extract documents
            documents = df[text_column].fillna('').astype(str).tolist()

            # Extract labels if available
            labels = None
            if label_column and label_column in df.columns:
                labels = df[label_column].tolist()

            # Store document info for analysis
            document_info = df.copy()

            self.logger.info(f"Loaded {len(documents)} documents")
            if labels:
                label_counts = pd.Series(labels).value_counts()
                self.logger.info(
                    f"Label distribution: {label_counts.to_dict()}")

            return documents, labels, document_info

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_vectorizer(self, custom_config: Optional[Dict] = None) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with Persian-optimized settings

        Args:
            custom_config: Optional custom configuration

        Returns:
            Configured TfidfVectorizer
        """
        config = custom_config or self.config

        vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 10000),
            min_df=config.get('min_df', 2),
            max_df=config.get('max_df', 0.8),
            ngram_range=config.get('ngram_range', (1, 2)),
            use_idf=config.get('use_idf', True),
            smooth_idf=config.get('smooth_idf', True),
            sublinear_tf=config.get('sublinear_tf', True),
            analyzer=self.persian_analyzer,  # Now using class method
            lowercase=False,  # Already handled in preprocessing
            strip_accents=None,  # Already handled in preprocessing
            stop_words=None,  # Already handled in preprocessing
            token_pattern=None,  # Using custom analyzer
            dtype=np.float32  # Memory optimization
        )

        self.logger.info(f"Created TF-IDF vectorizer with config: {config}")
        return vectorizer

    def fit_transform(self, documents: List[str],
                      labels: Optional[List[str]] = None,
                      document_info: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform documents

        Args:
            documents: List of preprocessed documents
            labels: Optional list of labels
            document_info: Optional DataFrame with document metadata

        Returns:
            TF-IDF feature matrix
        """
        start_time = datetime.now()
        self.logger.info("Starting TF-IDF fitting and transformation...")

        # Store data
        self.documents = documents
        self.labels = labels
        self.document_info = document_info

        # Create and fit vectorizer
        self.vectorizer = self.create_vectorizer()

        try:
            # Fit and transform
            self.feature_matrix = self.vectorizer.fit_transform(documents)

            # Extract feature information
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.vocabulary = self.vectorizer.vocabulary_

            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._calculate_statistics(processing_time)

            self.logger.info(
                f"TF-IDF extraction completed in {processing_time:.2f} seconds")
            self.logger.info(
                f"Feature matrix shape: {self.feature_matrix.shape}")
            self.logger.info(f"Vocabulary size: {len(self.vocabulary)}")
            self.logger.info(f"Matrix sparsity: {self.stats['sparsity']:.2%}")

            return self.feature_matrix

        except Exception as e:
            self.logger.error(f"Error in TF-IDF transformation: {e}")
            raise

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform new documents using fitted vectorizer

        Args:
            documents: List of preprocessed documents

        Returns:
            TF-IDF feature matrix for new documents
        """
        if self.vectorizer is None:
            raise ValueError(
                "Vectorizer not fitted. Call fit_transform first.")

        return self.vectorizer.transform(documents)

    def _calculate_statistics(self, processing_time: float):
        """Calculate and store feature extraction statistics"""
        self.stats.update({
            'documents_processed': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'feature_matrix_shape': self.feature_matrix.shape,
            'sparsity': 1.0 - (self.feature_matrix.nnz / (self.feature_matrix.shape[0] * self.feature_matrix.shape[1])),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })

        # Calculate feature statistics
        self._analyze_features()

    def _analyze_features(self):
        """Analyze TF-IDF features and calculate statistics"""
        if self.feature_matrix is None:
            return

        # Feature frequency statistics
        feature_sums = np.array(self.feature_matrix.sum(axis=0)).flatten()
        feature_means = np.array(self.feature_matrix.mean(axis=0)).flatten()
        feature_max = np.array(self.feature_matrix.max(axis=0)).flatten()

        self.feature_stats = {
            'feature_sums': feature_sums,
            'feature_means': feature_means,
            'feature_max': feature_max,
            'top_features': self._get_top_features(feature_sums),
            'feature_distribution': self._analyze_feature_distribution()
        }

    def _get_top_features(self, feature_scores: np.ndarray, top_n: int = 50) -> Dict[str, float]:
        """Get top features by score"""
        top_indices = np.argsort(feature_scores)[::-1][:top_n]
        top_features = {}

        for idx in top_indices:
            feature_name = self.feature_names[idx]
            score = feature_scores[idx]
            top_features[feature_name] = float(score)

        return top_features

    def _analyze_feature_distribution(self) -> Dict:
        """Analyze feature distribution across different categories"""
        distribution = {
            'total_features': len(self.feature_names),
            'unigrams': sum(1 for name in self.feature_names if ' ' not in name),
            'bigrams': sum(1 for name in self.feature_names if name.count(' ') == 1),
            'trigrams': sum(1 for name in self.feature_names if name.count(' ') == 2)
        }

        distribution['unigram_ratio'] = distribution['unigrams'] / \
            distribution['total_features']
        distribution['bigram_ratio'] = distribution['bigrams'] / \
            distribution['total_features']

        return distribution

    def analyze_by_sentiment(self) -> Dict:
        """Analyze TF-IDF features by sentiment labels"""
        if self.labels is None:
            self.logger.warning("No labels available for sentiment analysis")
            return {}

        sentiment_analysis = {}
        unique_labels = list(set(self.labels))

        for label in unique_labels:
            # Get documents for this sentiment
            label_indices = [i for i, l in enumerate(
                self.labels) if l == label]
            label_matrix = self.feature_matrix[label_indices]

            # Calculate average TF-IDF scores for this sentiment
            avg_scores = np.array(label_matrix.mean(axis=0)).flatten()

            # Get top features for this sentiment
            top_features = self._get_top_features(avg_scores, top_n=20)

            sentiment_analysis[label] = {
                'document_count': len(label_indices),
                'avg_tfidf_scores': avg_scores,
                'top_features': top_features
            }

        return sentiment_analysis

    def calculate_document_similarity(self, sample_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate cosine similarity between documents

        Args:
            sample_size: Optional limit for large datasets

        Returns:
            Similarity matrix
        """
        if self.feature_matrix is None:
            raise ValueError(
                "Feature matrix not available. Run fit_transform first.")

        matrix = self.feature_matrix

        # Sample for large datasets
        if sample_size and matrix.shape[0] > sample_size:
            sample_indices = np.random.choice(
                matrix.shape[0], sample_size, replace=False)
            matrix = matrix[sample_indices]
            self.logger.info(
                f"Using sample of {sample_size} documents for similarity calculation")

        self.logger.info("Calculating document similarity matrix...")
        self.similarity_matrix = cosine_similarity(matrix)

        return self.similarity_matrix

    def dimensionality_reduction(self, n_components: int = 100) -> Tuple[np.ndarray, TruncatedSVD]:
        """
        Perform dimensionality reduction using SVD

        Args:
            n_components: Number of components to keep

        Returns:
            Tuple of (reduced_matrix, svd_model)
        """
        if self.feature_matrix is None:
            raise ValueError(
                "Feature matrix not available. Run fit_transform first.")

        self.logger.info(
            f"Performing SVD dimensionality reduction to {n_components} components...")

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(self.feature_matrix)

        explained_variance_ratio = svd.explained_variance_ratio_.sum()
        self.logger.info(
            f"SVD completed. Explained variance ratio: {explained_variance_ratio:.3f}")

        return reduced_matrix, svd

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "tfidf"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Feature distribution plot
        plt.figure(figsize=(12, 8))

        # Top features by sum
        top_features = self.feature_stats['top_features']
        features = list(top_features.keys())[:20]
        scores = list(top_features.values())[:20]

        plt.subplot(2, 2, 1)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('TF-IDF Sum Score')
        plt.title('Top 20 Features by Total TF-IDF Score')
        plt.gca().invert_yaxis()

        # Feature type distribution
        plt.subplot(2, 2, 2)
        dist = self.feature_stats['feature_distribution']
        labels = ['Unigrams', 'Bigrams', 'Trigrams']
        sizes = [dist['unigrams'], dist['bigrams'], dist.get('trigrams', 0)]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Feature Type Distribution')

        # Sparsity visualization
        plt.subplot(2, 2, 3)
        sparsity_data = [self.stats['sparsity'] *
                         100, (1 - self.stats['sparsity']) * 100]
        plt.pie(sparsity_data, labels=[
                'Sparse', 'Non-zero'], autopct='%1.1f%%')
        plt.title(
            f'Matrix Sparsity\n({self.feature_matrix.shape[0]} × {self.feature_matrix.shape[1]})')

        # Feature frequency histogram
        plt.subplot(2, 2, 4)
        feature_sums = self.feature_stats['feature_sums']
        plt.hist(feature_sums, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Feature Sum Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Scores')
        plt.yscale('log')

        plt.tight_layout()
        plot_path = save_dir / 'tfidf_feature_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(plot_path))

        # 2. Sentiment-based analysis
        if self.labels:
            sentiment_analysis = self.analyze_by_sentiment()
            self._plot_sentiment_features(sentiment_analysis, save_dir)
            saved_plots.append(str(save_dir / 'tfidf_sentiment_analysis.png'))

        # 3. Document similarity heatmap (for small datasets)
        if self.feature_matrix.shape[0] <= 100:
            similarity_matrix = self.calculate_document_similarity()
            self._plot_similarity_heatmap(similarity_matrix, save_dir)
            saved_plots.append(str(save_dir / 'tfidf_similarity_heatmap.png'))

        # 4. Word cloud of top features
        self._create_feature_wordcloud(save_dir)
        saved_plots.append(str(save_dir / 'tfidf_wordcloud.png'))

        return saved_plots

    def _plot_sentiment_features(self, sentiment_analysis: Dict, save_dir: Path):
        """Plot sentiment-specific feature analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        sentiments = list(sentiment_analysis.keys())
        colors = ['red', 'gray', 'green'] if len(
            sentiments) == 3 else plt.cm.Set3(np.linspace(0, 1, len(sentiments)))

        for i, (sentiment, data) in enumerate(sentiment_analysis.items()):
            if i < len(axes):
                ax = axes[i]

                # Plot top features for this sentiment
                features = list(data['top_features'].keys())[:15]
                scores = list(data['top_features'].values())[:15]

                color = colors[i] if len(sentiments) <= 3 else colors[i]
                ax.barh(range(len(features)), scores, color=color, alpha=0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Average TF-IDF Score')
                ax.set_title(
                    f'Top Features - {sentiment.title()}\n({data["document_count"]} documents)')
                ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_dir / 'tfidf_sentiment_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_similarity_heatmap(self, similarity_matrix: np.ndarray, save_dir: Path):
        """Plot document similarity heatmap"""
        plt.figure(figsize=(10, 8))

        # Use labels for annotation if available
        labels = None
        if self.labels and len(self.labels) == similarity_matrix.shape[0]:
            labels = self.labels

        sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )

        plt.title('Document Similarity Matrix (TF-IDF)')
        plt.xlabel('Document Index')
        plt.ylabel('Document Index')

        plt.tight_layout()
        plt.savefig(save_dir / 'tfidf_similarity_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_feature_wordcloud(self, save_dir: Path):
        """Create word cloud from top features"""
        if not self.feature_stats:
            return

        # Prepare word frequencies
        word_freq = {}
        for feature, score in self.feature_stats['top_features'].items():
            # Only use unigrams for word cloud
            if ' ' not in feature:
                word_freq[feature] = score

        if not word_freq:
            return

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('TF-IDF Top Features Word Cloud', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'tfidf_wordcloud.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the fitted TF-IDF vectorizer and features

        Args:
            model_path: Optional path to save the model

        Returns:
            Path to saved model
        """
        if self.vectorizer is None:
            raise ValueError("No fitted vectorizer to save")

        if model_path is None:
            model_path = MODELS_DIR / "saved_models" / "tfidf_vectorizer.pkl"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model data
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'vocabulary': self.vocabulary,
            'config': self.config,
            'stats': self.stats,
            'feature_stats': self.feature_stats
        }

        # Save using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"TF-IDF model saved to {model_path}")
        return str(model_path)

    def load_model(self, model_path: str):
        """
        Load a saved TF-IDF vectorizer and features

        Args:
            model_path: Path to saved model
        """
        self.logger.info(f"Loading TF-IDF model from {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.vocabulary = model_data['vocabulary']
        self.config = model_data.get('config', {})
        self.stats = model_data.get('stats', {})
        self.feature_stats = model_data.get('feature_stats', {})

        self.logger.info("TF-IDF model loaded successfully")

    def generate_report(self) -> Dict:
        """Generate comprehensive TF-IDF analysis report"""
        report = {
            'extraction_summary': {
                'timestamp': self.stats.get('timestamp'),
                'documents_processed': self.stats.get('documents_processed'),
                'vocabulary_size': self.stats.get('vocabulary_size'),
                'feature_matrix_shape': self.stats.get('feature_matrix_shape'),
                'sparsity': self.stats.get('sparsity'),
                'processing_time_seconds': self.stats.get('processing_time')
            },
            'configuration': self.config,
            'feature_analysis': {
                'top_features': self.feature_stats.get('top_features', {}),
                'feature_distribution': self.feature_stats.get('feature_distribution', {}),
                'vocabulary_coverage': len(self.vocabulary) if self.vocabulary else 0
            },
            'data_characteristics': {
                'avg_document_length': np.mean([len(doc.split()) for doc in self.documents]) if self.documents else 0,
                'unique_documents': len(set(self.documents)) if self.documents else 0,
                'label_distribution': pd.Series(self.labels).value_counts().to_dict() if self.labels else {}
            }
        }

        # Add sentiment analysis if available
        if self.labels:
            report['sentiment_analysis'] = self.analyze_by_sentiment()

        return report

    def save_report(self, report_path: Optional[str] = None) -> str:
        """Save analysis report to JSON file"""
        if report_path is None:
            report_path = RESULTS_DIR / "reports" / "tfidf_analysis_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"TF-IDF analysis report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of TF-IDF extractor"""
    # Initialize extractor
    extractor = PersianTfidfExtractor()

    # Load data (use medium processed version)
    data_path = PROCESSED_DATA_DIR / "comments_medium_processed.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocessing first.")
        return

    documents, labels, doc_info = extractor.load_data(
        str(data_path),
        text_column='comment_processed',
        label_column='sentiment_label'
    )

    # Extract TF-IDF features
    feature_matrix = extractor.fit_transform(documents, labels, doc_info)

    # Create visualizations
    plots = extractor.create_visualizations()

    # Save model and report
    model_path = extractor.save_model()
    report_path = extractor.save_report()

    # Display results
    print(f"\n✅ TF-IDF Feature Extraction Completed!")
    print(f"📊 Feature matrix shape: {feature_matrix.shape}")
    print(f"📚 Vocabulary size: {len(extractor.vocabulary)}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"📈 Visualizations: {len(plots)} plots created")


if __name__ == "__main__":
    main()
