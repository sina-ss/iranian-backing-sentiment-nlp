"""
Comprehensive Metrics Calculator for Persian Banking Sentiment Classification
Advanced evaluation metrics with detailed analysis and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config import (
        EVALUATION_CONFIG,
        RESULTS_DIR,
        FIGURES_DIR,
        SENTIMENT_LABELS
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {Path(__file__).parent.parent.parent}")


class PersianSentimentMetricsCalculator:
    """
    Comprehensive metrics calculator for Persian sentiment analysis models
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize metrics calculator

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or EVALUATION_CONFIG
        self.setup_logging()

        # Data storage
        self.y_true = None
        self.y_pred = None
        self.y_proba = None
        self.class_names = None
        self.model_name = None

        # Results storage
        self.basic_metrics = {}
        self.advanced_metrics = {}
        self.class_metrics = {}
        self.confusion_matrices = {}
        self.roc_data = {}
        self.pr_data = {}

        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_classes': 0,
            'calculation_time': 0.0,
            'timestamp': None
        }

    def setup_logging(self):
        """Setup logging for metrics calculation"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "metrics_calculation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None,
                              model_name: str = "Model") -> Dict:
        """
        Calculate comprehensive metrics for sentiment classification

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: Class names (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing all calculated metrics
        """
        start_time = datetime.now()
        self.logger.info(
            f"Calculating comprehensive metrics for {model_name}...")

        # Store data
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
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

        # Update statistics
        self.stats['total_samples'] = len(y_true)
        self.stats['total_classes'] = len(self.class_names)

        # Calculate different types of metrics
        self.basic_metrics = self._calculate_basic_metrics()
        self.advanced_metrics = self._calculate_advanced_metrics()
        self.class_metrics = self._calculate_class_metrics()
        self.confusion_matrices = self._calculate_confusion_matrices()

        if y_proba is not None:
            self.roc_data = self._calculate_roc_metrics()
            self.pr_data = self._calculate_precision_recall_metrics()

        # Calculate processing time
        calculation_time = (datetime.now() - start_time).total_seconds()
        self.stats['calculation_time'] = calculation_time
        self.stats['timestamp'] = datetime.now().isoformat()

        # Compile all results
        all_metrics = {
            'model_name': model_name,
            'basic_metrics': self.basic_metrics,
            'advanced_metrics': self.advanced_metrics,
            'class_metrics': self.class_metrics,
            'confusion_matrices': self.confusion_matrices,
            'roc_data': self.roc_data,
            'precision_recall_data': self.pr_data,
            'statistics': self.stats
        }

        self.logger.info(
            f"Metrics calculation completed in {calculation_time:.3f} seconds")
        self.logger.info(f"Accuracy: {self.basic_metrics['accuracy']:.4f}")
        self.logger.info(
            f"F1-Score (weighted): {self.basic_metrics['f1_weighted']:.4f}")

        return all_metrics

    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        }

        return {k: float(v) for k, v in metrics.items()}

    def _calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced classification metrics"""
        metrics = {}

        # Matthews Correlation Coefficient
        try:
            metrics['matthews_corrcoef'] = float(
                matthews_corrcoef(self.y_true, self.y_pred))
        except:
            metrics['matthews_corrcoef'] = None

        # Cohen's Kappa
        try:
            metrics['cohen_kappa'] = float(
                cohen_kappa_score(self.y_true, self.y_pred))
        except:
            metrics['cohen_kappa'] = None

        # ROC AUC (for multiclass)
        if self.y_proba is not None:
            try:
                # One-vs-Rest AUC
                metrics['roc_auc_ovr'] = float(roc_auc_score(
                    self.y_true, self.y_proba,
                    multi_class='ovr', average='weighted'
                ))

                # One-vs-One AUC
                metrics['roc_auc_ovo'] = float(roc_auc_score(
                    self.y_true, self.y_proba,
                    multi_class='ovo', average='weighted'
                ))
            except:
                metrics['roc_auc_ovr'] = None
                metrics['roc_auc_ovo'] = None

        # Class distribution analysis
        true_distribution = np.bincount(self.y_true) / len(self.y_true)
        pred_distribution = np.bincount(self.y_pred) / len(self.y_pred)

        metrics['class_distribution'] = {
            'true': true_distribution.tolist(),
            'predicted': pred_distribution.tolist(),
            'distribution_difference': (pred_distribution - true_distribution).tolist()
        }

        # Error rate
        metrics['error_rate'] = float(
            1 - accuracy_score(self.y_true, self.y_pred))

        # Correct predictions per class
        correct_per_class = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (self.y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(
                    self.y_true[class_mask],
                    self.y_pred[class_mask]
                )
                correct_per_class.append(float(class_accuracy))
            else:
                correct_per_class.append(0.0)

        metrics['accuracy_per_class'] = dict(
            zip(self.class_names, correct_per_class))

        return metrics

    def _calculate_class_metrics(self) -> Dict:
        """Calculate per-class detailed metrics"""
        # Get classification report
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        class_metrics = {}

        # Extract per-class metrics
        for class_name in self.class_names:
            if class_name in report:
                class_metrics[class_name] = {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1_score': float(report[class_name]['f1-score']),
                    'support': int(report[class_name]['support'])
                }

        # Add summary metrics
        class_metrics['macro_avg'] = {
            'precision': float(report['macro avg']['precision']),
            'recall': float(report['macro avg']['recall']),
            'f1_score': float(report['macro avg']['f1-score']),
            'support': int(report['macro avg']['support'])
        }

        class_metrics['weighted_avg'] = {
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'f1_score': float(report['weighted avg']['f1-score']),
            'support': int(report['weighted avg']['support'])
        }

        return class_metrics

    def _calculate_confusion_matrices(self) -> Dict:
        """Calculate confusion matrices in different formats"""
        # Basic confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Normalized confusion matrices
        cm_normalized_true = confusion_matrix(
            self.y_true, self.y_pred, normalize='true')
        cm_normalized_pred = confusion_matrix(
            self.y_true, self.y_pred, normalize='pred')
        cm_normalized_all = confusion_matrix(
            self.y_true, self.y_pred, normalize='all')

        confusion_data = {
            'raw': cm.tolist(),
            'normalized_by_true': cm_normalized_true.tolist(),
            'normalized_by_pred': cm_normalized_pred.tolist(),
            'normalized_by_all': cm_normalized_all.tolist(),
            'class_names': self.class_names
        }

        # Calculate additional confusion matrix metrics
        if len(self.class_names) == 2:
            # Binary classification specific metrics
            tn, fp, fn, tp = cm.ravel()

            confusion_data['binary_metrics'] = {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            }

        return confusion_data

    def _calculate_roc_metrics(self) -> Dict:
        """Calculate ROC curve data and AUC metrics"""
        if self.y_proba is None:
            return {}

        roc_data = {}

        if len(self.class_names) == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba[:, 1])
            auc = roc_auc_score(self.y_true, self.y_proba[:, 1])

            roc_data['binary'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(auc)
            }

        else:
            # Multiclass classification (One-vs-Rest)
            label_binarizer = LabelBinarizer()
            y_true_binary = label_binarizer.fit_transform(self.y_true)

            roc_data['multiclass'] = {}

            for i, class_name in enumerate(self.class_names):
                try:
                    fpr, tpr, thresholds = roc_curve(
                        y_true_binary[:, i], self.y_proba[:, i])
                    auc = roc_auc_score(
                        y_true_binary[:, i], self.y_proba[:, i])

                    roc_data['multiclass'][class_name] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': float(auc)
                    }
                except:
                    roc_data['multiclass'][class_name] = None

            # Micro-average ROC
            try:
                fpr_micro, tpr_micro, _ = roc_curve(
                    y_true_binary.ravel(), self.y_proba.ravel())
                auc_micro = roc_auc_score(
                    y_true_binary, self.y_proba, average='micro')

                roc_data['micro_average'] = {
                    'fpr': fpr_micro.tolist(),
                    'tpr': tpr_micro.tolist(),
                    'auc': float(auc_micro)
                }
            except:
                roc_data['micro_average'] = None

        return roc_data

    def _calculate_precision_recall_metrics(self) -> Dict:
        """Calculate Precision-Recall curve data and AP metrics"""
        if self.y_proba is None:
            return {}

        pr_data = {}

        if len(self.class_names) == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(
                self.y_true, self.y_proba[:, 1])
            ap = average_precision_score(self.y_true, self.y_proba[:, 1])

            pr_data['binary'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'average_precision': float(ap)
            }

        else:
            # Multiclass classification
            label_binarizer = LabelBinarizer()
            y_true_binary = label_binarizer.fit_transform(self.y_true)

            pr_data['multiclass'] = {}

            for i, class_name in enumerate(self.class_names):
                try:
                    precision, recall, thresholds = precision_recall_curve(
                        y_true_binary[:, i], self.y_proba[:, i]
                    )
                    ap = average_precision_score(
                        y_true_binary[:, i], self.y_proba[:, i])

                    pr_data['multiclass'][class_name] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': thresholds.tolist(),
                        'average_precision': float(ap)
                    }
                except:
                    pr_data['multiclass'][class_name] = None

            # Micro-average PR
            try:
                precision_micro, recall_micro, _ = precision_recall_curve(
                    y_true_binary.ravel(), self.y_proba.ravel()
                )
                ap_micro = average_precision_score(
                    y_true_binary, self.y_proba, average='micro')

                pr_data['micro_average'] = {
                    'precision': precision_micro.tolist(),
                    'recall': recall_micro.tolist(),
                    'average_precision': float(ap_micro)
                }
            except:
                pr_data['micro_average'] = None

        return pr_data

    def create_visualizations(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create comprehensive metric visualizations

        Args:
            save_dir: Directory to save plots

        Returns:
            List of saved plot paths
        """
        if save_dir is None:
            save_dir = FIGURES_DIR / "evaluation" / \
                f"{self.model_name.lower().replace(' ', '_')}"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Confusion Matrix plots
        self._plot_confusion_matrices(save_dir)
        saved_plots.append(str(save_dir / 'confusion_matrices.png'))

        # 2. Performance metrics bar plot
        self._plot_performance_metrics(save_dir)
        saved_plots.append(str(save_dir / 'performance_metrics.png'))

        # 3. Per-class metrics
        self._plot_per_class_metrics(save_dir)
        saved_plots.append(str(save_dir / 'per_class_metrics.png'))

        # 4. ROC curves (if probabilities available)
        if self.roc_data:
            self._plot_roc_curves(save_dir)
            saved_plots.append(str(save_dir / 'roc_curves.png'))

        # 5. Precision-Recall curves (if probabilities available)
        if self.pr_data:
            self._plot_precision_recall_curves(save_dir)
            saved_plots.append(str(save_dir / 'precision_recall_curves.png'))

        # 6. Class distribution comparison
        self._plot_class_distribution(save_dir)
        saved_plots.append(str(save_dir / 'class_distribution.png'))

        return saved_plots

    def _plot_confusion_matrices(self, save_dir: Path):
        """Plot confusion matrices in different normalizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Raw confusion matrix
        cm_raw = np.array(self.confusion_matrices['raw'])
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[0, 0])
        axes[0, 0].set_title('Raw Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Normalized by true labels (recall)
        cm_norm_true = np.array(self.confusion_matrices['normalized_by_true'])
        sns.heatmap(cm_norm_true, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[0, 1])
        axes[0, 1].set_title('Normalized by True Labels (Recall)')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Normalized by predictions (precision)
        cm_norm_pred = np.array(self.confusion_matrices['normalized_by_pred'])
        sns.heatmap(cm_norm_pred, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[1, 0])
        axes[1, 0].set_title('Normalized by Predictions (Precision)')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')

        # Normalized by all samples
        cm_norm_all = np.array(self.confusion_matrices['normalized_by_all'])
        sns.heatmap(cm_norm_all, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[1, 1])
        axes[1, 1].set_title('Normalized by All Samples')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')

        plt.suptitle(f'{self.model_name} - Confusion Matrices', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_metrics(self, save_dir: Path):
        """Plot overall performance metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Basic metrics
        basic_metrics = ['accuracy', 'precision_weighted',
                         'recall_weighted', 'f1_weighted']
        basic_values = [self.basic_metrics[metric] for metric in basic_metrics]
        basic_labels = [metric.replace('_', ' ').title()
                        for metric in basic_metrics]

        bars1 = axes[0].bar(basic_labels, basic_values,
                            color='skyblue', alpha=0.8)
        axes[0].set_title('Basic Performance Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, basic_values):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f'{value:.3f}', ha='center', va='bottom')

        # Advanced metrics (if available)
        advanced_metrics = []
        advanced_values = []

        if self.advanced_metrics.get('matthews_corrcoef') is not None:
            advanced_metrics.append('Matthews Correlation')
            advanced_values.append(self.advanced_metrics['matthews_corrcoef'])

        if self.advanced_metrics.get('cohen_kappa') is not None:
            advanced_metrics.append('Cohen\'s Kappa')
            advanced_values.append(self.advanced_metrics['cohen_kappa'])

        if self.advanced_metrics.get('roc_auc_ovr') is not None:
            advanced_metrics.append('ROC AUC (OvR)')
            advanced_values.append(self.advanced_metrics['roc_auc_ovr'])

        if advanced_metrics:
            bars2 = axes[1].bar(
                advanced_metrics, advanced_values, color='lightgreen', alpha=0.8)
            axes[1].set_title('Advanced Performance Metrics')
            axes[1].set_ylabel('Score')
            axes[1].grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars2, advanced_values):
                axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                             f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1].text(0.5, 0.5, 'No advanced metrics available',
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Advanced Performance Metrics')

        plt.suptitle(f'{self.model_name} - Performance Overview', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_metrics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_class_metrics(self, save_dir: Path):
        """Plot per-class performance metrics"""
        classes = [cls for cls in self.class_names if cls in self.class_metrics]

        if not classes:
            return

        metrics = ['precision', 'recall', 'f1_score']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(classes))
        width = 0.25

        colors = ['skyblue', 'lightgreen', 'lightcoral']

        for i, metric in enumerate(metrics):
            values = [self.class_metrics[cls][metric] for cls in classes]
            bars = ax.bar(x + i*width, values, width, label=metric.title(),
                          color=colors[i], alpha=0.8)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'{self.model_name} - Per-Class Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_metrics.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, save_dir: Path):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))

        if 'binary' in self.roc_data:
            # Binary classification
            fpr = self.roc_data['binary']['fpr']
            tpr = self.roc_data['binary']['tpr']
            auc = self.roc_data['binary']['auc']

            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')

        elif 'multiclass' in self.roc_data:
            # Multiclass classification
            for class_name, roc_info in self.roc_data['multiclass'].items():
                if roc_info is not None:
                    fpr = roc_info['fpr']
                    tpr = roc_info['tpr']
                    auc = roc_info['auc']

                    plt.plot(fpr, tpr, lw=2,
                             label=f'{class_name} (AUC = {auc:.3f})')

            # Add micro-average if available
            if self.roc_data.get('micro_average'):
                fpr_micro = self.roc_data['micro_average']['fpr']
                tpr_micro = self.roc_data['micro_average']['tpr']
                auc_micro = self.roc_data['micro_average']['auc']

                plt.plot(fpr_micro, tpr_micro, lw=2, linestyle='--',
                         label=f'Micro-average (AUC = {auc_micro:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curves(self, save_dir: Path):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=(10, 8))

        if 'binary' in self.pr_data:
            # Binary classification
            precision = self.pr_data['binary']['precision']
            recall = self.pr_data['binary']['recall']
            ap = self.pr_data['binary']['average_precision']

            plt.plot(recall, precision, lw=2,
                     label=f'PR curve (AP = {ap:.3f})')

        elif 'multiclass' in self.pr_data:
            # Multiclass classification
            for class_name, pr_info in self.pr_data['multiclass'].items():
                if pr_info is not None:
                    precision = pr_info['precision']
                    recall = pr_info['recall']
                    ap = pr_info['average_precision']

                    plt.plot(recall, precision, lw=2,
                             label=f'{class_name} (AP = {ap:.3f})')

            # Add micro-average if available
            if self.pr_data.get('micro_average'):
                precision_micro = self.pr_data['micro_average']['precision']
                recall_micro = self.pr_data['micro_average']['recall']
                ap_micro = self.pr_data['micro_average']['average_precision']

                plt.plot(recall_micro, precision_micro, lw=2, linestyle='--',
                         label=f'Micro-average (AP = {ap_micro:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'precision_recall_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_class_distribution(self, save_dir: Path):
        """Plot class distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # True vs Predicted distribution
        true_dist = self.advanced_metrics['class_distribution']['true']
        pred_dist = self.advanced_metrics['class_distribution']['predicted']

        x = np.arange(len(self.class_names))
        width = 0.35

        bars1 = axes[0].bar(x - width/2, true_dist, width,
                            label='True', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, pred_dist, width,
                            label='Predicted', alpha=0.8)

        axes[0].set_xlabel('Classes')
        axes[0].set_ylabel('Proportion')
        axes[0].set_title('Class Distribution Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.3f}', ha='center', va='bottom')

        # Distribution difference
        diff = self.advanced_metrics['class_distribution']['distribution_difference']
        colors = ['red' if d < 0 else 'green' for d in diff]

        bars3 = axes[1].bar(self.class_names, diff, color=colors, alpha=0.7)
        axes[1].set_xlabel('Classes')
        axes[1].set_ylabel('Difference (Pred - True)')
        axes[1].set_title('Prediction Bias by Class')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars3, diff):
            axes[1].text(bar.get_x() + bar.get_width()/2.,
                         bar.get_height() + (0.01 if value >= 0 else -0.02),
                         f'{value:.3f}', ha='center',
                         va='bottom' if value >= 0 else 'top')

        plt.suptitle(
            f'{self.model_name} - Class Distribution Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'class_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics_report(self, report_path: Optional[str] = None) -> str:
        """
        Save comprehensive metrics report to JSON

        Args:
            report_path: Optional path to save report

        Returns:
            Path to saved report
        """
        if report_path is None:
            safe_model_name = self.model_name.lower().replace(' ', '_').replace('/', '_')
            report_path = RESULTS_DIR / "metrics" / \
                f"{safe_model_name}_metrics_report.json"

        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Compile comprehensive report
        report = {
            'model_name': self.model_name,
            'evaluation_timestamp': self.stats['timestamp'],
            'dataset_info': {
                'total_samples': self.stats['total_samples'],
                'total_classes': self.stats['total_classes'],
                'class_names': self.class_names
            },
            'basic_metrics': self.basic_metrics,
            'advanced_metrics': self.advanced_metrics,
            'per_class_metrics': self.class_metrics,
            'confusion_matrices': self.confusion_matrices,
            'roc_analysis': self.roc_data,
            'precision_recall_analysis': self.pr_data,
            'processing_info': {
                'calculation_time_seconds': self.stats['calculation_time']
            }
        }

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Metrics report saved to {report_path}")
        return str(report_path)


def main():
    """Example usage of metrics calculator"""
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3

    # Generate sample true labels
    y_true = np.random.choice(n_classes, n_samples)

    # Generate sample predictions (with some accuracy)
    y_pred = y_true.copy()
    noise_indices = np.random.choice(
        n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice(n_classes, len(noise_indices))

    # Generate sample probabilities
    y_proba = np.random.random((n_samples, n_classes))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    class_names = ['negative', 'neutral', 'positive']

    # Initialize calculator
    calculator = PersianSentimentMetricsCalculator()

    # Calculate all metrics
    all_metrics = calculator.calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=class_names,
        model_name="Sample Model"
    )

    # Create visualizations
    plots = calculator.create_visualizations()

    # Save report
    report_path = calculator.save_metrics_report()

    # Display results
    print(f"\n✅ Metrics Calculation Completed!")
    print(f"📊 Accuracy: {all_metrics['basic_metrics']['accuracy']:.4f}")
    print(
        f"📊 F1-Score (weighted): {all_metrics['basic_metrics']['f1_weighted']:.4f}")
    print(
        f"📊 Matthews Correlation: {all_metrics['advanced_metrics']['matthews_corrcoef']:.4f}")
    print(f"📈 Visualizations: {len(plots)} plots created")
    print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
