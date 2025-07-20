"""
Label Analysis Utility
Analyzes the results of OpenAI comment labeling and generates insights
"""

from config import FILE_PATHS, RESULTS_DIR, FIGURES_DIR
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from collections import Counter
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class LabelAnalyzer:
    """Analyze labeled comment data and generate insights"""

    def __init__(self, labeled_csv_path: str):
        """
        Initialize analyzer with labeled data

        Args:
            labeled_csv_path: Path to CSV file with labeled comments
        """
        self.data_path = labeled_csv_path
        self.df = pd.read_csv(labeled_csv_path)
        self.setup_plotting()

    def setup_plotting(self):
        """Setup matplotlib for Persian text"""
        plt.rcParams['font.family'] = [
            'Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")

    def basic_statistics(self) -> Dict:
        """Generate basic statistics about labeled data"""
        stats = {
            "total_comments": len(self.df),
            "sentiment_distribution": self.df['sentiment_label'].value_counts().to_dict(),
            "sentiment_percentages": (self.df['sentiment_label'].value_counts(normalize=True) * 100).round(2).to_dict(),
            "apps_analyzed": self.df['app_name'].nunique() if 'app_name' in self.df.columns else 0,
            "rating_sentiment_correlation": self._analyze_rating_sentiment() if 'rating' in self.df.columns else None
        }

        return stats

    def _analyze_rating_sentiment(self) -> Dict:
        """Analyze correlation between ratings and sentiment labels"""
        if 'rating' not in self.df.columns:
            return None

        # Cross-tabulation of rating and sentiment
        crosstab = pd.crosstab(
            self.df['rating'], self.df['sentiment_label'], normalize='index') * 100

        # Average rating per sentiment
        avg_rating = self.df.groupby('sentiment_label')[
            'rating'].mean().round(2).to_dict()

        return {
            "crosstab": crosstab.round(2).to_dict(),
            "average_rating_per_sentiment": avg_rating,
            "correlation_coefficient": self.df['rating'].corr(
                self.df['sentiment_label'].map(
                    {'negative': 1, 'neutral': 2, 'positive': 3})
            ).round(3) if len(self.df['sentiment_label'].unique()) > 1 else None
        }

    def app_level_analysis(self) -> Dict:
        """Analyze sentiment by banking app"""
        if 'app_name' not in self.df.columns:
            return {"error": "app_name column not found"}

        app_sentiment = self.df.groupby('app_name')['sentiment_label'].value_counts(
            normalize=True).unstack(fill_value=0) * 100
        app_counts = self.df['app_name'].value_counts()

        # Find most positive and negative apps
        app_sentiment['positive_score'] = app_sentiment.get(
            'positive', 0) - app_sentiment.get('negative', 0)

        return {
            "app_sentiment_distribution": app_sentiment.round(2).to_dict(),
            "app_comment_counts": app_counts.to_dict(),
            "most_positive_apps": app_sentiment.nlargest(5, 'positive_score')['positive_score'].to_dict(),
            "most_negative_apps": app_sentiment.nsmallest(5, 'positive_score')['positive_score'].to_dict()
        }

    def temporal_analysis(self) -> Dict:
        """Analyze sentiment trends over time"""
        if 'date' not in self.df.columns:
            return {"error": "date column not found"}

        try:
            # Convert Persian dates or handle date formats
            self.df['date_parsed'] = pd.to_datetime(
                self.df['date'], errors='coerce')

            if self.df['date_parsed'].isna().all():
                # Try to parse Persian dates (basic conversion)
                return {"error": "Could not parse date format"}

            # Group by date and sentiment
            temporal_sentiment = self.df.groupby([
                self.df['date_parsed'].dt.date,
                'sentiment_label'
            ]).size().unstack(fill_value=0)

            return {
                "daily_sentiment_counts": temporal_sentiment.to_dict(),
                "sentiment_trend": "analysis_available"
            }
        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}

    def text_analysis(self) -> Dict:
        """Analyze text characteristics by sentiment"""
        # Comment length analysis
        self.df['comment_length'] = self.df['comment'].str.len()
        length_by_sentiment = self.df.groupby('sentiment_label')['comment_length'].agg([
            'mean', 'median', 'std']).round(2)

        # Common words by sentiment (basic analysis)
        def get_common_words(text_series, top_n=10):
            all_text = ' '.join(text_series.fillna(''))
            # Simple word extraction (for Persian, you might want to use hazm)
            words = re.findall(r'\b\w+\b', all_text.lower())
            return dict(Counter(words).most_common(top_n))

        common_words = {}
        for sentiment in self.df['sentiment_label'].unique():
            sentiment_comments = self.df[self.df['sentiment_label']
                                         == sentiment]['comment']
            common_words[sentiment] = get_common_words(sentiment_comments)

        return {
            "comment_length_stats": length_by_sentiment.to_dict(),
            "common_words_by_sentiment": common_words
        }

    def generate_visualizations(self, save_dir: str = None) -> List[str]:
        """Generate and save visualization plots"""
        if save_dir is None:
            save_dir = FIGURES_DIR

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        # 1. Sentiment Distribution Pie Chart
        plt.figure(figsize=(10, 8))
        sentiment_counts = self.df['sentiment_label'].value_counts()
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # red, yellow, green
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Overall Sentiment Distribution',
                  fontsize=16, fontweight='bold')
        plot_path = save_dir / 'sentiment_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(plot_path))

        # 2. Rating vs Sentiment Analysis
        if 'rating' in self.df.columns:
            plt.figure(figsize=(12, 8))

            # Box plot
            plt.subplot(2, 2, 1)
            sns.boxplot(data=self.df, x='sentiment_label', y='rating')
            plt.title('Rating Distribution by Sentiment')
            plt.xticks(rotation=45)

            # Heatmap
            plt.subplot(2, 2, 2)
            crosstab = pd.crosstab(
                self.df['rating'], self.df['sentiment_label'], normalize='index')
            sns.heatmap(crosstab, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Rating vs Sentiment Heatmap')

            # Average rating per sentiment
            plt.subplot(2, 2, 3)
            avg_rating = self.df.groupby('sentiment_label')['rating'].mean()
            bars = plt.bar(avg_rating.index, avg_rating.values, color=colors)
            plt.title('Average Rating by Sentiment')
            plt.ylabel('Average Rating')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plot_path = save_dir / 'rating_sentiment_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(plot_path))

        # 3. App-level Analysis
        if 'app_name' in self.df.columns and self.df['app_name'].nunique() > 1:
            plt.figure(figsize=(14, 10))

            # App sentiment distribution
            app_sentiment = self.df.groupby(
                ['app_name', 'sentiment_label']).size().unstack(fill_value=0)
            app_sentiment_pct = app_sentiment.div(
                app_sentiment.sum(axis=1), axis=0) * 100

            plt.subplot(2, 1, 1)
            app_sentiment_pct.plot(
                kind='bar', stacked=True, color=colors, ax=plt.gca())
            plt.title('Sentiment Distribution by Banking App')
            plt.xlabel('Banking App')
            plt.ylabel('Percentage')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45, ha='right')

            # Comment count by app
            plt.subplot(2, 1, 2)
            app_counts = self.df['app_name'].value_counts()
            plt.bar(range(len(app_counts)), app_counts.values)
            plt.title('Number of Comments by Banking App')
            plt.xlabel('Banking App')
            plt.ylabel('Comment Count')
            plt.xticks(range(len(app_counts)), app_counts.index,
                       rotation=45, ha='right')

            plt.tight_layout()
            plot_path = save_dir / 'app_level_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(plot_path))

        # 4. Comment Length Analysis
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        self.df['comment_length'] = self.df['comment'].str.len()
        sns.boxplot(data=self.df, x='sentiment_label', y='comment_length')
        plt.title('Comment Length by Sentiment')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        for sentiment in self.df['sentiment_label'].unique():
            sentiment_lengths = self.df[self.df['sentiment_label']
                                        == sentiment]['comment_length']
            plt.hist(sentiment_lengths, alpha=0.7, label=sentiment, bins=20)
        plt.title('Comment Length Distribution')
        plt.xlabel('Comment Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plot_path = save_dir / 'comment_length_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(plot_path))

        return saved_plots

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "data_source": self.data_path,
            "basic_statistics": self.basic_statistics(),
            "app_level_analysis": self.app_level_analysis(),
            "temporal_analysis": self.temporal_analysis(),
            "text_analysis": self.text_analysis()
        }

        return report

    def save_report(self, output_path: str = None) -> str:
        """Save analysis report to JSON file"""
        if output_path is None:
            output_path = RESULTS_DIR / "reports" / "label_analysis_report.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return str(output_path)


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze labeled comment data")
    parser.add_argument('--input', '-i',
                        default=str(FILE_PATHS["labeled_comments"]),
                        help='Path to labeled CSV file')
    parser.add_argument('--output-report', '-o',
                        help='Path to save analysis report JSON')
    parser.add_argument('--output-plots', '-p',
                        default=str(FIGURES_DIR),
                        help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    print("📊 Starting label analysis...")

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = LabelAnalyzer(args.input)

    # Generate report
    print("📝 Generating analysis report...")
    report_path = analyzer.save_report(args.output_report)
    print(f"✅ Report saved to: {report_path}")

    # Generate visualizations
    if not args.no_plots:
        print("📈 Generating visualizations...")
        plot_paths = analyzer.generate_visualizations(args.output_plots)
        print(f"✅ {len(plot_paths)} plots saved to: {args.output_plots}")
        for plot in plot_paths:
            print(f"  - {Path(plot).name}")

    # Display summary
    basic_stats = analyzer.basic_statistics()
    print(f"\n📋 Summary:")
    print(f"  Total comments: {basic_stats['total_comments']}")
    print(f"  Sentiment distribution:")
    for sentiment, count in basic_stats['sentiment_distribution'].items():
        percentage = basic_stats['sentiment_percentages'][sentiment]
        print(f"    {sentiment.capitalize()}: {count} ({percentage}%)")

    if basic_stats['apps_analyzed'] > 0:
        print(f"  Banking apps analyzed: {basic_stats['apps_analyzed']}")


if __name__ == "__main__":
    main()
