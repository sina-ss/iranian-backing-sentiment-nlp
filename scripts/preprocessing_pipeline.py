#!/usr/bin/env python3
"""
Persian Text Preprocessing Pipeline
Processes raw comments and prepares them for feature extraction and modeling
"""


import argparse
import pandas as pd
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
from datetime import datetime
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the src directory specifically
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from config import (
        FILE_PATHS,
        PROCESSED_DATA_DIR,
        PREPROCESSING_CONFIG,
        RESULTS_DIR
    )
    from src.preprocessing.persian_cleaner import PersianTextCleaner
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Persian banking comments
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessing pipeline

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or PREPROCESSING_CONFIG
        self.setup_logging()

        # Initialize text cleaner
        self.cleaner = PersianTextCleaner(config)

        # Statistics tracking
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'input_comments': 0,
            'output_comments': 0,
            'filtered_comments': 0,
            'processing_stages': {}
        }

    def setup_logging(self):
        """Setup logging for preprocessing pipeline"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "preprocessing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load input data

        Args:
            input_path: Path to input CSV file

        Returns:
            DataFrame with comments
        """
        self.logger.info(f"Loading data from {input_path}")

        try:
            df = pd.read_csv(input_path)
            self.logger.info(f"Loaded {len(df)} records")

            # Validate required columns
            required_columns = ['comment']
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}")

            # Basic data validation
            original_count = len(df)
            df = df.dropna(subset=['comment'])
            df = df[df['comment'].str.strip() != '']

            if len(df) < original_count:
                self.logger.info(
                    f"Removed {original_count - len(df)} empty/null comments")

            self.pipeline_stats['input_comments'] = len(df)
            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def analyze_input_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze input data characteristics

        Args:
            df: Input DataFrame

        Returns:
            Analysis results dictionary
        """
        self.logger.info("Analyzing input data characteristics...")

        analysis = {
            'total_comments': len(df),
            'comment_length_stats': {
                'mean': df['comment'].str.len().mean(),
                'median': df['comment'].str.len().median(),
                'min': df['comment'].str.len().min(),
                'max': df['comment'].str.len().max(),
                'std': df['comment'].str.len().std()
            },
            'unique_comments': df['comment'].nunique(),
            'duplicate_comments': len(df) - df['comment'].nunique(),
        }

        # App-specific analysis if available
        if 'app_name' in df.columns:
            analysis['apps_count'] = df['app_name'].nunique()
            analysis['comments_per_app'] = df['app_name'].value_counts().to_dict()

        # Rating analysis if available
        if 'rating' in df.columns:
            analysis['rating_distribution'] = df['rating'].value_counts(
            ).sort_index().to_dict()
            analysis['average_rating'] = df['rating'].mean()

        # Language detection (basic)
        persian_char_pattern = r'[آ-ی]'
        analysis['persian_text_ratio'] = df['comment'].str.contains(
            persian_char_pattern, na=False).mean()

        self.logger.info(
            f"Analysis complete. Persian text ratio: {analysis['persian_text_ratio']:.2%}")

        return analysis

    def preprocess_comments(self, df: pd.DataFrame,
                            cleaning_level: str = 'medium',
                            tokenize: bool = True,
                            remove_stopwords: bool = True,
                            stem: bool = False,
                            lemmatize: bool = False) -> pd.DataFrame:
        """
        Preprocess comments using the text cleaner

        Args:
            df: Input DataFrame
            cleaning_level: Level of text cleaning
            tokenize: Whether to tokenize
            remove_stopwords: Whether to remove stopwords
            stem: Whether to stem
            lemmatize: Whether to lemmatize

        Returns:
            DataFrame with processed comments
        """
        self.logger.info(
            f"Starting text preprocessing with level: {cleaning_level}")

        # Reset cleaner statistics
        self.cleaner.reset_statistics()

        # Stage 1: Basic cleaning
        self.logger.info("Stage 1: Basic text cleaning...")
        df['comment_cleaned'] = df['comment'].apply(
            lambda x: self.cleaner.clean_text(x, level=cleaning_level)
        )

        # Filter out empty results
        before_filter = len(df)
        df = df[df['comment_cleaned'].str.len() > 0]
        after_basic_clean = len(df)

        if before_filter > after_basic_clean:
            self.logger.info(
                f"Removed {before_filter - after_basic_clean} empty comments after cleaning")

        # Stage 2: Tokenization and advanced processing
        if tokenize:
            self.logger.info(
                "Stage 2: Tokenization and advanced processing...")

            processed_texts = []
            for text in df['comment_cleaned']:
                processed = self.cleaner.preprocess_text(
                    text,
                    tokenize=True,
                    remove_stopwords=remove_stopwords,
                    stem=stem,
                    lemmatize=lemmatize
                )

                # Join tokens back to string for storage
                if isinstance(processed, list):
                    processed = ' '.join(processed)

                processed_texts.append(processed)

            df['comment_processed'] = processed_texts

            # Filter by length after processing
            before_length_filter = len(df)
            df = df[df['comment_processed'].str.len(
            ) >= self.config.get('min_comment_length', 10)]
            after_length_filter = len(df)

            if before_length_filter > after_length_filter:
                filtered_count = before_length_filter - after_length_filter
                self.logger.info(
                    f"Removed {filtered_count} comments below minimum length")
                self.pipeline_stats['filtered_comments'] = filtered_count

        else:
            df['comment_processed'] = df['comment_cleaned']

        # Update statistics
        self.pipeline_stats['output_comments'] = len(df)
        self.pipeline_stats['processing_stages'] = self.cleaner.get_statistics()

        self.logger.info(f"Preprocessing complete. {len(df)} comments remain.")

        return df

    def create_multiple_versions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create multiple preprocessed versions for comparison

        Args:
            df: Input DataFrame

        Returns:
            Dictionary of different preprocessed versions
        """
        self.logger.info("Creating multiple preprocessing versions...")

        versions = {}

        # Version 1: Light cleaning only
        self.logger.info("Creating version 1: Light cleaning")
        df_light = df.copy()
        df_light['comment_cleaned'] = df_light['comment'].apply(
            lambda x: self.cleaner.clean_text(x, level='light')
        )
        df_light['comment_processed'] = df_light['comment_cleaned']
        versions['light'] = df_light[df_light['comment_processed'].str.len() > 0]

        # Version 2: Medium cleaning + tokenization
        self.logger.info("Creating version 2: Medium cleaning + tokenization")
        versions['medium'] = self.preprocess_comments(
            df.copy(),
            cleaning_level='medium',
            tokenize=True,
            remove_stopwords=True,
            stem=False,
            lemmatize=False
        )

        # Version 3: Heavy cleaning + stemming
        self.logger.info("Creating version 3: Heavy cleaning + stemming")
        versions['heavy_stem'] = self.preprocess_comments(
            df.copy(),
            cleaning_level='heavy',
            tokenize=True,
            remove_stopwords=True,
            stem=True,
            lemmatize=False
        )

        # Version 4: Heavy cleaning + lemmatization
        self.logger.info("Creating version 4: Heavy cleaning + lemmatization")
        versions['heavy_lemma'] = self.preprocess_comments(
            df.copy(),
            cleaning_level='heavy',
            tokenize=True,
            remove_stopwords=True,
            stem=False,
            lemmatize=True
        )

        # Log version statistics
        for version_name, version_df in versions.items():
            self.logger.info(
                f"Version '{version_name}': {len(version_df)} comments")

        return versions

    def save_processed_data(self, processed_data: Dict[str, pd.DataFrame],
                            output_dir: str = None) -> Dict[str, str]:
        """
        Save processed data to files

        Args:
            processed_data: Dictionary of processed DataFrames
            output_dir: Output directory path

        Returns:
            Dictionary of saved file paths
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for version_name, df in processed_data.items():
            output_path = output_dir / f"comments_{version_name}_processed.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            saved_files[version_name] = str(output_path)
            self.logger.info(f"Saved {version_name} version to {output_path}")

        return saved_files

    def generate_processing_report(self, input_analysis: Dict,
                                   saved_files: Dict[str, str]) -> Dict:
        """
        Generate comprehensive processing report

        Args:
            input_analysis: Input data analysis
            saved_files: Dictionary of saved file paths

        Returns:
            Processing report dictionary
        """
        self.pipeline_stats['end_time'] = datetime.now()
        duration = (self.pipeline_stats['end_time'] -
                    self.pipeline_stats['start_time']).total_seconds()

        report = {
            'preprocessing_summary': {
                'start_time': self.pipeline_stats['start_time'].isoformat(),
                'end_time': self.pipeline_stats['end_time'].isoformat(),
                'duration_seconds': duration,
                'input_comments': self.pipeline_stats['input_comments'],
                'output_comments': self.pipeline_stats['output_comments'],
                'filtered_comments': self.pipeline_stats['filtered_comments'],
                'success_rate': self.pipeline_stats['output_comments'] / self.pipeline_stats['input_comments'] if self.pipeline_stats['input_comments'] > 0 else 0
            },
            'input_analysis': input_analysis,
            'processing_configuration': self.config,
            'cleaner_statistics': self.pipeline_stats.get('processing_stages', {}),
            'output_files': saved_files,
            'versions_created': list(saved_files.keys())
        }

        return report

    def save_report(self, report: Dict, output_path: str = None) -> str:
        """
        Save processing report to JSON file

        Args:
            report: Processing report dictionary
            output_path: Output file path

        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = RESULTS_DIR / "reports" / "preprocessing_report.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"Processing report saved to {output_path}")
        return str(output_path)

    def run_pipeline(self, input_path: str,
                     create_versions: bool = True,
                     save_report: bool = True) -> Dict:
        """
        Run the complete preprocessing pipeline

        Args:
            input_path: Path to input CSV file
            create_versions: Whether to create multiple versions
            save_report: Whether to save processing report

        Returns:
            Pipeline results dictionary
        """
        self.pipeline_stats['start_time'] = datetime.now()

        try:
            # Load data
            df = self.load_data(input_path)

            # Analyze input data
            input_analysis = self.analyze_input_data(df)

            # Process data
            if create_versions:
                processed_data = self.create_multiple_versions(df)
            else:
                # Single version processing
                processed_df = self.preprocess_comments(df)
                processed_data = {'default': processed_df}

            # Save processed data
            saved_files = self.save_processed_data(processed_data)

            # Generate and save report
            report = self.generate_processing_report(
                input_analysis, saved_files)

            if save_report:
                report_path = self.save_report(report)
                report['report_path'] = report_path

            self.logger.info("Preprocessing pipeline completed successfully!")

            return {
                'success': True,
                'processed_data': processed_data,
                'saved_files': saved_files,
                'report': report
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Preprocess Persian banking comments"
    )

    parser.add_argument(
        '--input', '-i',
        default=str(FILE_PATHS["raw_comments"]),
        help='Input CSV file path'
    )

    parser.add_argument(
        '--labeled-input', '-l',
        help='Use labeled comments as input'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=str(PROCESSED_DATA_DIR),
        help='Output directory for processed files'
    )

    parser.add_argument(
        '--single-version', '-s',
        action='store_true',
        help='Create only one version instead of multiple'
    )

    parser.add_argument(
        '--cleaning-level',
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Text cleaning level'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating processing report'
    )

    parser.add_argument(
        '--stem',
        action='store_true',
        help='Enable stemming'
    )

    parser.add_argument(
        '--lemmatize',
        action='store_true',
        help='Enable lemmatization'
    )

    args = parser.parse_args()

    print("🧹 Persian Text Preprocessing Pipeline")
    print("=" * 50)

    # Determine input file
    input_path = args.labeled_input if args.labeled_input else args.input

    # Check if input file exists
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"📖 Input file: {input_path}")
    print(f"📁 Output directory: {args.output_dir}")

    # Initialize pipeline
    pipeline = PreprocessingPipeline()

    # Run pipeline
    if args.single_version:
        print("🔄 Running single-version preprocessing...")
        df = pipeline.load_data(input_path)
        input_analysis = pipeline.analyze_input_data(df)

        processed_df = pipeline.preprocess_comments(
            df,
            cleaning_level=args.cleaning_level,
            stem=args.stem,
            lemmatize=args.lemmatize
        )

        saved_files = pipeline.save_processed_data(
            {'default': processed_df},
            args.output_dir
        )

        print(f"✅ Processed {len(processed_df)} comments")
        print(f"📄 Saved to: {saved_files['default']}")

    else:
        print("🔄 Running multi-version preprocessing...")
        results = pipeline.run_pipeline(
            input_path=input_path,
            create_versions=True,
            save_report=not args.no_report
        )

        if results['success']:
            print("✅ Preprocessing completed successfully!")
            print(f"\n📊 Results Summary:")
            print(
                f"  Input comments: {results['report']['preprocessing_summary']['input_comments']}")
            print(
                f"  Output comments: {results['report']['preprocessing_summary']['output_comments']}")
            print(
                f"  Success rate: {results['report']['preprocessing_summary']['success_rate']:.2%}")
            print(
                f"  Duration: {results['report']['preprocessing_summary']['duration_seconds']:.2f} seconds")

            print(f"\n📁 Created versions:")
            for version, path in results['saved_files'].items():
                df_version = pd.read_csv(path)
                print(
                    f"  {version}: {len(df_version)} comments ({Path(path).name})")

            if 'report_path' in results['report']:
                print(
                    f"\n📋 Report saved to: {results['report']['report_path']}")
        else:
            print(f"❌ Preprocessing failed: {results['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
