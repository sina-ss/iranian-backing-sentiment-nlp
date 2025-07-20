#!/usr/bin/env python3
"""
Script to run the OpenAI comment labeling process
Usage: python scripts/run_labeling.py [--dry-run] [--sample-size N]
"""

import argparse
import asyncio
import sys
from pathlib import Path
import pandas as pd
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# Also add the src directory specifically
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from config import FILE_PATHS, PROCESSED_DATA_DIR
    from src.utils.openai_labeler import OpenAICommentLabeler
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Label Persian banking comments using OpenAI GPT-4o-mini"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Estimate costs without actually labeling comments'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of comments to sample for labeling (useful for testing)'
    )

    parser.add_argument(
        '--input-file',
        type=str,
        default=str(FILE_PATHS["raw_comments"]),
        help='Input CSV file path'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=str(FILE_PATHS["labeled_comments"]),
        help='Output CSV file path'
    )

    parser.add_argument(
        '--force-concurrent',
        action='store_true',
        help='Force concurrent processing instead of batch API'
    )

    return parser.parse_args()


async def run_dry_run(input_path: str, sample_size: int = None):
    """Run cost estimation without actual labeling"""
    print("🧮 Running cost estimation...")

    # Load data
    df = pd.read_csv(input_path)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Clean comments
    df = df.dropna(subset=['comment'])
    df = df[df['comment'].str.len() > 10]

    print(f"📊 Found {len(df)} valid comments")

    # Initialize labeler and estimate cost
    labeler = OpenAICommentLabeler()
    cost_estimate = labeler.estimate_cost(df['comment'].tolist())

    print("\n💰 Cost Estimation:")
    print(f"  Total input tokens: {cost_estimate['total_input_tokens']:,}")
    print(
        f"  Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
    print(f"  Input cost: ${cost_estimate['input_cost']:.4f}")
    print(f"  Output cost: ${cost_estimate['output_cost']:.4f}")
    print(f"  Total estimated cost: ${cost_estimate['total_cost']:.4f}")
    print(f"  Cost per comment: ${cost_estimate['cost_per_comment']:.6f}")

    # Additional insights
    if len(df) > 1000:
        print(f"\n📈 Batch Processing Benefits:")
        print(f"  Large dataset detected ({len(df)} comments)")
        print(f"  Recommended: Use Batch API for 50% cost savings")
        print(f"  Estimated savings: ${cost_estimate['total_cost'] * 0.5:.4f}")

    return cost_estimate


async def run_labeling(args):
    """Run the actual labeling process"""
    print("🚀 Starting comment labeling process...")

    # Load and prepare data
    df = pd.read_csv(args.input_file)
    original_count = len(df)

    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(
            f"📝 Using sample of {len(df)} comments from {original_count} total")

    # Initialize labeler with custom config if needed
    labeler = OpenAICommentLabeler()

    if args.force_concurrent:
        labeler.config["use_batch_api"] = False
        print("⚡ Forced concurrent processing mode")

    # Create temp file for sample if needed
    input_path = args.input_file
    if args.sample_size:
        temp_file = PROCESSED_DATA_DIR / \
            f"sample_{args.sample_size}_comments.csv"
        df.to_csv(temp_file, index=False, encoding='utf-8')
        input_path = str(temp_file)
        print(f"💾 Created temporary sample file: {temp_file}")

    try:
        # Run labeling
        stats = await labeler.label_comments_from_csv(
            input_csv_path=input_path,
            output_csv_path=args.output_file
        )

        # Display results
        print("\n✅ Labeling Complete!")
        print(f"📊 Statistics:")
        print(
            f"  Total comments processed: {stats['summary']['total_comments']}")
        print(
            f"  Successfully labeled: {stats['summary']['labeled_comments']}")
        print(f"  Failed: {stats['summary']['failed_comments']}")
        print(f"  Success rate: {stats['summary']['success_rate']:.2%}")

        print(f"\n💸 Cost Analysis:")
        print(f"  Total cost: ${stats['cost_analysis']['total_cost_usd']:.4f}")
        print(
            f"  Cost per comment: ${stats['cost_analysis']['cost_per_comment']:.6f}")
        print(f"  Input tokens: {stats['cost_analysis']['input_tokens']:,}")
        print(f"  Output tokens: {stats['cost_analysis']['output_tokens']:,}")

        print(f"\n⏱️ Performance:")
        print(f"  Duration: {stats['timing']['duration_seconds']:.2f} seconds")
        print(
            f"  Speed: {stats['timing']['comments_per_second']:.2f} comments/second")

        print(f"\n📁 Output saved to: {args.output_file}")

        # Quick sentiment distribution
        labeled_df = pd.read_csv(args.output_file)
        sentiment_dist = labeled_df['sentiment_label'].value_counts()
        print(f"\n📈 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(labeled_df)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

        # Cleanup temp file
        if args.sample_size and Path(input_path).exists():
            Path(input_path).unlink()
            print(f"🗑️ Cleaned up temporary file")

    except Exception as e:
        print(f"❌ Error during labeling: {str(e)}")
        if args.sample_size and Path(input_path).exists():
            Path(input_path).unlink()
        raise


def check_prerequisites():
    """Check if all prerequisites are met"""
    issues = []

    # Check if input file exists
    if not Path(FILE_PATHS["raw_comments"]).exists():
        issues.append(f"Input file not found: {FILE_PATHS['raw_comments']}")

    # Check OpenAI API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        issues.append(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    # Check output directory
    output_dir = Path(FILE_PATHS["labeled_comments"]).parent
    if not output_dir.exists():
        print(f"📁 Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    if issues:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


async def main():
    """Main execution function"""
    print("🏷️ OpenAI Persian Comment Labeler")
    print("=" * 50)

    args = parse_arguments()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Display input file info
    input_df = pd.read_csv(args.input_file)
    print(f"📖 Input file: {args.input_file}")
    print(f"📊 Total comments: {len(input_df)}")
    print(f"📝 Valid comments: {len(input_df.dropna(subset=['comment']))}")

    if args.dry_run:
        # Run cost estimation only
        await run_dry_run(args.input_file, args.sample_size)
    else:
        # Confirm before proceeding with actual labeling
        if not args.sample_size:
            print(f"\n⚠️ You are about to label {len(input_df)} comments.")
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Aborted by user")
                sys.exit(0)

        # Run labeling
        await run_labeling(args)

if __name__ == "__main__":
    asyncio.run(main())
