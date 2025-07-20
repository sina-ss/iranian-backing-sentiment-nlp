#!/usr/bin/env python3
"""
Script to run Cafe Bazaar scraper with different options
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the src directory specifically
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Verify the path exists
scraper_file = project_root / "src" / \
    "data_collection" / "cafe_bazaar_scraper.py"
if not scraper_file.exists():
    print(f"Error: Scraper file not found at {scraper_file}")
    print("Please make sure the file structure is correct.")
    sys.exit(1)

try:
    from src.data_collection.cafe_bazaar_scraper import CafeBazaarScraper
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")

    # Try alternative import
    try:
        sys.path.insert(0, str(project_root / "src" / "data_collection"))
        from cafe_bazaar_scraper import CafeBazaarScraper
        print("Successfully imported using alternative method")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run Cafe Bazaar scraper for banking apps')

    parser.add_argument(
        '--apps-file',
        type=str,
        help='Path to banking apps JSON file',
        default=None
    )

    parser.add_argument(
        '--output-file',
        type=str,
        help='Output CSV file path',
        default=None
    )

    parser.add_argument(
        '--max-comments',
        type=int,
        help='Maximum comments per app (default: 200)',
        default=200
    )

    parser.add_argument(
        '--single-app',
        type=str,
        help='Package name of a single app to scrape',
        default=None
    )

    parser.add_argument(
        '--delay',
        type=float,
        help='Delay between requests in seconds (default: 2.0)',
        default=2.0
    )

    args = parser.parse_args()

    # Create scraper instance
    scraper = CafeBazaarScraper(args.output_file)

    # Update delay if specified
    if args.delay:
        scraper.config["delay_between_requests"] = args.delay

    if args.single_app:
        # Scrape single app
        print(f"Scraping single app: {args.single_app}")

        # Create app info
        app_info = {
            "package_name": args.single_app,
            "app_name": args.single_app.split('.')[-1].title()
        }

        # Scrape comments
        comments = scraper.scrape_app_comments(app_info, args.max_comments)

        if comments:
            scraper.all_comments = comments
            scraper.save_results()
            print(f"Collected {len(comments)} comments from {args.single_app}")
        else:
            print(f"No comments collected from {args.single_app}")
    else:
        # Scrape all apps
        print("Starting Cafe Bazaar scraping for all banking apps...")
        scraper.scrape_all_apps(args.apps_file, args.max_comments)

    print("Scraping completed!")


if __name__ == "__main__":
    main()
