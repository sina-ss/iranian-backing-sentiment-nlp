"""
Cafe Bazaar Scraper for Persian Banking App Comments
Collects user reviews and comments from Cafe Bazaar API
"""

from config import CAFE_BAZAAR_CONFIG, FILE_PATHS, LOGGING_CONFIG
import json
import time
import requests
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import random
from pathlib import Path
import sys

# Add the project root to the path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))


class CafeBazaarScraper:
    """
    Scraper class for collecting Persian banking app comments from Cafe Bazaar
    """

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize the scraper

        Args:
            output_file: Path to save collected comments
        """
        self.config = CAFE_BAZAAR_CONFIG
        self.session = requests.Session()
        self.session.headers.update(self.config["headers"])

        # Setup logging
        self.setup_logging()

        # Data storage
        self.output_file = output_file or FILE_PATHS["raw_comments"]
        self.all_comments = []
        self.failed_apps = []
        self.scraping_stats = {
            "start_time": None,
            "end_time": None,
            "total_apps": 0,
            "successful_apps": 0,
            "failed_apps": 0,
            "total_comments": 0,
            "apps_stats": {}
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(
                    "cafe_bazaar_scraper.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_banking_apps(self, apps_file: Optional[str] = None) -> List[Dict]:
        """
        Load banking apps list from JSON file

        Args:
            apps_file: Path to banking apps JSON file

        Returns:
            List of banking app dictionaries
        """
        apps_file = apps_file or FILE_PATHS["banking_apps_list"]

        try:
            with open(apps_file, 'r', encoding='utf-8') as f:
                apps = json.load(f)
            self.logger.info(
                f"Loaded {len(apps)} banking apps from {apps_file}")
            return apps
        except FileNotFoundError:
            self.logger.error(f"Banking apps file not found: {apps_file}")
            # Return a default list if file doesn't exist
            return self.get_default_banking_apps()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file: {e}")
            return []

    def get_default_banking_apps(self) -> List[Dict]:
        """
        Return default banking apps if file doesn't exist

        Returns:
            List of default banking app dictionaries
        """
        default_apps = [
            {"package_name": "ir.tejaratbank.tata.mobile.android.tejarat",
                "app_name": "Tejarat Bank"},
            {"package_name": "com.pmb.mobile", "app_name": "Mellat Bank"},
            {"package_name": "com.refahbank.dpi.android", "app_name": "Refah Bank"},
            {"package_name": "com.bmi.mobilebanking", "app_name": "Melli Bank"},
            {"package_name": "com.saderat.saderatemobile",
                "app_name": "Saderat Bank"},
            {"package_name": "com.saman.mobile", "app_name": "Saman Bank"},
            {"package_name": "com.parsian.pec.mobile", "app_name": "Parsian Bank"},
            {"package_name": "com.eghtesadnovin.enhbank",
                "app_name": "Eghtesad Novin Bank"},
            {"package_name": "com.karafarin.mobile", "app_name": "Karafarin Bank"},
            {"package_name": "com.dey.mobile", "app_name": "Dey Bank"}
        ]

        # Save default apps to file
        with open(FILE_PATHS["banking_apps_list"], 'w', encoding='utf-8') as f:
            json.dump(default_apps, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Created default banking apps file with {len(default_apps)} apps")
        return default_apps

    def create_request_payload(self, package_name: str, start: int = 0, end: int = 10) -> Dict:
        """
        Create request payload for Cafe Bazaar API

        Args:
            package_name: App package name
            start: Start index for pagination
            end: End index for pagination

        Returns:
            Request payload dictionary
        """
        return {
            "properties": self.config["properties"],
            "singleRequest": {
                "reviewRequest": {
                    "packageName": package_name,
                    "start": start,
                    "end": end
                }
            }
        }

    def make_request(self, payload: Dict) -> Optional[Dict]:
        """
        Make API request to Cafe Bazaar

        Args:
            payload: Request payload

        Returns:
            API response or None if failed
        """
        try:
            response = self.session.post(
                self.config["api_url"],
                json=payload,
                timeout=self.config["timeout"]
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            return None

    def extract_comments_from_response(self, response: Dict, app_info: Dict) -> List[Dict]:
        """
        Extract comments from API response

        Args:
            response: API response
            app_info: App information

        Returns:
            List of comment dictionaries
        """
        comments = []

        try:
            reviews = response.get("singleReply", {}).get(
                "reviewReply", {}).get("reviews", [])

            for review in reviews:
                comment_data = {
                    "id": review.get("id"),
                    "app_package": app_info["package_name"],
                    "app_name": app_info["app_name"],
                    "user": review.get("user"),
                    "comment": review.get("comment"),
                    "rating": review.get("rate"),
                    "likes": review.get("likes", 0),
                    "total_votes": review.get("total", 0),
                    "date": review.get("date"),
                    "version_code": review.get("versionCode"),
                    "from_developer": review.get("fromDeveloper", False),
                    "is_edited": review.get("isEdited", False),
                    "scraped_at": datetime.now().isoformat()
                }
                comments.append(comment_data)

        except (KeyError, TypeError) as e:
            self.logger.error(f"Error extracting comments: {e}")

        return comments

    def get_next_page_cursor(self, response: Dict) -> Optional[str]:
        """
        Extract next page cursor for pagination

        Args:
            response: API response

        Returns:
            Next page cursor or None
        """
        try:
            return response.get("singleReply", {}).get("reviewReply", {}).get("nextPageCursor")
        except (KeyError, TypeError):
            return None

    def scrape_app_comments(self, app_info: Dict, max_comments: Optional[int] = None) -> List[Dict]:
        """
        Scrape comments for a single app

        Args:
            app_info: App information dictionary
            max_comments: Maximum comments to collect

        Returns:
            List of collected comments
        """
        package_name = app_info["package_name"]
        app_name = app_info["app_name"]
        max_comments = max_comments or self.config["max_comments_per_app"]

        self.logger.info(f"Starting to scrape {app_name} ({package_name})")

        comments = []
        start_index = 0
        comments_per_request = self.config["comments_per_request"]

        while len(comments) < max_comments:
            # Create payload for current page
            end_index = start_index + comments_per_request
            payload = self.create_request_payload(
                package_name, start_index, end_index)

            # Make request with retry logic
            response = None
            for attempt in range(self.config["max_retries"]):
                response = self.make_request(payload)
                if response:
                    break

                if attempt < self.config["max_retries"] - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(
                        f"Request failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            if not response:
                self.logger.error(
                    f"Failed to get response after {self.config['max_retries']} attempts")
                break

            # Extract comments from response
            page_comments = self.extract_comments_from_response(
                response, app_info)

            if not page_comments:
                self.logger.info(f"No more comments found for {app_name}")
                break

            comments.extend(page_comments)
            self.logger.info(
                f"Collected {len(page_comments)} comments from {app_name} (Total: {len(comments)})")

            # Check if we have enough comments
            if len(comments) >= max_comments:
                comments = comments[:max_comments]
                break

            # Update start index for next page
            start_index = end_index

            # Add delay between requests
            delay = self.config["delay_between_requests"]
            # Add some randomness to avoid being detected as a bot
            delay += random.uniform(0.5, 1.5)
            time.sleep(delay)

        self.logger.info(
            f"Finished scraping {app_name}: {len(comments)} comments collected")
        return comments

    def scrape_all_apps(self, apps_file: Optional[str] = None, max_comments_per_app: Optional[int] = None) -> None:
        """
        Scrape comments from all banking apps

        Args:
            apps_file: Path to banking apps file
            max_comments_per_app: Maximum comments per app
        """
        self.scraping_stats["start_time"] = datetime.now().isoformat()

        # Load banking apps
        banking_apps = self.load_banking_apps(apps_file)
        if not banking_apps:
            self.logger.error("No banking apps to scrape")
            return

        self.scraping_stats["total_apps"] = len(banking_apps)

        # Scrape each app
        for i, app_info in enumerate(banking_apps, 1):
            app_name = app_info["app_name"]
            package_name = app_info["package_name"]

            self.logger.info(
                f"Processing app {i}/{len(banking_apps)}: {app_name}")

            try:
                # Scrape comments for this app
                app_comments = self.scrape_app_comments(
                    app_info, max_comments_per_app)

                if app_comments:
                    self.all_comments.extend(app_comments)
                    self.scraping_stats["successful_apps"] += 1
                    self.scraping_stats["apps_stats"][app_name] = {
                        "package_name": package_name,
                        "comments_collected": len(app_comments),
                        "status": "success"
                    }
                    self.logger.info(
                        f"Successfully collected {len(app_comments)} comments from {app_name}")
                else:
                    self.failed_apps.append({
                        "app_name": app_name,
                        "package_name": package_name,
                        "error": "No comments collected"
                    })
                    self.scraping_stats["failed_apps"] += 1
                    self.scraping_stats["apps_stats"][app_name] = {
                        "package_name": package_name,
                        "comments_collected": 0,
                        "status": "failed",
                        "error": "No comments collected"
                    }

            except Exception as e:
                self.logger.error(f"Error scraping {app_name}: {e}")
                self.failed_apps.append({
                    "app_name": app_name,
                    "package_name": package_name,
                    "error": str(e)
                })
                self.scraping_stats["failed_apps"] += 1
                self.scraping_stats["apps_stats"][app_name] = {
                    "package_name": package_name,
                    "comments_collected": 0,
                    "status": "failed",
                    "error": str(e)
                }

        self.scraping_stats["end_time"] = datetime.now().isoformat()
        self.scraping_stats["total_comments"] = len(self.all_comments)

        # Save results
        self.save_results()
        self.save_scraping_log()

        # Print summary
        self.print_summary()

    def save_results(self) -> None:
        """Save collected comments to CSV file"""
        if not self.all_comments:
            self.logger.warning("No comments to save")
            return

        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.all_comments)

            # Ensure output directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            self.logger.info(
                f"Saved {len(self.all_comments)} comments to {self.output_file}")

            # Save failed apps if any
            if self.failed_apps:
                failed_file = FILE_PATHS["failed_apps"]
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_apps, f,
                              ensure_ascii=False, indent=2)
                self.logger.info(
                    f"Saved {len(self.failed_apps)} failed apps to {failed_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def save_scraping_log(self) -> None:
        """Save scraping statistics"""
        try:
            log_file = FILE_PATHS["scraping_log"]
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.scraping_stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved scraping log to {log_file}")
        except Exception as e:
            self.logger.error(f"Error saving scraping log: {e}")

    def print_summary(self) -> None:
        """Print scraping summary"""
        stats = self.scraping_stats

        print("\n" + "="*60)
        print("CAFE BAZAAR SCRAPING SUMMARY")
        print("="*60)
        print(f"Total Apps Processed: {stats['total_apps']}")
        print(f"Successful Apps: {stats['successful_apps']}")
        print(f"Failed Apps: {stats['failed_apps']}")
        print(f"Total Comments Collected: {stats['total_comments']}")

        if stats['successful_apps'] > 0:
            avg_comments = stats['total_comments'] / stats['successful_apps']
            print(f"Average Comments per App: {avg_comments:.1f}")

        print("\nApp-wise Results:")
        print("-" * 40)
        for app_name, app_stats in stats['apps_stats'].items():
            status_emoji = "✅" if app_stats['status'] == 'success' else "❌"
            print(
                f"{status_emoji} {app_name}: {app_stats['comments_collected']} comments")

        print("="*60)


def main():
    """Main function to run the scraper"""
    # Create scraper instance
    scraper = CafeBazaarScraper()

    # Start scraping
    print("Starting Cafe Bazaar scraping...")
    scraper.scrape_all_apps()

    print("\nScraping completed!")


if __name__ == "__main__":
    main()
