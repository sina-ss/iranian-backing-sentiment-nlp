# Cafe Bazaar Scraper Usage Guide

## Overview

The Cafe Bazaar scraper is designed to collect Persian user comments from banking applications on the Cafe Bazaar platform. It uses the official Cafe Bazaar API to gather reviews and ratings.

## Features

- ✅ Scrapes comments from multiple banking apps
- ✅ Handles pagination automatically
- ✅ Includes rate limiting and retry logic
- ✅ Saves data in CSV format with Persian text support
- ✅ Comprehensive logging and error handling
- ✅ Respects API limits and includes delays
- ✅ Collects metadata (ratings, likes, dates, etc.)

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Create project structure:**

```bash
python scripts/setup_project.py
```

## Usage

### Basic Usage - Scrape All Banking Apps

```bash
python scripts/run_scraper.py
```

### Advanced Usage Options

**Specify maximum comments per app:**

```bash
python scripts/run_scraper.py --max-comments 100
```

**Use custom banking apps file:**

```bash
python scripts/run_scraper.py --apps-file data/raw/my_apps.json
```

**Specify output file:**

```bash
python scripts/run_scraper.py --output-file data/raw/my_comments.csv
```

**Scrape a single app:**

```bash
python scripts/run_scraper.py --single-app com.pmb.mobile
```

**Adjust delay between requests:**

```bash
python scripts/run_scraper.py --delay 3.0
```

### Programmatic Usage

```python
from src.data_collection.cafe_bazaar_scraper import CafeBazaarScraper

# Create scraper instance
scraper = CafeBazaarScraper()

# Scrape all apps
scraper.scrape_all_apps()

# Or scrape a single app
app_info = {
    "package_name": "com.pmb.mobile",
    "app_name": "Mellat Bank"
}
comments = scraper.scrape_app_comments(app_info, max_comments=100)
```

## Output Format

The scraper saves comments in CSV format with the following columns:

| Column           | Description                            |
| ---------------- | -------------------------------------- |
| `id`             | Unique comment ID from Cafe Bazaar     |
| `app_package`    | App package name                       |
| `app_name`       | Human-readable app name                |
| `user`           | Username of the reviewer               |
| `comment`        | The actual Persian comment text        |
| `rating`         | Star rating (1-5)                      |
| `likes`          | Number of likes on the comment         |
| `total_votes`    | Total votes on the comment             |
| `date`           | Persian date of the comment            |
| `version_code`   | App version when comment was made      |
| `from_developer` | Boolean - is this a developer response |
| `is_edited`      | Boolean - was the comment edited       |
| `scraped_at`     | Timestamp when data was collected      |

## Configuration

Key settings in `config.py`:

```python
CAFE_BAZAAR_CONFIG = {
    "api_url": "https://api.cafebazaar.ir/rest-v1/process/ReviewRequest",
    "max_comments_per_app": 200,
    "comments_per_request": 10,  # API limitation
    "delay_between_requests": 2,
    "max_retries": 3,
    "timeout": 30
}
```

## API Limitations

- Maximum 10 comments per request
- Rate limiting recommended (2+ seconds between requests)
- Some apps may have fewer than expected comments
- API may occasionally return empty responses

## Error Handling

The scraper includes comprehensive error handling:

- **Network errors:** Automatic retries with exponential backoff
- **API errors:** Logged and skipped, scraping continues
- **Data errors:** Invalid responses are logged and skipped
- **Rate limiting:** Built-in delays prevent being blocked

## Logging

Logs are saved to:

- `cafe_bazaar_scraper.log` - Detailed scraping logs
- `data/raw/scraping_log.json` - Summary statistics
- `data/raw/failed_apps.json` - Apps that failed to scrape

## File Structure

```
data/raw/
├── banking_apps_list.json    # List of apps to scrape
├── cafe_bazaar_comments.csv  # Collected comments
├── scraping_log.json         # Scraping statistics
└── failed_apps.json          # Failed apps list
```

## Troubleshooting

**Common Issues:**

1. **No comments collected:**
   - Check if app package name is correct
   - Some apps may have very few reviews
   - Check network connectivity
2. **API errors:**
   - Increase delay between requests
   - Check if Cafe Bazaar API is accessible
   - Verify headers and payload format
3. **Rate limiting:**
   - Increase `delay_between_requests` in config
   - Use smaller batch sizes
4. **Persian text encoding:**
   - Ensure UTF-8 encoding is used
   - Check if Persian fonts are installed

**Debug mode:**

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Next Steps

After collecting data:

1. **Data cleaning:** Use `src/preprocessing/persian_cleaner.py`
2. **Labeling:** Use `src/utils/data_labeling_tool.py`
3. **Analysis:** Run Jupyter notebooks in `notebooks/`

## Legal and Ethical Considerations

- Respect Cafe Bazaar's terms of service
- Use reasonable delays to avoid overloading servers
- Only collect publicly available review data
- Ensure compliance with data protection regulations
- Give proper attribution to Cafe Bazaar as data source
