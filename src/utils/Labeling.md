# OpenAI Comment Labeling System

This document describes the OpenAI-powered sentiment labeling system for Persian banking app comments.

## Overview

The labeling system uses GPT-4o-mini to efficiently label Persian comments with sentiment (positive, negative, neutral) while minimizing costs through batch processing and smart request optimization.

## Key Features

- **Cost-Efficient** : Uses GPT-4o-mini (most affordable OpenAI model)
- **Batch Processing** : Leverages OpenAI Batch API for 50% cost savings on large datasets
- **Async Processing** : Concurrent requests for faster processing
- **Persian Text Support** : Optimized for Persian language sentiment analysis
- **Smart Retry Logic** : Handles API errors and rate limits gracefully
- **Comprehensive Analytics** : Detailed cost tracking and performance metrics
- **Error Handling** : Robust error handling with fallback mechanisms

## Setup

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Get OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Set environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Run Setup Script

```bash
python scripts/setup_labeling.py
```

This will:

- Check dependencies
- Validate API key
- Create necessary directories
- Generate cost estimates
- Test API connection

## Usage

### Quick Start

```bash
# 1. Estimate costs (recommended first step)
python scripts/run_labeling.py --dry-run

# 2. Label a small sample for testing
python scripts/run_labeling.py --sample-size 100

# 3. Label all comments
python scripts/run_labeling.py
```

### Command Line Options

```bash
python scripts/run_labeling.py [OPTIONS]

Options:
  --dry-run              Estimate costs without labeling
  --sample-size N        Label only N comments (for testing)
  --input-file PATH      Custom input CSV file
  --output-file PATH     Custom output CSV file
  --force-concurrent     Use concurrent instead of batch processing
  --help                 Show help message
```

### Examples

```bash
# Cost estimation for full dataset
python scripts/run_labeling.py --dry-run

# Test with 50 comments
python scripts/run_labeling.py --sample-size 50

# Process custom file
python scripts/run_labeling.py --input-file custom_comments.csv --output-file labeled_output.csv

# Force concurrent processing (faster but more expensive)
python scripts/run_labeling.py --force-concurrent
```

## Processing Modes

### 1. Batch API Mode (Recommended for >100 comments)

- **Cost** : 50% cheaper than standard API
- **Speed** : Slower (up to 24 hours)
- **Best for** : Large datasets, cost-sensitive projects
- **Automatic** : Enabled by default for datasets >100 comments

### 2. Concurrent Mode

- **Cost** : Standard API pricing
- **Speed** : Faster (minutes to hours)
- **Best for** : Small datasets, time-sensitive projects
- **Usage** : Add `--force-concurrent` flag

## Cost Analysis

### Pricing (GPT-4o-mini)

- **Input tokens** : $0.000150 per 1K tokens
- **Output tokens** : $0.000600 per 1K tokens
- **Batch API** : 50% discount on both input and output

### Typical Costs

```
Per comment (average):
- Input: ~100 tokens ($0.000015)
- Output: ~5 tokens ($0.000003)
- Total: ~$0.000018 per comment

For 1,000 comments:
- Standard API: ~$0.018
- Batch API: ~$0.009
```

### Cost Estimation

```bash
# Get cost estimate before processing
python scripts/run_labeling.py --dry-run
```

## Output Format

The system generates `data/processed/labeled_comments.csv` with these columns:

```csv
id,app_package,app_name,user,comment,rating,likes,total_votes,date,version_code,from_developer,is_edited,scraped_at,sentiment_label,metadata
```

### New Columns Added:

- **sentiment_label** : positive/negative/neutral
- **metadata** : JSON with token usage, costs, timestamps

## Analysis Tools

### 1. Label Analysis

```bash
# Generate comprehensive analysis report
python src/utils/label_analyzer.py

# Custom input/output
python src/utils/label_analyzer.py --input labeled_comments.csv --output-plots ./figures/
```

### 2. Generated Reports

- **JSON Report** : `results/reports/label_analysis_report.json`
- **Visualizations** : `results/figures/`
- Sentiment distribution pie chart
- Rating vs sentiment analysis
- App-level comparison
- Comment length analysis

### 3. Statistics Tracking

Automatic generation of:

- `data/processed/labeling_stats.json`: Detailed processing statistics
- Token usage and costs
- Processing time and speed
- Success/failure rates

## Configuration

### Main Configuration (`config.py`)

```python
OPENAI_LABELING_CONFIG = {
    "model": "gpt-4.1-mini-2025-04-14",           # Most cost-effective
    "batch_size": 50,                 # Comments per batch
    "max_tokens": 50,                 # Keep responses short
    "temperature": 0.2,               # Low for consistency
    "use_batch_api": True,            # Enable batch processing
    "timeout": 300,                   # 5-minute timeout
    "max_retries": 3,                 # Retry failed requests
}
```

### Prompts (`config.py`)

The system uses carefully crafted Persian prompts:

```python
LABELING_PROMPTS = {
    "system_prompt": """شما یک متخصص تحلیل احساسات متن فارسی هستید...""",
    "user_prompt_template": """لطفاً احساس این نظر درباره اپلیکیشن بانکی را تعیین کنید: "{comment}" """,
}
```

## Advanced Usage

### 1. Programmatic Usage

```python
from src.utils.openai_labeler import OpenAICommentLabeler
import asyncio

async def label_custom_data():
    labeler = OpenAICommentLabeler()
    stats = await labeler.label_comments_from_csv(
        input_csv_path="my_comments.csv",
        output_csv_path="my_labeled_comments.csv"
    )
    print(f"Labeled {stats['summary']['labeled_comments']} comments")
    print(f"Total cost: ${stats['cost_analysis']['total_cost_usd']}")

# Run the labeling
asyncio.run(label_custom_data())
```

### 2. Custom Analysis

```python
from src.utils.label_analyzer import LabelAnalyzer

# Initialize analyzer
analyzer = LabelAnalyzer("data/processed/labeled_comments.csv")

# Generate custom report
report = analyzer.generate_report()

# Create visualizations
plots = analyzer.generate_visualizations("./my_plots/")

# Get app-level insights
app_analysis = analyzer.app_level_analysis()
```

## Monitoring and Logs

### Log Files

- **Main logs** : `logs/comment_labeling.log`
- **Scraper logs** : `logs/cafe_bazaar_scraper.log`

### Real-time Monitoring

```bash
# Monitor labeling progress
tail -f logs/comment_labeling.log

# Check for errors
grep "ERROR" logs/comment_labeling.log
```

## Troubleshooting

### Common Issues

#### 1. API Key Issues

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Test API connection
python scripts/setup_labeling.py
```

#### 2. Rate Limiting

- **Solution** : System automatically handles rate limits
- **Manual** : Reduce batch size in config.py

#### 3. Memory Issues

```bash
# For large datasets, process in chunks
python scripts/run_labeling.py --sample-size 1000
# Repeat with different samples
```

#### 4. Persian Text Encoding

- Ensure CSV files are saved with UTF-8 encoding
- System automatically handles Persian text normalization

### Error Recovery

The system includes robust error handling:

- **API failures** : Automatic retry with exponential backoff
- **Network issues** : Request timeout and retry
- **Invalid responses** : Default to 'neutral' sentiment
- **Batch failures** : Fallback to concurrent processing

## Performance Optimization

### For Large Datasets (>1,000 comments)

1. Use Batch API (automatic)
2. Process during off-peak hours
3. Monitor token usage
4. Consider chunking very large datasets

### For Time-Sensitive Projects

1. Use concurrent mode: `--force-concurrent`
2. Reduce max_tokens in config
3. Process critical apps first

### Cost Optimization

1. Always run `--dry-run` first
2. Use sample testing: `--sample-size 100`
3. Enable batch processing for large datasets
4. Monitor token usage in reports

## Integration with ML Pipeline

The labeled data integrates seamlessly with the rest of your ML pipeline:

```python
# In your model training script
import pandas as pd

# Load labeled data
df = pd.read_csv("data/processed/labeled_comments.csv")

# Prepare for training
X = df['comment']
y = df['sentiment_label']

# Convert to numeric labels
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y_numeric = y.map(label_map)

# Continue with your ML pipeline...
```

## Quality Assurance

### Label Quality Metrics

- **Consistency** : Low temperature (0.2) for consistent labeling
- **Validation** : Manual review of sample predictions
- **Inter-rater reliability** : Compare with human annotations

### Validation Workflow

1. Label small sample manually
2. Compare with OpenAI labels
3. Adjust prompts if needed
4. Re-run on full dataset

## Support and Maintenance

### Regular Tasks

- Monitor API costs
- Update model when newer versions available
- Review and improve prompts based on results
- Backup labeled data

### Updates

- Check OpenAI pricing updates
- Monitor model deprecation notices
- Update dependencies regularly

## FAQ

**Q: How accurate is GPT-4o-mini for Persian sentiment analysis?**
A: Generally 85-90% accuracy for clear sentiments. Test with your specific domain data.

**Q: Can I use my own prompts?**
A: Yes, modify `LABELING_PROMPTS` in config.py.

**Q: How do I handle mixed sentiment comments?**
A: Current system assigns single label. Consider multi-label approach for complex cases.

**Q: What about cost control?**
A: Use `--dry-run` and `--sample-size` for testing. Set up billing alerts in OpenAI dashboard.

**Q: Can I label comments in other languages?**
A: Yes, modify prompts in config.py for your target language.

## License and Credits

This labeling system is part of the Persian Banking Sentiment Analysis project.

**Credits:**

- OpenAI GPT-4o-mini for sentiment classification
- Async processing for efficiency
- Persian language optimization
