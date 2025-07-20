"""
OpenAI-based Persian Comment Sentiment Labeler
Efficiently labels Persian banking app comments using GPT-4o-mini with batch processing
"""

from config import (
    OPENAI_LABELING_CONFIG,
    LABELING_PROMPTS,
    FILE_PATHS,
    API_CONFIG,
    PROCESSED_DATA_DIR
)
import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from openai import AsyncOpenAI
import tiktoken

# Import project configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class LabelingStats:
    """Track labeling statistics and costs"""
    total_comments: int = 0
    labeled_comments: int = 0
    failed_comments: int = 0
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: datetime = None
    end_time: datetime = None
    batch_results: List[Dict] = None

    def __post_init__(self):
        if self.batch_results is None:
            self.batch_results = []


class OpenAICommentLabeler:
    """
    Efficient Persian comment sentiment labeler using OpenAI GPT-4o-mini
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the labeler

        Args:
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.api_key = api_key or API_CONFIG["openai_api_key"]
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.config = OPENAI_LABELING_CONFIG
        self.prompts = LABELING_PROMPTS

        # Initialize tokenizer for cost calculation
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config["model"])
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Fallback

        # Setup logging
        self.setup_logging()

        # Initialize stats
        self.stats = LabelingStats()

    def setup_logging(self):
        """Setup logging for the labeling process"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "comment_labeling.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def estimate_cost(self, comments: List[str]) -> Dict[str, float]:
        """
        Estimate the cost of labeling comments

        Args:
            comments: List of comments to label

        Returns:
            Dictionary with cost estimation details
        """
        total_input_tokens = 0
        estimated_output_tokens = 0

        system_prompt_tokens = len(
            self.tokenizer.encode(self.prompts["system_prompt"]))

        for comment in comments:
            # Calculate input tokens (system prompt + user prompt + comment)
            user_prompt = self.prompts["user_prompt_template"].format(
                comment=comment)
            comment_tokens = len(self.tokenizer.encode(user_prompt))
            total_input_tokens += system_prompt_tokens + comment_tokens

            # Estimate output tokens (simple sentiment label: ~5 tokens)
            estimated_output_tokens += 5

        input_cost = (total_input_tokens / 1000) * \
            self.config["cost_per_1k_tokens"]["input"]
        output_cost = (estimated_output_tokens / 1000) * \
            self.config["cost_per_1k_tokens"]["output"]
        total_cost = input_cost + output_cost

        return {
            "total_input_tokens": total_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_comment": total_cost / len(comments) if comments else 0
        }

    async def label_single_comment(self, comment: str, comment_id: str) -> Tuple[str, str, Dict]:
        """
        Label a single comment using OpenAI API

        Args:
            comment: The comment text to label
            comment_id: Unique identifier for the comment

        Returns:
            Tuple of (comment_id, sentiment_label, metadata)
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system",
                        "content": self.prompts["system_prompt"]},
                    {"role": "user", "content": self.prompts["user_prompt_template"].format(
                        comment=comment)}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                timeout=self.config["timeout"]
            )

            sentiment = response.choices[0].message.content.strip().lower()

            # Validate sentiment label
            if sentiment not in ["positive", "negative", "neutral"]:
                self.logger.warning(
                    f"Invalid sentiment '{sentiment}' for comment {comment_id}, defaulting to 'neutral'")
                sentiment = "neutral"

            # Track token usage
            usage = response.usage
            metadata = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "model": self.config["model"],
                "timestamp": datetime.now().isoformat()
            }

            return comment_id, sentiment, metadata

        except Exception as e:
            self.logger.error(f"Error labeling comment {comment_id}: {str(e)}")
            return comment_id, "neutral", {"error": str(e)}

    async def label_batch_concurrent(self, batch_data: List[Tuple[str, str]]) -> List[Tuple[str, str, Dict]]:
        """
        Label a batch of comments concurrently

        Args:
            batch_data: List of (comment_id, comment_text) tuples

        Returns:
            List of (comment_id, sentiment_label, metadata) tuples
        """
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def label_with_semaphore(comment_id: str, comment: str):
            async with semaphore:
                return await self.label_single_comment(comment, comment_id)

        tasks = [label_with_semaphore(cid, comment)
                 for cid, comment in batch_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                comment_id = batch_data[i][0]
                self.logger.error(
                    f"Exception for comment {comment_id}: {str(result)}")
                processed_results.append(
                    (comment_id, "neutral", {"error": str(result)}))
            else:
                processed_results.append(result)

        return processed_results

    def create_batch_file_content(self, batch_data: List[Tuple[str, str]]) -> List[Dict]:
        """
        Create batch file content for OpenAI Batch API

        Args:
            batch_data: List of (comment_id, comment_text) tuples

        Returns:
            List of batch request objects
        """
        batch_requests = []

        for comment_id, comment in batch_data:
            request = {
                "custom_id": f"comment_{comment_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.config["model"],
                    "messages": [
                        {"role": "system",
                            "content": self.prompts["system_prompt"]},
                        {"role": "user", "content": self.prompts["user_prompt_template"].format(
                            comment=comment)}
                    ],
                    "max_tokens": self.config["max_tokens"],
                    "temperature": self.config["temperature"]
                }
            }
            batch_requests.append(request)

        return batch_requests

    async def process_with_batch_api(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process comments using OpenAI Batch API for maximum cost efficiency

        Args:
            comments_df: DataFrame with comments to label

        Returns:
            DataFrame with sentiment labels added
        """
        self.logger.info(
            "Using OpenAI Batch API for cost-efficient processing...")

        # Prepare batch data
        batch_data = [(str(row['id']), row['comment'])
                      for _, row in comments_df.iterrows()]

        # Split into batches (max 50,000 requests per batch, but we'll use smaller batches)
        # Larger batches for Batch API
        batch_size = min(self.config["batch_size"] * 20, 1000)

        labeled_results = []

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            self.logger.info(
                f"Processing batch {i//batch_size + 1} with {len(batch)} comments...")

            try:
                # Create batch file
                batch_requests = self.create_batch_file_content(batch)

                # Create temporary file for batch
                batch_file_path = PROCESSED_DATA_DIR / \
                    f"batch_{i//batch_size + 1}.jsonl"
                with open(batch_file_path, 'w', encoding='utf-8') as f:
                    for request in batch_requests:
                        f.write(json.dumps(request) + '\n')

                # Upload batch file
                with open(batch_file_path, 'rb') as f:
                    batch_input_file = await self.client.files.create(
                        file=f,
                        purpose="batch"
                    )

                # Create batch job
                batch_job = await self.client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": self.config["batch_description"]}
                )

                # Wait for batch completion
                self.logger.info(
                    f"Batch job created: {batch_job.id}. Waiting for completion...")
                await self._wait_for_batch_completion(batch_job.id)

                # Download and process results
                batch_results = await self._download_batch_results(batch_job.id)
                labeled_results.extend(batch_results)

                # Cleanup
                batch_file_path.unlink(missing_ok=True)

            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
                # Fallback to concurrent processing for this batch
                self.logger.info("Falling back to concurrent processing...")
                concurrent_results = await self.label_batch_concurrent(batch)
                labeled_results.extend(concurrent_results)

        # Merge results back to DataFrame
        return self._merge_results_to_dataframe(comments_df, labeled_results)

    async def _wait_for_batch_completion(self, batch_id: str, max_wait_time: int = 3600):
        """Wait for batch job completion"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            batch_status = await self.client.batches.retrieve(batch_id)

            if batch_status.status == "completed":
                self.logger.info(f"Batch {batch_id} completed successfully!")
                return
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                raise Exception(
                    f"Batch {batch_id} failed with status: {batch_status.status}")

            self.logger.info(
                f"Batch {batch_id} status: {batch_status.status}. Waiting...")
            await asyncio.sleep(30)  # Check every 30 seconds

        raise Exception(
            f"Batch {batch_id} timed out after {max_wait_time} seconds")

    async def _download_batch_results(self, batch_id: str) -> List[Tuple[str, str, Dict]]:
        """Download and process batch results"""
        batch = await self.client.batches.retrieve(batch_id)

        if batch.output_file_id:
            # Download results file
            result_file = await self.client.files.content(batch.output_file_id)

            results = []
            for line in result_file.text.strip().split('\n'):
                if line:
                    result_data = json.loads(line)
                    custom_id = result_data["custom_id"]
                    comment_id = custom_id.replace("comment_", "")

                    if "response" in result_data and result_data["response"]["status_code"] == 200:
                        response_body = result_data["response"]["body"]
                        sentiment = response_body["choices"][0]["message"]["content"].strip(
                        ).lower()

                        # Validate sentiment
                        if sentiment not in ["positive", "negative", "neutral"]:
                            sentiment = "neutral"

                        metadata = {
                            "input_tokens": response_body.get("usage", {}).get("prompt_tokens", 0),
                            "output_tokens": response_body.get("usage", {}).get("completion_tokens", 0),
                            "total_tokens": response_body.get("usage", {}).get("total_tokens", 0),
                            "model": self.config["model"],
                            "timestamp": datetime.now().isoformat(),
                            "batch_id": batch_id
                        }

                        results.append((comment_id, sentiment, metadata))
                    else:
                        # Handle error
                        results.append(
                            (comment_id, "neutral", {"error": "Batch API error"}))

            return results

        raise Exception(f"No output file for batch {batch_id}")

    def _merge_results_to_dataframe(self, original_df: pd.DataFrame, results: List[Tuple[str, str, Dict]]) -> pd.DataFrame:
        """Merge labeling results back to original DataFrame"""
        # Create results DataFrame
        results_df = pd.DataFrame(
            results, columns=['id', 'sentiment_label', 'metadata'])
        results_df['id'] = results_df['id'].astype(str)

        # Merge with original DataFrame
        original_df['id'] = original_df['id'].astype(str)
        merged_df = original_df.merge(results_df, on='id', how='left')

        # Fill missing labels with 'neutral'
        merged_df['sentiment_label'].fillna('neutral', inplace=True)

        # Calculate total token usage and cost
        total_input_tokens = sum([meta.get('input_tokens', 0)
                                 for meta in merged_df['metadata'].fillna({}) if isinstance(meta, dict)])
        total_output_tokens = sum([meta.get('output_tokens', 0)
                                  for meta in merged_df['metadata'].fillna({}) if isinstance(meta, dict)])

        # Update stats
        self.stats.input_tokens = total_input_tokens
        self.stats.output_tokens = total_output_tokens
        self.stats.total_cost = (total_input_tokens / 1000) * self.config["cost_per_1k_tokens"]["input"] + \
            (total_output_tokens / 1000) * \
            self.config["cost_per_1k_tokens"]["output"]
        self.stats.labeled_comments = len(
            merged_df[merged_df['sentiment_label'].notna()])
        self.stats.failed_comments = len(
            merged_df[merged_df['sentiment_label'].isna()])

        return merged_df

    async def label_comments_from_csv(self, input_csv_path: str, output_csv_path: str) -> Dict:
        """
        Main method to label comments from CSV file

        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to save labeled CSV file

        Returns:
            Dictionary with labeling statistics
        """
        self.stats.start_time = datetime.now()

        try:
            # Load comments
            self.logger.info(f"Loading comments from {input_csv_path}")
            df = pd.read_csv(input_csv_path)

            # Validate required columns
            if 'comment' not in df.columns:
                raise ValueError("CSV file must contain 'comment' column")

            # Clean and filter comments
            df = df.dropna(subset=['comment'])
            df = df[df['comment'].str.len() > 10]  # Minimum comment length
            df = df.reset_index(drop=True)

            self.stats.total_comments = len(df)
            self.logger.info(
                f"Found {self.stats.total_comments} valid comments to label")

            # Estimate cost
            cost_estimate = self.estimate_cost(df['comment'].tolist())
            self.logger.info(
                f"Estimated cost: ${cost_estimate['total_cost']:.4f} ({cost_estimate['cost_per_comment']:.6f} per comment)")

            # Process comments
            if self.config["use_batch_api"] and self.stats.total_comments > 50:
                labeled_df = await self.process_with_batch_api(df)
            else:
                # Use concurrent processing for smaller datasets
                batch_data = [(str(row['id']), row['comment'])
                              for _, row in df.iterrows()]
                results = await self.label_batch_concurrent(batch_data)
                labeled_df = self._merge_results_to_dataframe(df, results)

            # Save results
            labeled_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            self.logger.info(f"Labeled comments saved to {output_csv_path}")

            # Save labeling statistics
            self.stats.end_time = datetime.now()
            stats_dict = self._generate_stats_report()

            stats_file = PROCESSED_DATA_DIR / "labeling_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"Labeling completed! Stats saved to {stats_file}")
            return stats_dict

        except Exception as e:
            self.logger.error(f"Error in labeling process: {str(e)}")
            raise

    def _generate_stats_report(self) -> Dict:
        """Generate comprehensive statistics report"""
        duration = (self.stats.end_time -
                    self.stats.start_time).total_seconds() if self.stats.end_time else 0

        return {
            "summary": {
                "total_comments": self.stats.total_comments,
                "labeled_comments": self.stats.labeled_comments,
                "failed_comments": self.stats.failed_comments,
                "success_rate": self.stats.labeled_comments / self.stats.total_comments if self.stats.total_comments > 0 else 0
            },
            "cost_analysis": {
                "total_cost_usd": round(self.stats.total_cost, 4),
                "input_tokens": self.stats.input_tokens,
                "output_tokens": self.stats.output_tokens,
                "cost_per_comment": round(self.stats.total_cost / self.stats.total_comments, 6) if self.stats.total_comments > 0 else 0
            },
            "timing": {
                "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None,
                "end_time": self.stats.end_time.isoformat() if self.stats.end_time else None,
                "duration_seconds": round(duration, 2),
                "comments_per_second": round(self.stats.labeled_comments / duration, 2) if duration > 0 else 0
            },
            "configuration": {
                "model": self.config["model"],
                "batch_size": self.config["batch_size"],
                "use_batch_api": self.config["use_batch_api"],
                "temperature": self.config["temperature"]
            }
        }


async def main():
    """Main execution function"""
    # Initialize labeler
    labeler = OpenAICommentLabeler()

    # Paths
    input_path = FILE_PATHS["raw_comments"]
    output_path = FILE_PATHS["labeled_comments"]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run labeling
    stats = await labeler.label_comments_from_csv(
        input_csv_path=str(input_path),
        output_csv_path=str(output_path)
    )

    print("\n=== Labeling Complete ===")
    print(f"Total comments: {stats['summary']['total_comments']}")
    print(f"Successfully labeled: {stats['summary']['labeled_comments']}")
    print(f"Success rate: {stats['summary']['success_rate']:.2%}")
    print(f"Total cost: ${stats['cost_analysis']['total_cost_usd']}")
    print(f"Cost per comment: ${stats['cost_analysis']['cost_per_comment']}")
    print(f"Duration: {stats['timing']['duration_seconds']} seconds")

if __name__ == "__main__":
    asyncio.run(main())
