#!/usr/bin/env python3
"""
Setup script for OpenAI labeling system
Checks dependencies, validates API keys, and prepares the environment
"""

import os
import sys
import subprocess
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from config import FILE_PATHS
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def check_package_installation(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")

    requirements_file = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(
                requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        ("openai", "OpenAI API client"),
        ("tiktoken", "Token counting for OpenAI"),
        ("aiohttp", "Async HTTP client"),
        ("pandas", "Data manipulation"),
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical visualization")
    ]

    missing_packages = []

    for package, description in required_packages:
        if check_package_installation(package):
            print(f"✅ {package} ({description})")
        else:
            print(f"❌ {package} ({description}) - MISSING")
            missing_packages.append(package)

    return len(missing_packages) == 0, missing_packages


def setup_api_key():
    """Setup OpenAI API key"""
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        # Validate key format
        if api_key.startswith("sk-") and len(api_key) > 20:
            print("✅ OpenAI API key found and appears valid")
            return True
        else:
            print("⚠️ OpenAI API key found but format seems incorrect")

    print("\n🔑 OpenAI API Key Setup")
    print("You need an OpenAI API key to use the labeling system.")
    print("Get your API key from: https://platform.openai.com/api-keys")

    while True:
        user_input = input(
            "\nEnter your OpenAI API key (or 'skip' to continue): ").strip()

        if user_input.lower() == 'skip':
            print(
                "⚠️ Skipping API key setup. Set OPENAI_API_KEY environment variable before running labeling.")
            return False

        if user_input.startswith("sk-") and len(user_input) > 20:
            # Save to .env file
            env_file = Path(__file__).parent.parent / ".env"
            with open(env_file, "a") as f:
                f.write(f"\nOPENAI_API_KEY={user_input}\n")
            print(f"✅ API key saved to {env_file}")
            print("🔄 Restart your terminal or source the .env file to use the key")
            return True
        else:
            print(
                "❌ Invalid API key format. Should start with 'sk-' and be longer than 20 characters.")


def test_api_connection():
    """Test OpenAI API connection"""
    try:
        import openai
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ No API key found, skipping connection test")
            return False

        client = OpenAI(api_key=api_key)

        # Test with a minimal request
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )

        print("✅ OpenAI API connection successful")
        return True

    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False


def setup_directories():
    """Create necessary directories"""
    project_root = Path(__file__).parent.parent

    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "models/saved_models",
        "models/checkpoints",
        "results/figures",
        "results/reports",
        "results/metrics",
        "logs",
        "src/utils"
    ]

    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 {dir_path}")

    print("✅ Directory structure created")


def create_sample_env_file():
    """Create a sample .env file"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Other API Keys (optional)
HUGGINGFACE_TOKEN=your_huggingface_token_here
GOOGLE_TRANSLATE_KEY=your_google_translate_key_here

# Project Configuration
PERSIAN_SENTIMENT_DEBUG=false
PERSIAN_SENTIMENT_LOG_LEVEL=INFO
"""

    env_file = Path(__file__).parent.parent / ".env.example"
    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"✅ Sample environment file created: {env_file}")


def check_input_data():
    """Check if input data file exists"""

    input_file = FILE_PATHS["raw_comments"]

    if input_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(input_file)
            print(f"✅ Input data found: {len(df)} comments in {input_file}")

            # Check required columns
            required_columns = ['id', 'comment']
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"⚠️ Missing required columns: {missing_columns}")
                return False

            return True

        except Exception as e:
            print(f"❌ Error reading input data: {e}")
            return False
    else:
        print(f"⚠️ Input data not found: {input_file}")
        print("   Make sure to run the scraper first to collect comments")
        return False


def generate_cost_estimate():
    """Generate a quick cost estimate"""
    try:
        import pandas as pd

        input_file = FILE_PATHS["raw_comments"]
        if not input_file.exists():
            return

        df = pd.read_csv(input_file)
        valid_comments = df.dropna(subset=['comment'])
        valid_comments = valid_comments[valid_comments['comment'].str.len(
        ) > 10]

        # Rough estimate: ~100 tokens per comment (input) + 5 tokens output
        estimated_input_tokens = len(valid_comments) * 100
        estimated_output_tokens = len(valid_comments) * 5

        # GPT-4o-mini pricing
        input_cost = (estimated_input_tokens / 1000) * 0.000150
        output_cost = (estimated_output_tokens / 1000) * 0.000600
        total_cost = input_cost + output_cost

        print(f"\n💰 Estimated Labeling Cost:")
        print(f"   Comments to label: {len(valid_comments)}")
        print(f"   Estimated total cost: ${total_cost:.4f}")
        print(f"   Cost per comment: ${total_cost/len(valid_comments):.6f}")

        if len(valid_comments) > 1000:
            batch_savings = total_cost * 0.5
            print(f"   Potential savings with Batch API: ${batch_savings:.4f}")

    except Exception as e:
        print(f"⚠️ Could not generate cost estimate: {e}")


def main():
    """Main setup function"""
    print("🚀 Persian Banking Sentiment Analysis - Labeling System Setup")
    print("=" * 60)

    success = True

    # Check Python version
    if not check_python_version():
        success = False

    print("\n📦 Checking Dependencies")
    print("-" * 30)

    # Check packages
    packages_ok, missing = check_required_packages()

    if not packages_ok:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        install_choice = input(
            "Install missing packages? (y/N): ").strip().lower()

        if install_choice == 'y':
            if install_requirements():
                packages_ok = True
            else:
                success = False
        else:
            success = False

    print("\n🏗️ Setting up Project Structure")
    print("-" * 30)
    setup_directories()
    create_sample_env_file()

    print("\n🔑 API Configuration")
    print("-" * 30)
    api_ok = setup_api_key()

    if api_ok:
        print("\n🔌 Testing API Connection")
        print("-" * 30)
        test_api_connection()

    print("\n📊 Checking Input Data")
    print("-" * 30)
    data_ok = check_input_data()

    if data_ok:
        generate_cost_estimate()

    print("\n" + "=" * 60)

    if success and packages_ok and data_ok:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in environment variables")
        print("2. Run: python scripts/run_labeling.py --dry-run")
        print("3. Run: python scripts/run_labeling.py")
    else:
        print("⚠️ Setup completed with warnings")
        print("\nIssues to resolve:")
        if not packages_ok:
            print("- Install missing Python packages")
        if not api_ok:
            print("- Set up OpenAI API key")
        if not data_ok:
            print("- Ensure input data is available")

    print(f"\nFor help, check the documentation or run with --help")


if __name__ == "__main__":
    main()
