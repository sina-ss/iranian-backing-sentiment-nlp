#!/usr/bin/env python3
"""
Setup script for Persian text preprocessing components
Validates dependencies and prepares external resources
"""

import sys
import subprocess
from pathlib import Path
import json
import requests
from typing import List


def check_hazm_installation():
    """Check if hazm is properly installed"""
    try:
        import hazm
        print("✅ Hazm library installed successfully")

        # Test basic functionality
        normalizer = hazm.Normalizer()
        test_text = "سلام! این یک متن تست است."
        normalized = normalizer.normalize(test_text)
        print(f"✅ Hazm normalization test passed")

        return True
    except ImportError:
        print("❌ Hazm library not found")
        return False
    except Exception as e:
        print(f"⚠️ Hazm installation issue: {e}")
        return False


def install_hazm():
    """Install hazm library"""
    print("📦 Installing hazm library...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "hazm"])
        print("✅ Hazm installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install hazm: {e}")
        return False


def download_persian_stopwords():
    """Download or create Persian stopwords list"""
    stopwords_file = Path("data/external/persian_stopwords.txt")

    if stopwords_file.exists():
        print("✅ Persian stopwords file already exists")
        return True

    # Create comprehensive Persian stopwords list
    persian_stopwords = [
        # Common Persian stopwords
        "در", "به", "از", "که", "این", "آن", "را", "با", "تا", "برای",
        "و", "یا", "اما", "ولی", "چون", "اگر", "وقتی", "زمانی", "هنگامی",
        "است", "بود", "باشد", "شد", "می", "خواهد", "کرد", "نمی", "نخواهد",
        "من", "تو", "او", "ما", "شما", "آنها", "خود", "خودم", "خودت", "خودش",
        "هر", "همه", "بعضی", "چند", "یک", "دو", "سه", "چهار", "پنج",
        "روی", "زیر", "کنار", "نزدیک", "دور", "جلو", "عقب", "بالا", "پایین",
        "خیلی", "بسیار", "کمی", "اندکی", "زیاد", "کم", "فقط", "هم", "نیز",
        "البته", "مطمئناً", "احتمالاً", "شاید", "باید", "نباید", "باز", "دوباره",

        # Banking and app specific stopwords
        "بانک", "اپ", "اپلیکیشن", "موبایل", "سیستم", "وب", "سایت",
        "برنامه", "نرم افزار", "دانلود", "نصب", "بروزرسانی", "آپدیت",
        "کارت", "رمز", "پسورد", "حساب", "اکانت", "لاگین", "ورود", "خروج",

        # Common filler words
        "آره", "بله", "نه", "خب", "اوکی", "باشه", "چشم", "مرسی", "ممنون",
        "سلام", "خداحافظ", "درست", "غلط", "صحیح", "نادرست"
    ]

    # Ensure directory exists
    stopwords_file.parent.mkdir(parents=True, exist_ok=True)

    # Write stopwords to file
    with open(stopwords_file, 'w', encoding='utf-8') as f:
        for word in persian_stopwords:
            f.write(word + '\n')

    print(
        f"✅ Created Persian stopwords file with {len(persian_stopwords)} words")
    return True


def validate_emoji_mapping():
    """Validate emoji mapping file"""
    emoji_file = Path("data/external/persian_emoji_mapping.json")

    if not emoji_file.exists():
        print("❌ Emoji mapping file not found")
        return False

    try:
        with open(emoji_file, 'r', encoding='utf-8') as f:
            emoji_data = json.load(f)

        # Count total emojis
        total_emojis = 0
        for category in emoji_data.values():
            if isinstance(category, dict):
                total_emojis += len(category)

        print(f"✅ Emoji mapping file validated with {total_emojis} emojis")
        return True

    except Exception as e:
        print(f"❌ Error validating emoji mapping: {e}")
        return False


def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample text"""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))

        from src.preprocessing.persian_cleaner import PersianTextCleaner

        # Initialize cleaner
        cleaner = PersianTextCleaner()

        # Test texts
        test_texts = [
            "سلام! اپ بانک عالیه 😊 خیلی راحت کار میکنه",
            "این برنامه خراب است 😡 مشکل داره",
            "www.example.com شماره تماس: 09123456789",
            "نرم‌افزار خوبی است ولی باید بهتر بشه"
        ]

        print("🧪 Testing preprocessing pipeline...")

        for i, text in enumerate(test_texts, 1):
            try:
                # Test different cleaning levels
                light = cleaner.clean_text(text, level='light')
                medium = cleaner.clean_text(text, level='medium')
                heavy = cleaner.clean_text(text, level='heavy')

                # Test tokenization
                tokens = cleaner.tokenize(medium)

                print(f"✅ Test {i} passed")

            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                return False

        # Get statistics
        stats = cleaner.get_statistics()
        print(
            f"✅ Pipeline test completed. Processed {stats['processing_stats']['processed_texts']} texts")

        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def create_sample_data():
    """Create sample data for testing"""
    sample_file = Path("data/raw/sample_comments.csv")

    if sample_file.exists():
        print("✅ Sample data file already exists")
        return True

    # Sample Persian banking comments
    sample_data = [
        {
            'id': 1,
            'app_name': 'Test Bank',
            'comment': 'اپلیکیشن عالی است! 😊 خیلی راحت استفاده می‌شود',
            'rating': 5
        },
        {
            'id': 2,
            'app_name': 'Test Bank',
            'comment': 'برنامه مشکل دارد 😠 نمی‌تونم وارد بشم',
            'rating': 1
        },
        {
            'id': 3,
            'app_name': 'Another Bank',
            'comment': 'سرعت خوبی داره ولی رابط کاربری بهتر باشه بهتره',
            'rating': 3
        },
        {
            'id': 4,
            'app_name': 'Test Bank',
            'comment': 'کارت به کارت راحت شده 👍 ممنون از تیم برنامه نویسی',
            'rating': 4
        },
        {
            'id': 5,
            'app_name': 'Another Bank',
            'comment': 'چرا اینقدر کند است؟ لطفاً بهش رسیدگی کنید',
            'rating': 2
        }
    ]

    import pandas as pd

    # Ensure directory exists
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    df.to_csv(sample_file, index=False, encoding='utf-8')

    print(f"✅ Created sample data file with {len(sample_data)} comments")
    return True


def main():
    """Main setup function"""
    print("🚀 Persian Text Preprocessing Setup")
    print("=" * 50)

    success = True

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        success = False
    else:
        print(f"✅ Python {sys.version.split()[0]} detected")

    print("\n📦 Checking Dependencies")
    print("-" * 30)

    # Check hazm installation
    if not check_hazm_installation():
        install_choice = input("Install hazm library? (y/N): ").strip().lower()
        if install_choice == 'y':
            if not install_hazm():
                success = False
        else:
            print("⚠️ Hazm is required for optimal Persian text processing")

    print("\n📁 Setting up External Resources")
    print("-" * 30)

    # Setup stopwords
    if not download_persian_stopwords():
        success = False

    # Validate emoji mapping
    if not validate_emoji_mapping():
        print("⚠️ Emoji mapping file missing. Please ensure it exists at data/external/persian_emoji_mapping.json")

    print("\n🧪 Testing Components")
    print("-" * 30)

    # Test preprocessing pipeline
    if not test_preprocessing_pipeline():
        success = False

    print("\n📝 Creating Sample Data")
    print("-" * 30)

    # Create sample data for testing
    if not create_sample_data():
        success = False

    print("\n" + "=" * 50)

    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/preprocessing_pipeline.py --input data/raw/sample_comments.csv")
        print("2. Check results in data/processed/")
        print("3. Analyze results with notebooks/02_preprocessing_analysis.ipynb")
    else:
        print("⚠️ Setup completed with warnings")
        print("Please resolve the issues above before proceeding")

    print(f"\nFor detailed preprocessing, run:")
    print(f"python scripts/preprocessing_pipeline.py --help")


if __name__ == "__main__":
    main()
